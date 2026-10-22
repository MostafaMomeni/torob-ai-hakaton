import json
import logging
from typing import Optional, Dict, Any
from llm_adapter import LLMAdapter, LLMError
from search_spec import SearchSpec

logger = logging.getLogger('shopping_assistant.search_extractor')

SEARCH_SYSTEM = '''You are a JSON extractor. Given a Persian user instruction about a shopping request, extract a SearchSpec JSON matching the project's SearchSpec pydantic schema. The JSON must follow the schema exactly and use null for unknowns. Use currency IRR and prices in toman integers when possible. Output only the JSON object.'''


def _normalize_llm_search_output(parsed: Any) -> Dict[str, Any]:
    """Accept both the project's nested SearchSpec shape and common flat LLM JSON."""
    if not isinstance(parsed, dict):
        return {}
    # Native nested output must be preserved; the previous implementation discarded it.
    if any(key in parsed for key in ("product", "price", "location", "search_options", "conversation")):
        out = {k: parsed[k] for k in ("product", "price", "location", "search_options", "conversation") if k in parsed}
        if "product" not in out:
            out["product"] = {}
        if "price" not in out:
            out["price"] = {}
        if "location" not in out:
            out["location"] = {}
        if "search_options" not in out:
            out["search_options"] = {}
        return out

    product = {}
    name = parsed.get("query") or parsed.get("product_name") or parsed.get("name")
    if name:
        product["name"] = name
    for key in ("brand", "model", "keywords"):
        if parsed.get(key) is not None:
            product[key] = parsed[key]
    attrs = parsed.get("attributes") or parsed.get("features") or parsed.get("product_attributes")
    if isinstance(attrs, list):
        product["features"] = [
            {"name": a, "operator": "eq", "value": True} if isinstance(a, str)
            else a for a in attrs if isinstance(a, str) or (isinstance(a, dict) and a.get("name"))
        ]
    price = {}
    if parsed.get("min_price") is not None:
        price["min"] = parsed["min_price"]
    if parsed.get("max_price") is not None:
        price["max"] = parsed["max_price"]
    if parsed.get("currency"):
        price["currency"] = parsed["currency"]
    raw_location = parsed.get("location")
    location = {}
    if isinstance(raw_location, dict):
        location = {"city": raw_location.get("city") or raw_location.get("text"),
                    "latitude": raw_location.get("latitude"), "longitude": raw_location.get("longitude"),
                    "radius_km": raw_location.get("radius_km")}
        location = {k: v for k, v in location.items() if v is not None}
    elif isinstance(raw_location, str) and raw_location.strip():
        location["city"] = raw_location.strip()
    options = {}
    if parsed.get("sort_by"):
        options["sort_by"] = parsed["sort_by"]
    if parsed.get("limit") is not None:
        try:
            options["limit"] = max(1, int(parsed["limit"]))
        except (TypeError, ValueError):
            pass
    out = {"product": product, "price": price, "location": location, "search_options": options}
    if parsed.get("conversation_id"):
        out["conversation"] = {"session_id": parsed["conversation_id"]}
    return out


def extract_search_spec(text: str, context: Optional[Dict[str, Any]] = None, adapter: Optional[LLMAdapter] = None, max_attempts: int = 2) -> SearchSpec:
    adapter = adapter or LLMAdapter()
    context = context or {}
    attempts = 0
    last_err = None
    while attempts < max_attempts:
        attempts += 1
        messages = [
            {"role": "system", "content": SEARCH_SYSTEM},
            {"role": "user", "content": f"User input: {text}\nContext: {json.dumps(context, ensure_ascii=False)}\nReturn the SearchSpec JSON."}
        ]
        try:
            raw = adapter.generate(messages, max_tokens=1024)
        except LLMError as e:
            logger.exception('LLM call failed for search_spec')
            raise

        try:
            parsed = json.loads(raw)
        except Exception as e:
            logger.warning('LLM returned non-JSON for search_spec on attempt %d: %s', attempts, raw[:400])
            last_err = e
            if attempts < max_attempts:
                followup_msg = [{"role": "system", "content": SEARCH_SYSTEM}, {"role": "user", "content": "Previous response was not valid JSON. Return only the valid JSON object."}]
                try:
                    raw = adapter.generate(followup_msg, max_tokens=1024)
                    parsed = json.loads(raw)
                except Exception as e2:
                    logger.warning('Retry JSON parsing failed for search_spec: %s', e2)
                    last_err = e2
                    continue
            else:
                raise LLMError('LLM did not return valid JSON for search_spec')

        # If LLM returned a flat or variant schema, normalize into SearchSpec-like dict
        try:
            normalized = _normalize_llm_search_output(parsed)
            # Merge normalized fields into a base spec dict to validate
            base_spec = {}
            base_spec.update(normalized)
            # Ensure required sub-objects exist so pydantic can populate defaults
            from search_spec import ProductSpec, PriceSpec, LocationSpec, SearchOptionsSpec
            if "product" not in base_spec:
                base_spec["product"] = {}
            if "price" not in base_spec:
                base_spec["price"] = {}
            if "location" not in base_spec:
                base_spec["location"] = {}
            if "search_options" not in base_spec:
                base_spec["search_options"] = {}

            # Validate via SearchSpec constructor (this will enforce strict fields)
            spec = SearchSpec(**base_spec)
            return spec
        except Exception as e:
            logger.warning('SearchSpec validation failed: %s; parsed keys: %s', e, list(parsed.keys()) if isinstance(parsed, dict) else type(parsed))
            last_err = e
            if attempts < max_attempts:
                hint = json.dumps(parsed, ensure_ascii=False) if isinstance(parsed, (dict, list)) else str(parsed)
                followup_msg = [{"role": "system", "content": SEARCH_SYSTEM}, {"role": "user", "content": f"The previous JSON did not validate. Here is what you returned: {hint}. Please return corrected JSON that matches the SearchSpec schema exactly."}]
                try:
                    raw = adapter.generate(followup_msg, max_tokens=1024)
                    parsed = json.loads(raw)
                except Exception as e2:
                    logger.warning('Retry after validation failed for search_spec: %s', e2)
                    last_err = e2
                    continue
                try:
                    normalized = _normalize_llm_search_output(parsed)
                    base_spec = {}
                    base_spec.update(normalized)
                    if "product" not in base_spec:
                        base_spec["product"] = {}
                    if "price" not in base_spec:
                        base_spec["price"] = {}
                    if "location" not in base_spec:
                        base_spec["location"] = {}
                    if "search_options" not in base_spec:
                        base_spec["search_options"] = {}
                    spec = SearchSpec(**base_spec)
                    return spec
                except Exception as e3:
                    logger.warning('Second validation failed for search_spec: %s', e3)
                    last_err = e3
                    continue
            else:
                break
    raise LLMError(f'search_spec extraction failed after {attempts} attempts: {last_err}')
