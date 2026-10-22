import json
import logging
from typing import Optional, Dict, Any
from llm_adapter import LLMAdapter, LLMError
from search_spec import SearchSpec

logger = logging.getLogger('shopping_assistant.search_extractor')

SEARCH_SYSTEM = '''You are a JSON extractor. Given a Persian user instruction about a shopping request, extract a SearchSpec JSON matching the project's SearchSpec pydantic schema. The JSON must follow the schema exactly and use null for unknowns. Use currency IRR and prices in toman integers when possible. Output only the JSON object.'''


def _normalize_llm_search_output(parsed: Any) -> Dict[str, Any]:
    """
    Normalize common LLM output shapes into the project's SearchSpec dict shape.

    The LLM may return a flat dict with keys like 'query', 'category', 'min_price', etc.
    We map those into SearchSpec fields conservatively without allowing arbitrary extras.
    """
    if not isinstance(parsed, dict):
        return {}

    out: Dict[str, Any] = {}
    # product
    product = {}
    # prefer explicit product.name/product_name/query
    if parsed.get("query"):
        product["name"] = parsed.get("query")
    elif parsed.get("product_name"):
        product["name"] = parsed.get("product_name")
    if parsed.get("brand"):
        product["brand"] = parsed.get("brand")
    if parsed.get("model"):
        product["model"] = parsed.get("model")
    # attributes/features
    attrs = parsed.get("attributes") or parsed.get("features") or parsed.get("product_attributes")
    if isinstance(attrs, list):
        # convert simple list of strings to FeatureSpec-like dicts
        feats = []
        for a in attrs:
            if isinstance(a, str):
                feats.append({"name": a, "operator": "eq", "value": True})
            elif isinstance(a, dict) and a.get("name"):
                feats.append(a)
        if feats:
            product["features"] = feats

    # keywords
    if parsed.get("keywords") and isinstance(parsed.get("keywords"), list):
        product["keywords"] = parsed.get("keywords")

    out["product"] = product

    # price
    price = {}
    if parsed.get("min_price") is not None:
        price["min"] = parsed.get("min_price")
    if parsed.get("max_price") is not None:
        price["max"] = parsed.get("max_price")
    if parsed.get("currency"):
        price["currency"] = parsed.get("currency")
    if price:
        out["price"] = price

    # location
    loc = {}
    if parsed.get("location") and isinstance(parsed.get("location"), dict):
        loc_mapping = parsed.get("location")
        loc["city"] = loc_mapping.get("text")
        loc["latitude"] = loc_mapping.get("latitude")
        loc["longitude"] = loc_mapping.get("longitude")
        loc["radius_km"] = loc_mapping.get("radius_km")
    elif parsed.get("location") and isinstance(parsed.get("location"), str):
        loc["city"] = parsed.get("location")
    if loc:
        out["location"] = loc

    # search options: sort_by, limit
    search_options = {}
    if parsed.get("sort_by"):
        search_options["sort_by"] = parsed.get("sort_by")
    if parsed.get("limit") is not None:
        try:
            search_options["limit"] = int(parsed.get("limit"))
        except Exception:
            pass
    if search_options:
        out["search_options"] = search_options

    # conversation / context - keep minimal
    conv = {}
    if parsed.get("conversation_id"):
        conv["session_id"] = parsed.get("conversation_id")
    if conv:
        out["conversation"] = conv

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
