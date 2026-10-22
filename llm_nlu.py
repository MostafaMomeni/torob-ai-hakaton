from typing import Optional, Dict, Any
import logging
import json

from llm_adapter import LLMAdapter, LLMError
from nlu_schema import NLUResult

logger = logging.getLogger('shopping_assistant.nlu')


# System prompt tailored to produce strictly JSON output following our schema.
SYSTEM_PROMPT = '''You are an NLU component for a Persian (Farsi) shopping assistant.

You MUST output exactly one JSON object and nothing else.
Follow this schema (keys): schema_version, language, user_goal, entities, filters, context_reference, missing_information, clarification_question, confidence, raw_text.

- language should be "fa" for Persian.
- The NLU should NOT emit an "intent" field. All user inputs must be interpreted as product search queries. Do not output any intent or intent-like values.
- entities must include product_name, brand, category, model, features (array), location (object with text, latitude, longitude, radius_km).
- filters must include min_price, max_price, currency, brands, categories, features, sort_by.
- context_reference must include previous_result_ids (array), selected_product_id, selected_store_id.
- missing_information should be an array of strings indicating which pieces are missing for performing a product search (e.g., 'product_name', 'budget').
- clarification_question should be null or a short Persian question when asking the user for info.
- confidence should be a float 0.0-1.0.
- raw_text should contain the original user input string (for debugging) but must NOT contain PII.

Rules:
1) Output must be valid JSON parsable by a strict JSON parser. No surrounding markdown, comments, or text.
2) Use null for unknown values and empty arrays for empty lists.
3) Do not invent numeric budgets or coordinates. If unspecified, leave them null.
4) When ambiguous, do not output intent; instead populate missing_information and provide a short Persian clarification_question.
5) Preserve order of important fields: schema_version, language, user_goal, entities, filters, context_reference, missing_information, clarification_question, confidence, raw_text.

Respond only with JSON.'''


# A wrapper function that takes user text and optional context and returns validated NLUResult

def analyze_request(text: str, context: Optional[Dict[str, Any]] = None, adapter: Optional[LLMAdapter] = None, max_attempts: int = 2) -> NLUResult:
    """Analyze a user request (Persian) using the LLM adapter and return a validated NLUResult.

    Will attempt to parse the LLM output as JSON and validate against NLUResult. If the LLM returns invalid JSON,
    it will retry up to max_attempts times asking the model to correct to valid JSON. Raises LLMError on failure.
    """
    adapter = adapter or LLMAdapter()
    context = context or {}

    attempts = 0
    last_err = None
    while attempts < max_attempts:
        attempts += 1
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"User input: {text}\nContext: {json.dumps(context, ensure_ascii=False)}\nReturn the NLU JSON as specified."}
        ]
        try:
            raw = adapter.generate(messages, max_tokens=1024)
        except LLMError as e:
            logger.exception('LLM call failed')
            raise

        # Try to load JSON
        try:
            parsed = json.loads(raw)
        except Exception as e:
            logger.warning('LLM returned non-JSON on attempt %d: %s', attempts, raw[:400])
            last_err = e
            # prepare follow-up instruction to the model to correct output
            if attempts < max_attempts:
                # ask for pure JSON only
                followup_msg = [{"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": "The previous response was not valid JSON. Please return only the valid JSON object conforming to the schema, nothing else."}]
                try:
                    raw = adapter.generate(followup_msg, max_tokens=1024)
                    parsed = json.loads(raw)
                except Exception as e2:
                    logger.warning('Retry JSON parsing failed: %s', e2)
                    last_err = e2
                    continue
            else:
                raise LLMError('LLM did not return valid JSON')

        # Now validate against Pydantic model
        try:
            # Remove any intent key if model mistakenly included it
            if isinstance(parsed, dict) and 'intent' in parsed:
                del parsed['intent']
            nlu = NLUResult(**parsed)
            # attach raw_text safely
            nlu.raw_text = text
            return nlu
        except Exception as e:
            logger.warning('NLU schema validation failed: %s; parsed keys: %s', e, list(parsed.keys()) if isinstance(parsed, dict) else type(parsed))
            last_err = e
            # If we still have attempts left, ask model to correct to match schema (provide the parsed object as hint)
            if attempts < max_attempts:
                hint = json.dumps(parsed, ensure_ascii=False) if isinstance(parsed, (dict, list)) else str(parsed)
                followup_msg = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"The previous JSON did not validate against the schema. Here is what you returned: {hint}. Please return a corrected JSON that matches the schema exactly, with keys in the required order."}
                ]
                try:
                    raw = adapter.generate(followup_msg, max_tokens=1024)
                    parsed = json.loads(raw)
                except Exception as e2:
                    logger.warning('Retry after schema validation failed: %s', e2)
                    last_err = e2
                    continue
                try:
                    if isinstance(parsed, dict) and 'intent' in parsed:
                        del parsed['intent']
                    nlu = NLUResult(**parsed)
                    nlu.raw_text = text
                    return nlu
                except Exception as e3:
                    logger.warning('Second validation failed: %s', e3)
                    last_err = e3
                    continue
            else:
                break

    raise LLMError(f'NLU analyze_request failed after {attempts} attempts: {last_err}')
