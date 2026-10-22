import json
import pytest

from shopping_assistant_data.llm_nlu import analyze_request
from shopping_assistant_data.nlu_schema import NLUResult
from shopping_assistant_data.llm_nlu import SYSTEM_PROMPT
from shopping_assistant_data.llm_adapter import LLMError


class MockAdapter:
    def __init__(self, responses):
        # responses: list of raw strings to return sequentially
        self._responses = list(responses)
        self.calls = 0

    def generate(self, messages, max_tokens=1024):
        self.calls += 1
        if not self._responses:
            return ''
        return self._responses.pop(0)


def valid_search_response():
    return json.dumps({
        "schema_version": "1.0",
        "language": "fa",
        "user_goal": "پیدا کردن گوشی سامسونگ ارزان",
        "entities": {
            "product_name": "گوشی",
            "brand": "سامسونگ",
            "category": "موبایل",
            "model": None,
            "features": [],
            "location": {"text": None, "latitude": None, "longitude": None, "radius_km": None}
        },
        "filters": {"min_price": None, "max_price": None, "currency": None, "brands": ["سامسونگ"], "categories": ["موبایل"], "features": [], "sort_by": "relevance"},
        "context_reference": {"previous_result_ids": [], "selected_product_id": None, "selected_store_id": None},
        "missing_information": [],
        "clarification_question": None,
        "confidence": 0.85,
        "raw_text": "یه گوشی سامسونگ ارزون می‌خوام"
    }, ensure_ascii=False)


def invalid_text():
    return "I am not returning JSON. Here is text. {not: valid}"


def test_analyze_request_valid_direct():
    adapter = MockAdapter([valid_search_response()])
    nlu = analyze_request("یه گوشی سامسونگ ارزون می‌خوام", adapter=adapter)
    assert isinstance(nlu, NLUResult)
    # intent removed — verify NLU parsed brand and language
    assert nlu.entities.brand == 'سامسونگ'
    assert nlu.language == 'fa'
    assert nlu.raw_text == 'یه گوشی سامسونگ ارزون می\u200cخوام' or nlu.raw_text.startswith('یه گوشی سامسونگ')


def test_analyze_request_invalid_then_repair():
    # first response invalid text, second valid JSON
    adapter = MockAdapter([invalid_text(), valid_search_response()])
    nlu = analyze_request("یه گوشی سامسونگ ارزون می‌خوام", adapter=adapter, max_attempts=2)
    assert isinstance(nlu, NLUResult)
    # intent removed — ensure we repaired and parsed
    assert nlu.entities.brand == 'سامسونگ'
    assert adapter.calls >= 2


def test_analyze_request_invalid_twice_raises():
    adapter = MockAdapter([invalid_text(), "still not json"])
    with pytest.raises(LLMError):
        analyze_request("چه چیزایی دارید؟", adapter=adapter, max_attempts=2)
