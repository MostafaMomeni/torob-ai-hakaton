import pytest
from datetime import datetime
from orchestrator import Orchestrator
from conversation_state import ConversationState
from nlu_schema import NLUResult


def make_nlu(payload: dict) -> NLUResult:
    # ensure schema_version present
    payload.setdefault('schema_version', '1.0')
    payload.setdefault('language', 'fa')
    # ensure minimal keys
    payload.setdefault('user_goal', None)
    payload.setdefault('entities', {
        'product_name': None,
        'brand': None,
        'category': None,
        'model': None,
        'features': [],
        'location': {'text': None, 'latitude': None, 'longitude': None, 'radius_km': None}
    })
    payload.setdefault('filters', {'min_price': None, 'max_price': None, 'currency': None, 'brands': [], 'categories': [], 'features': [], 'sort_by': 'relevance'})
    payload.setdefault('context_reference', {'previous_result_ids': [], 'selected_product_id': None, 'selected_store_id': None})
    payload.setdefault('missing_information', [])
    payload.setdefault('clarification_question', None)
    payload.setdefault('confidence', 0.9)
    payload.setdefault('raw_text', None)
    return NLUResult(**payload)


class SeqNLUMock:
    def __init__(self, responses):
        # responses: list of NLUResult objects to return sequentially
        self._responses = list(responses)

    def __call__(self, text, context=None):
        if not self._responses:
            raise RuntimeError('No more NLU responses configured')
        n = self._responses.pop(0)
        # attach raw_text if missing
        if n.raw_text is None:
            n.raw_text = text
        return n


def test_conversation_filters_and_references(tmp_path):
    repo = None
    orch = Orchestrator(repo)
    state = ConversationState(conversation_id='conv_test')

    # Prepare NLU sequence
    seq = []
    # 1) User: "یه گوشی سامسونگ می‌خوام." -> search product brand samsung
    seq.append(make_nlu({
        'user_goal': 'پیدا کردن گوشی سامسونگ',
        'entities': {'product_name': 'گوشی', 'brand': 'سامسونگ', 'category': 'موبایل', 'model': None, 'features': [], 'location': {'text': None, 'latitude': None, 'longitude': None, 'radius_km': None}},
        'filters': {'min_price': None, 'max_price': None, 'currency': None, 'brands': ['سامسونگ'], 'categories': ['موبایل'], 'features': [], 'sort_by': 'relevance'}
    }))
    # 2) User: "تا ۲۰ میلیون." -> set max_price
    seq.append(make_nlu({
        'filters': {'min_price': None, 'max_price': 20000000, 'currency': 'toman', 'brands': [], 'categories': [], 'features': [], 'sort_by': 'relevance'}
    }))
    # 3) User: "اطراف تجریش باشه." -> set location
    seq.append(make_nlu({
        'entities': {'location': {'text': 'تجریش', 'latitude': None, 'longitude': None, 'radius_km': None}},
    }))
    # 4) User: "حالا ارزون‌ترها رو نشون بده." -> apply cheaper
    seq.append(make_nlu({
        'filters': {'sort_by': 'price_asc'}
    }))
    # 5) User: "حذف بودجه" -> remove price filters (we rely on textual heuristic in orchestrator)
    seq.append(make_nlu({
        'filters': {}
    }))

    orch.analyze_request = SeqNLUMock(seq)

    # TURN 1
    out1 = orch.handle_request('یه گوشی سامسونگ می‌خوام.', state)
    # intent removed: ensure brand was recorded
    assert state.confirmed_info.get('brand') == 'سامسونگ' or 'سامسونگ' in state.active_filters.get('brands', [])

    # TURN 2: add budget
    out2 = orch.handle_request('تا ۲۰ میلیون.', state)
    assert state.active_filters.get('max_price') == 20000000
    # budget persists
    assert 'max_price' in state.active_filters

    # TURN 3: location change
    out3 = orch.handle_request('اطراف تجریش باشه.', state)
    assert state.active_filters.get('location') == 'تجریش'
    # budget should remain
    assert state.active_filters.get('max_price') == 20000000

    # TURN 4: sort cheaper, should set sort_by but not remove budget or location
    out4 = orch.handle_request('حالا ارزون‌ترها رو نشون بده.', state)
    assert state.active_filters.get('sort') == 'price_asc' or state.active_filters.get('sort_by') == 'price_asc'
    assert state.active_filters.get('location') == 'تجریش'
    assert state.active_filters.get('max_price') == 20000000

    # TURN 5: remove budget via textual heuristic
    out5 = orch.handle_request('حذف بودجه', state)
    assert state.active_filters.get('max_price') is None


def test_topic_change_and_session_isolation():
    orch = Orchestrator()
    state_a = ConversationState(conversation_id='A')
    state_b = ConversationState(conversation_id='B')

    # Prepare NLU: A sets budget; B sets brand
    seq_a = [make_nlu({'entities': {'product_name': 'گوشی'}, 'filters': {}}),
             make_nlu({'filters': {'max_price': 15000000}}),
             make_nlu({'entities': {'product_name': 'لپتاپ'}})  # topic change
             ]
    seq_b = [make_nlu({'entities': {'product_name': 'هدفون'}, 'filters': {'brands': ['اپل']}})]

    orch.analyze_request = SeqNLUMock(seq_a + seq_b)

    # session A turn1
    orch.handle_request('یه گوشی می‌خوام', state_a)
    orch.handle_request('تا 15 میلیون', state_a)
    assert state_a.active_filters.get('max_price') == 15000000

    # session B independent
    orch.handle_request('هدفون اپل چی داری؟', state_b)
    assert state_b.confirmed_info.get('product_name') == 'هدفون' or 'اپل' in state_b.confirmed_info.get('brand', '') or 'اپل' in state_b.active_filters.get('brands', [])

    # session A topic change (to laptop) should backup and clear filters
    orch.handle_request('میخوام لپتاپ ببینم', state_a)
    assert state_a.confirmed_info.get('topic_changed') is True or 'prev_filters_backup' in state_a.confirmed_info
    # ensure session B unaffected
    assert state_b.active_filters.get('max_price') is None


def test_reload_state_persistence(tmp_path):
    orch = Orchestrator()
    state = ConversationState(conversation_id='reload_test')
    seq = [make_nlu({'entities': {'product_name': 'گوشی'}}),
           make_nlu({'filters': {'max_price': 12000000}})]
    orch.analyze_request = SeqNLUMock(seq)


    orch.handle_request('یه گوشی', state)
    orch.handle_request('تا 12 میلیون', state)
    # save state
    orch.save_state(state)
    loaded = orch.load_state(state.conversation_id)
    assert loaded is not None
    assert loaded.active_filters.get('max_price') == 12000000


def test_ambiguous_reference_to_previous():
    orch = Orchestrator()
    state = ConversationState(conversation_id='ambig')
    # set previous_results with two products
    state.previous_results['products'] = [{'product_id': 'p1', 'persian_title': 'A'}, {'product_id': 'p2', 'persian_title': 'B'}]

    # NLU returns a neutral follow-up-like response (no intent field)
    seq = [make_nlu({})]
    orch.analyze_request = SeqNLUMock(seq)

    orch.handle_request('همون', state)
    # ambiguous -> missing which_result should be set
    assert 'which_result' in state.missing_information
