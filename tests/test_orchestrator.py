from shopping_assistant_data.orchestrator import Orchestrator
from shopping_assistant_data.conversation_state import ConversationState


def test_search_fallback():
    orch = Orchestrator()
    state = ConversationState(conversation_id='t1')
    res = orch.handle_request('لطفا آیفون های موجود رو پیدا کن', state)
    assert isinstance(res, dict)
    assert 'text' in res


def test_apply_cheaper_filter():
    orch = Orchestrator()
    state = ConversationState(conversation_id='t2')
    # simulate previous results
    state.previous_results['products'] = [{'product_id': 'p1', 'persian_title': 'نمونه', 'offers': [{'offer_id': 'o1', 'price_toman': 5000}, {'offer_id': 'o2', 'price_toman': 4000}]}]
    res = orch.handle_request('ارزون‌ترها رو نشون بده', state)
    assert res['status'] in ('success', 'partial')
    assert 'products' in res


def test_location_filter():
    orch = Orchestrator()
    state = ConversationState(conversation_id='t3')
    res = orch.handle_request('فقط اطراف تجریش', state)
    assert 'stores' in res
    assert isinstance(res['stores'], list)
