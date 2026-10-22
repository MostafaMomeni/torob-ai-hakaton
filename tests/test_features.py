from shopping_assistant_data.orchestrator_features import OrchestratorFeatures
from shopping_assistant_data.conversation_state import ConversationState


def test_compare_product():
    of = OrchestratorFeatures()
    # pick any product id from repo
    prod_ids = list(of.repo._products.keys())
    if not prod_ids:
        assert False, 'no products in dataset'
    res = of.compare_product(prod_ids[0])
    assert 'offers' in res


def test_find_substitutes():
    of = OrchestratorFeatures()
    prod_ids = list(of.repo._products.keys())
    if len(prod_ids) < 2:
        assert False, 'not enough products'
    res = of.find_substitutes(prod_ids[0])
    assert 'substitutes' in res


def test_plan_visit():
    of = OrchestratorFeatures()
    prod_ids = list(of.repo._products.keys())[:3]
    res = of.plan_visit(prod_ids, user_location=None)
    assert 'plan' in res
