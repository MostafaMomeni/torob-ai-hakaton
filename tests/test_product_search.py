from shopping_assistant_data.product_search import ProductSearcher, run_structured_search
from shopping_assistant_data.repository import DataRepository
from shopping_assistant_data.text_utils import normalize_persian_text, tokenize_persian


def setup_repo():
    repo = DataRepository()
    repo.load_all()
    return repo


def stub_repo(titles):
    """Build an in-memory DataRepository from product titles (no CSV loading).

    Each product gets one offer so run_structured_search's price/offer handling works.
    """
    from shopping_assistant_data.schemas import Offer, Product

    repo = DataRepository()
    for i, title in enumerate(titles):
        pid = f'p{i:03d}'
        repo._products[pid] = Product(
            product_id=pid,
            persian_title=title,
            brand_name=None,
            category_name=None,
            model=None,
            image_urls=None,
            specs=None,
            data_observed_at=None,
        )
        repo._offers[f'o{i:03d}'] = Offer(
            offer_id=f'o{i:03d}',
            product_id=pid,
            shop_id='s1',
            price_toman=1000 + i,
            last_price_changed_at=None,
            seller_title=None,
            warranty=None,
            data_observed_at=None,
        )
    return repo


def test_full_phrase_ranks_above_token_match():
    repo = setup_repo()
    ps = ProductSearcher(repo)
    # query that should match exact product title in dataset
    results = run_structured_search(repo, {'product': {'name': 'صفحه مسی'}}, limit=20)
    # Ensure results deduped and ordered
    assert isinstance(results, list)
    # If dataset contains exact matches, first result's title should contain the full phrase
    if results:
        first_title = (results[0]['persian_title'] or '').lower()
        assert 'صفحه مسی' in normalize_persian_text(first_title)


def test_ordered_phrase_preference_over_single_token():
    repo = setup_repo()
    ps = ProductSearcher(repo)
    results = run_structured_search(repo, {'product': {'name': 'صفحه مسی ضخیم'}}, limit=20)
    # If a product with full phrase exists, it should be top
    if results:
        ft = normalize_persian_text(results[0]['persian_title'] or '')
        assert 'صفحه مسی' in ft or 'صفحه مسی ضخیم' in ft


def test_token_coverage_prefers_two_tokens_over_one():
    repo = setup_repo()
    results = run_structured_search(repo, {'product': {'name': 'صفحه نمایش'}}, limit=50)
    # Ensure that products matching both tokens rank above those matching only one token
    if len(results) >= 2:
        top_tokens = results[0]['match_record']['matched_tokens']
        second_tokens = results[1]['match_record']['matched_tokens']
        assert len(top_tokens) >= len(second_tokens)


def test_persian_arabic_variants():
    repo = setup_repo()
    # Use Arabic variants
    results_ar = run_structured_search(repo, {'product': {'name': 'صفحه مسي'}}, limit=20)
    results_pr = run_structured_search(repo, {'product': {'name': 'صفحه مسی'}}, limit=20)
    # Their top result product ids should be equal if both found
    if results_ar and results_pr:
        assert results_ar[0]['product_id'] == results_pr[0]['product_id']


def test_deduplication_of_stages():
    repo = setup_repo()
    results = run_structured_search(repo, {'product': {'name': 'صفحه مسی'}}, limit=200)
    ids = [r['product_id'] for r in results]
    assert len(ids) == len(set(ids))


def test_price_constraints_enforced():
    repo = setup_repo()
    # pick a price window that's tiny to make some offers excluded
    spec = {'product': {'name': 'صفحه'}, 'price': {'min': 1, 'max': 1000}}
    results = run_structured_search(repo, spec, limit=50)
    # All returned offers must satisfy price constraints
    for r in results:
        for o in r.get('offers', []):
            if o.get('price_toman') is not None:
                assert 1 <= o.get('price_toman') <= 1000


def test_no_results_returns_empty():
    repo = setup_repo()
    results = run_structured_search(repo, {'product': {'name': 'عدم وجود کالا عجیبی'}})
    assert isinstance(results, list)
    assert results == []


def test_price_cannot_override_stronger_name_match():
    repo = setup_repo()
    results = run_structured_search(repo, {'product': {'name': 'صفحه مسی'}}, limit=20)
    # Relevance-first ranking: tiers must be lexicographically ascending, so a weaker
    # name match is never promoted above a stronger one — price is not a ranking key.
    tiers = [r['match_record']['tier'] for r in results]
    assert tiers == sorted(tiers)
    # Direct spot-check: if the second result is a strictly weaker name match, it stays
    # below the top result even when it happens to be cheaper.
    if (
        len(results) >= 2
        and results[0]['match_record']['tier'] < results[1]['match_record']['tier']
    ):
        def min_price(r):
            prices = [
                o['price_toman']
                for o in r.get('offers', [])
                if o.get('price_toman') is not None
            ]
            return min(prices) if prices else None

        top_p = min_price(results[0])
        second_p = min_price(results[1])
        if top_p is not None and second_p is not None:
            assert second_p <= top_p, 'cheaper weaker name match must not outrank'


def test_fallback_to_shorter_phrases_and_tokens_when_no_exact_match():
    repo = setup_repo()
    ps = ProductSearcher(repo)
    # Take two real tokens from a product title and append a nonsense token so the
    # full phrase cannot exist as a contiguous substring anywhere in the dataset.
    sample = next(
        p
        for p in repo._products.values()
        if len(tokenize_persian(normalize_persian_text(p.persian_title or ''))) >= 2
    )
    toks = tokenize_persian(normalize_persian_text(sample.persian_title or ''))[:2]
    query = ' '.join(toks + ['xzzq'])
    assert ps._stage1_full_phrase(query) == []
    # Fallback stages (ordered n-grams / individual tokens) must still find candidates.
    cands = ps.retrieve_candidates(product_name=query, limit=20)
    assert cands, 'fallback must retrieve candidates via shorter phrases or tokens'
    stage_nums = {s[0] for c in cands for s in c['match_record'].stages}
    assert stage_nums & {2, 3}, 'fallback must use stage-2 (n-gram) or stage-3 (token) matches'


# ---------------------------------------------------------------------------
# Priority-ladder tests (stub data): the whole user query must outrank its pieces,
# and pieces must outrank individual tokens.
# ---------------------------------------------------------------------------

def test_full_query_phrase_outranks_its_pieces():
    # User example: «سیم مسی» — the exact full phrase first, then in-order/token pieces.
    repo = stub_repo([
        'سیم مسی نمره 240',          # starts with the full phrase (tier 2 + prefix) -> #1
        'کابل اتو دو سیم مسی مرغوب',  # full phrase in the middle of the title (tier 2) -> #2
        'سیم روکشدار مسی نمره 240',  # query words in order with filler (tier 3)
        'مسی آلیاژ نمره 5',         # only one query token -> not a candidate
        'سیم پلاستیکی نمره 2',      # only one query token -> not a candidate
    ])
    results = run_structured_search(repo, {'product': {'name': 'سیم مسی'}}, limit=10)
    titles = [r['persian_title'] for r in results]
    assert titles[0] == 'سیم مسی نمره 240'
    assert titles[1] == 'کابل اتو دو سیم مسی مرغوب'
    tiers = {r['persian_title']: r['match_record']['tier'] for r in results}
    assert tiers['سیم مسی نمره 240'] == 2
    assert tiers['کابل اتو دو سیم مسی مرغوب'] == 2
    assert tiers['سیم روکشدار مسی نمره 240'] == 3
    assert 'مسی آلیاژ نمره 5' not in titles[:3]
    assert 'سیم پلاستیکی نمره 2' not in titles[:3]


def test_in_order_gap_ranks_the_exact_item_first():
    # «سیم مسی 240» — "سیم مسی نمره 240" is the exact item (one filler word); it must
    # beat "سیم افشان ارت مسی 240" (two filler words), the smaller piece match, and
    # the token-only product.
    repo = stub_repo([
        'سیم مسی نمره 240',
        'سیم افشان ارت مسی 240',
        'سیم مسی نمره 185',
        'کابل سیم تلفنی 240',
    ])
    results = run_structured_search(repo, {'product': {'name': 'سیم مسی 240'}}, limit=10)
    titles = [r['persian_title'] for r in results]
    assert titles[0] == 'سیم مسی نمره 240'
    assert titles.index('سیم افشان ارت مسی 240') == 1
    # a product matching a piece of the query outranks one matching only tokens
    assert titles.index('سیم مسی نمره 185') < titles.index('کابل سیم تلفنی 240')


def test_whole_query_words_in_order_beat_partial_pieces():
    # «کابل سیم مسی» — a product containing all query words in order must beat
    # products that only match a 2-word piece like «کابل سیم» (telephone cables).
    repo = stub_repo([
        'کابل اتو دو سیم مسی مرغوب',  # all query words in order (tier 3)
        'کابل سیم تلفنی',           # matches only the 2-word piece (tier 4)
        'قرقره حمل کابل سیم بکسل',    # matches only the 2-word piece (tier 4)
    ])
    results = run_structured_search(repo, {'product': {'name': 'کابل سیم مسی'}}, limit=10)
    tiers = {r['persian_title']: r['match_record']['tier'] for r in results}
    assert results[0]['persian_title'] == 'کابل اتو دو سیم مسی مرغوب'
    assert tiers['کابل اتو دو سیم مسی مرغوب'] < tiers['کابل سیم تلفنی']
    assert tiers['کابل سیم تلفنی'] == 4


def test_single_tokens_rank_below_n_gram_pieces():
    # Individual word matches must be at clearly lower priority than phrase pieces.
    repo = stub_repo([
        'سیم مسی نمره 185',  # piece "سیم مسی" of the query (tier 4)
        'مسی استیل 240',     # two query tokens, no phrase (tier 5)
    ])
    results = run_structured_search(repo, {'product': {'name': 'سیم مسی 240'}}, limit=10)
    tiers = {r['persian_title']: r['match_record']['tier'] for r in results}
    assert results[0]['persian_title'] == 'سیم مسی نمره 185'
    assert tiers['سیم مسی نمره 185'] == 4
    assert tiers['مسی استیل 240'] == 5
    assert tiers['سیم مسی نمره 185'] < tiers['مسی استیل 240']
