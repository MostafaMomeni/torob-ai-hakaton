import pytest
from llm_tools import LLMTolls, ToolExecutor
from orchestrator import Orchestrator
from repository import DataRepository
from pathlib import Path
from schemas import Product, Offer, Shop
from datetime import datetime


def make_stub_repo():
    class StubRepo:
        def __init__(self):
            self._products = {}
            self._shops = {}
            self._offers = {}
            # seed 3 products
            p1 = Product(product_id='p1', persian_title='گوشی مدل A', brand_name='برندX', category_name='موبایل', model=None, image_urls=[], specs=[], data_observed_at=datetime.utcnow())
            p2 = Product(product_id='p2', persian_title='گوشی مدل B', brand_name='برندY', category_name='موبایل', model=None, image_urls=[], specs=[], data_observed_at=datetime.utcnow())
            p3 = Product(product_id='p3', persian_title='هدفون مدل Z', brand_name='برندX', category_name='هدفون', model=None, image_urls=[], specs=[], data_observed_at=datetime.utcnow())
            self._products = {p.product_id: p for p in [p1, p2, p3]}
            s1 = Shop(shop_id='s1', name='فروشگاه مرکزی', address='تهران', lat=35.7, lon=51.4, location_label='تهران', category_names=[], contacts=[], website_url=None, weekly_hours=None, data_observed_at=datetime.utcnow())
            s2 = Shop(shop_id='s2', name='فروشگاه شمال', address='شمال تهران', lat=35.8, lon=51.4, location_label='شمال تهران', category_names=[], contacts=[], website_url=None, weekly_hours=None, data_observed_at=datetime.utcnow())
            self._shops = {s1.shop_id: s1, s2.shop_id: s2}
            o1 = Offer(offer_id='o1', product_id='p1', shop_id='s1', price_toman=5000000, last_price_changed_at=datetime.utcnow(), seller_title=None, warranty=None, data_observed_at=datetime.utcnow())
            o2 = Offer(offer_id='o2', product_id='p1', shop_id='s2', price_toman=4900000, last_price_changed_at=datetime.utcnow(), seller_title=None, warranty=None, data_observed_at=datetime.utcnow())
            o3 = Offer(offer_id='o3', product_id='p2', shop_id='s1', price_toman=7000000, last_price_changed_at=datetime.utcnow(), seller_title=None, warranty=None, data_observed_at=datetime.utcnow())
            self._offers = {o.offer_id: o for o in [o1, o2, o3]}

        def load_all(self):
            return

        def get_product(self, pid):
            return self._products.get(pid)

        def get_offers_by_product(self, pid):
            return [o for o in self._offers.values() if o.product_id == pid]

        def get_offers_by_shop(self, sid):
            return [o for o in self._offers.values() if o.shop_id == sid]

        def get_shop(self, shop_id):
            return self._shops.get(shop_id)

        def validate_references(self):
            return {"missing_products": [], "missing_shops": []}

        def find_duplicate_shops(self):
            return []

        def shops_with_valid_coords(self):
            return list(self._shops.values())

    return StubRepo()


@pytest.fixture
def tools():
    repo = make_stub_repo()
    t = LLMTolls(repo=repo)
    return t


def test_search_products_basic(tools):
    r = tools.search_products({'query': 'گوشی'})
    assert r['ok'] and isinstance(r['result'], list)
    assert any('p1' in x['product_id'] for x in r['result'])


def test_search_with_budget_and_brand(tools):
    # validate filter input via update_search_filters
    r = tools.update_search_filters({'filters': {'max_price': 6000000, 'brands': ['برندX']}})
    assert r['ok']
    # search with explicit product ids
    r2 = tools.search_products({'product_ids': ['p1']})
    assert r2['ok'] and len(r2['result']) == 1


def test_modify_filters_invalid(tools):
    r = tools.update_search_filters({'filters': {'max_price': -5}})
    assert not r['ok']


def test_compare_two_offers(tools):
    r = tools.compare_offers({'offer_ids': ['o1', 'o2']})
    assert r['ok']
    assert len(r['result']) == 2
    assert r['result'][0]['price_toman'] <= r['result'][1]['price_toman']


def test_find_nearby_stores(tools):
    # location near Tehran
    r = tools.find_nearby_stores({'latitude': 35.7, 'longitude': 51.4, 'radius_km': 5})
    assert r['ok']
    assert len(r['result']) >= 1


def test_find_alternatives(tools):
    r = tools.find_alternatives({'product_id': 'p1'})
    assert r['ok']
    assert any(x['product_id'] == 'p2' for x in r['result'])


def test_empty_result_and_errors(tools):
    r = tools.search_products({'query': 'ناموجودXYZ'})
    assert r['ok'] and r['result'] == []
    r2 = tools.get_product_details({'product_id': 'missing'})
    assert not r2['ok'] and r2['error'] == 'product_not_found'


def test_tool_executor_limits(tools):
    exec = ToolExecutor(tools=tools, max_steps=2)
    a = exec.call('search_products', {'query': 'گوشی'})
    assert a['ok']
    b = exec.call('compare_offers', {'offer_ids': ['o1']})
    assert b['ok']
    c = exec.call('search_products', {'query': 'گوشی'})
    assert not c['ok'] and c['error'] == 'max_steps_exceeded'
