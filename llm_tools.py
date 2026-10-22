from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field, validator
from orchestrator import Orchestrator
from repository import DataRepository
from schemas import Product, Offer, Shop
import math


# ---------- Input schemas ----------
class SearchProductsParams(BaseModel):
    query: Optional[str] = None
    product_ids: Optional[List[str]] = None
    limit: int = Field(10, ge=1, le=100)

    @validator('query')
    def non_empty_query(cls, v, values):
        if not v and not values.get('product_ids'):
            raise ValueError('Either query or product_ids must be provided')
        return v


class GetProductDetailsParams(BaseModel):
    product_id: str


class FindProductOffersParams(BaseModel):
    product_id: str
    limit: int = Field(20, ge=1, le=200)


class SearchStoresParams(BaseModel):
    query: Optional[str] = None
    limit: int = Field(10, ge=1, le=100)


class FindNearbyStoresParams(BaseModel):
    latitude: float
    longitude: float
    radius_km: float = Field(..., gt=0, le=500)
    limit: int = Field(20, ge=1, le=500)


class CompareOffersParams(BaseModel):
    offer_ids: Optional[List[str]] = None
    product_ids: Optional[List[str]] = None
    limit: int = Field(10, ge=1, le=200)

    @validator('offer_ids', always=True)
    def at_least_one(cls, v, values):
        if not v and not values.get('product_ids'):
            raise ValueError('offer_ids or product_ids required')
        return v


class FindAlternativesParams(BaseModel):
    product_id: str
    limit: int = Field(5, ge=1, le=50)


class UpdateSearchFiltersParams(BaseModel):
    filters: Dict[str, Any]

    @validator('filters')
    def validate_filters(cls, v):
        # validate sensitive numeric filters
        if 'max_price' in v:
            mp = v['max_price']
            if mp is not None:
                if not isinstance(mp, int) or mp <= 0:
                    raise ValueError('max_price must be positive integer in toman')
        if 'min_price' in v:
            mn = v['min_price']
            if mn is not None:
                if not isinstance(mn, int) or mn < 0:
                    raise ValueError('min_price must be non-negative integer in toman')
        if 'radius_km' in v:
            r = v['radius_km']
            if not isinstance(r, (int, float)) or r <= 0 or r > 500:
                raise ValueError('radius_km must be a number in (0,500]')
        return v


class GetShoppingPlanParams(BaseModel):
    product_ids: List[str]
    max_distance_km: Optional[float] = Field(None, gt=0, le=500)


# ---------- Utility functions ----------

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    # approximate distance in kilometers
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


# ---------- Tool wrapper class ----------
class LLMTolls:
    """Validated, safe wrappers the LLM can request. The LLM should call only
    methods on this class. Each method returns a structured dict with 'ok', 'result',
    and 'error' keys. None of these methods executes arbitrary code or SQL.
    """

    def __init__(self, orchestrator: Optional[Orchestrator] = None, repo: Optional[DataRepository] = None):
        self.orch = orchestrator or Orchestrator(repo=repo)
        # ensure repository loaded
        self.orch._ensure_loaded()
        self.repo = self.orch.repo
        # expose a structured search method that accepts SearchSpec-like dict

    def search_products(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            p = SearchProductsParams(**params)
        except Exception as e:
            return {'ok': False, 'error': f'Invalid parameters: {e}', 'result': None}
        try:
            res = self.orch._search_products(product_ids=p.product_ids, query=p.query, limit=p.limit)
            return {'ok': True, 'result': res, 'error': None}
        except Exception as e:
            return {'ok': False, 'error': str(e), 'result': None}

    def get_product_details(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            p = GetProductDetailsParams(**params)
        except Exception as e:
            return {'ok': False, 'error': f'Invalid parameters: {e}', 'result': None}
        prod = self.repo.get_product(p.product_id)
        if not prod:
            return {'ok': False, 'error': 'product_not_found', 'result': None}
        full = self.orch._product_with_offers(prod)
        return {'ok': True, 'result': full, 'error': None}

    def find_product_offers(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            p = FindProductOffersParams(**params)
        except Exception as e:
            return {'ok': False, 'error': f'invalid_params: {e}', 'result': None}
        prod = self.repo.get_product(p.product_id)
        if not prod:
            return {'ok': False, 'error': 'product_not_found', 'result': None}
        offers = self.repo.get_offers_by_product(p.product_id)[:p.limit]
        off_list = []
        for o in offers:
            off_list.append({
                'offer_id': o.offer_id,
                'shop_id': o.shop_id,
                'price_toman': o.price_toman,
                'last_price_changed_at': o.last_price_changed_at.isoformat() if o.last_price_changed_at else None,
                'warranty': o.warranty,
            })
        return {'ok': True, 'result': off_list, 'error': None}

    def search_stores(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            p = SearchStoresParams(**params)
        except Exception as e:
            return {'ok': False, 'error': f'invalid_params: {e}', 'result': None}
        results = []
        q = (p.query or '').lower()
        for s in self.repo._shops.values():
            hay = ' '.join(filter(None, [s.name or '', s.address or '', s.location_label or ''])).lower()
            if q in hay:
                results.append({
                    'shop_id': s.shop_id,
                    'name': s.name,
                    'address': s.address,
                    'lat': s.lat,
                    'lon': s.lon,
                })
                if len(results) >= p.limit:
                    break
        return {'ok': True, 'result': results, 'error': None}

    def find_nearby_stores(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            p = FindNearbyStoresParams(**params)
        except Exception as e:
            return {'ok': False, 'error': f'invalid_params: {e}', 'result': None}
        candidates = []
        for s in self.repo.shops_with_valid_coords():
            dist = _haversine_km(p.latitude, p.longitude, float(s.lat), float(s.lon))
            if dist <= p.radius_km:
                candidates.append((dist, s))
        candidates.sort(key=lambda x: x[0])
        out = []
        for dist, s in candidates[:p.limit]:
            out.append({
                'shop_id': s.shop_id,
                'name': s.name,
                'address': s.address,
                'lat': s.lat,
                'lon': s.lon,
                'distance_km': round(dist, 3)
            })
        return {'ok': True, 'result': out, 'error': None}

    def compare_offers(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            p = CompareOffersParams(**params)
        except Exception as e:
            return {'ok': False, 'error': f'invalid_params: {e}', 'result': None}
        offers: List[Offer] = []
        if p.offer_ids:
            for oid in p.offer_ids:
                o = self.repo._offers.get(oid)
                if not o:
                    return {'ok': False, 'error': f'offer_not_found:{oid}', 'result': None}
                offers.append(o)
        elif p.product_ids:
            for pid in p.product_ids:
                offs = self.repo.get_offers_by_product(pid)
                offers.extend(offs)
        # prepare comparison table
        table = []
        for o in offers[:p.limit]:
            shop = self.repo.get_shop(o.shop_id) if o.shop_id else None
            table.append({
                'offer_id': o.offer_id,
                'product_id': o.product_id,
                'shop_id': o.shop_id,
                'shop_name': shop.name if shop else None,
                'price_toman': o.price_toman,
                'last_price_changed_at': o.last_price_changed_at.isoformat() if o.last_price_changed_at else None,
                'warranty': o.warranty
            })
        # sort by price asc (None last)
        table.sort(key=lambda x: (x['price_toman'] is None, x['price_toman']))
        return {'ok': True, 'result': table, 'error': None}

    def find_alternatives(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            p = FindAlternativesParams(**params)
        except Exception as e:
            return {'ok': False, 'error': f'invalid_params: {e}', 'result': None}
        prod = self.repo.get_product(p.product_id)
        if not prod:
            return {'ok': False, 'error': 'product_not_found', 'result': None}
        candidates = []
        for other in self.repo._products.values():
            if other.product_id == prod.product_id:
                continue
            # same category preferred, then same brand
            score = 0
            if other.category_name and prod.category_name and other.category_name == prod.category_name:
                score += 10
            if other.brand_name and prod.brand_name and other.brand_name == prod.brand_name:
                score += 5
            candidates.append((score, other))
        candidates.sort(key=lambda x: x[0], reverse=True)
        out = []
        for score, o in candidates[:p.limit]:
            out.append({'product_id': o.product_id, 'persian_title': o.persian_title, 'brand': o.brand_name, 'score': score})
        return {'ok': True, 'result': out, 'error': None}

    def update_search_filters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            p = UpdateSearchFiltersParams(**params)
        except Exception as e:
            return {'ok': False, 'error': f'invalid_params: {e}', 'result': None}
        # return the validated filters (this function does not mutate conversation state)
        return {'ok': True, 'result': p.filters, 'error': None}

    def get_shopping_plan(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            p = GetShoppingPlanParams(**params)
        except Exception as e:
            return {'ok': False, 'error': f'invalid_params: {e}', 'result': None}
        plan = []
        # For each product, pick cheapest recorded offer and its shop
        for pid in p.product_ids:
            prod = self.repo.get_product(pid)
            if not prod:
                plan.append({'product_id': pid, 'error': 'product_not_found'})
                continue
            offers = self.repo.get_offers_by_product(pid)
            if not offers:
                plan.append({'product_id': pid, 'error': 'no_offers'})
                continue
            offers = [o for o in offers if o.price_toman is not None]
            if not offers:
                plan.append({'product_id': pid, 'error': 'no_priced_offers'})
                continue
            best = sorted(offers, key=lambda x: x.price_toman)[0]
            shop = self.repo.get_shop(best.shop_id) if best.shop_id else None
            entry = {'product_id': pid, 'offer_id': best.offer_id, 'price_toman': best.price_toman, 'shop_id': best.shop_id}
            if shop and shop.lat is not None and p.max_distance_km is not None:
                # check distance from shop to itself (no origin provided) -- cannot filter, so just include distance None
                entry['distance_km'] = None
            plan.append(entry)
        return {'ok': True, 'result': plan, 'error': None}

    def structured_search(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Perform a dataset search using a SearchSpec-like dict and return matching shops/offers.

        This function is intentionally conservative: it validates the input and then performs
        repository lookups, filtering by price, brand, keywords, and location (when textual).
        Sorting is applied according to spec['search_options']['sort_by'] or price if requested.
        """
        try:
            # validate spec via pydantic
            from search_spec import SearchSpec
            s = SearchSpec(**spec) if not isinstance(spec, SearchSpec) else spec
        except Exception as e:
            return {'ok': False, 'error': f'invalid_spec: {e}', 'result': None}

        # simple matching: gather candidate offers
        candidates = []  # tuples (score, offer, product, shop)
        # prepare keyword lowercased lists
        kws = [k.lower() for k in (s.product.keywords or [])]
        excl = [k.lower() for k in (s.product.exclude_keywords or [])]
        brand = (s.product.brand or '').lower() if s.product.brand else None
        pname = (s.product.name or '').lower() if s.product.name else None

        for o in self.repo._offers.values():
            # load product
            p = self.repo.get_product(o.product_id)
            if not p:
                continue
            # price filter
            if s.price.max is not None and o.price_toman is not None:
                if o.price_toman > s.price.max:
                    continue
            if s.price.min is not None and o.price_toman is not None:
                if o.price_toman < s.price.min:
                    continue
            # brand filter
            if brand and p.brand_name and brand not in (p.brand_name or '').lower():
                continue
            # name/keyword filter
            hay = ' '.join(filter(None, [p.persian_title or '', p.brand_name or '', p.model or ''])).lower()
            if pname and pname not in hay:
                continue
            if kws and not any(k in hay for k in kws):
                # if keywords provided but none match, skip
                continue
            if excl and any(k in hay for k in excl):
                continue
            # score: basic heuristic
            score = 0
            if s.price.max is not None and o.price_toman is not None:
                # closer to max gets higher score
                score += max(0, (s.price.max - o.price_toman) / (s.price.max + 1)) * 30
            if brand and p.brand_name and brand == (p.brand_name or '').lower():
                score += 20
            # prefer offers with recorded price
            if o.price_toman is not None:
                score += 10
            # features matching (simple substring match)
            for f in s.product.features:
                fn = f.name.lower()
                if p.specs and isinstance(p.specs, dict):
                    for k, v in p.specs.items():
                        if fn in str(k).lower() and str(f.value).lower() in str(v).lower():
                            score += 5
            shop = self.repo.get_shop(o.shop_id) if o.shop_id else None
            candidates.append((score, o, p, shop))

        # sort candidates by score desc then price asc
        candidates.sort(key=lambda x: (-x[0], x[1].price_toman if x[1].price_toman is not None else 10**18))

        # apply limit
        limit = s.search_options.limit or 10
        out = []
        for score, o, p, shop in candidates[:limit]:
            out.append({
                'score': round(score, 3),
                'offer_id': o.offer_id,
                'product_id': p.product_id,
                'persian_title': p.persian_title,
                'brand': p.brand_name,
                'price_toman': o.price_toman,
                'shop_id': shop.shop_id if shop else None,
                'shop_name': shop.name if shop else None,
                'shop_address': shop.address if shop else None,
            })

        return {'ok': True, 'result': out, 'error': None}


# simple executor to limit steps
class ToolExecutor:
    def __init__(self, tools: LLMTolls, max_steps: int = 6):
        self.tools = tools
        self.max_steps = max_steps
        self.steps = 0

    def call(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if self.steps >= self.max_steps:
            return {'ok': False, 'error': 'max_steps_exceeded', 'result': None}
        self.steps += 1
        fn = getattr(self.tools, tool_name, None)
        if not fn:
            return {'ok': False, 'error': 'tool_not_found', 'result': None}
        return fn(params)


if __name__ == '__main__':
    import logging
    logging.getLogger('shopping_assistant.llm_tools').info('LLM tools module loaded')
