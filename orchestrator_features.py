from typing import List, Dict, Any, Optional, Tuple
from repository import DataRepository
from conversation_state import ConversationState
from datetime import datetime
from math import radians, cos, sin, asin, sqrt


# Utilities

def haversine_km(lat1, lon1, lat2, lon2):
    # return distance in kilometers
    if None in (lat1, lon1, lat2, lon2):
        return None
    try:
        lat1, lon1, lat2, lon2 = map(float, (lat1, lon1, lat2, lon2))
    except Exception:
        return None
    # haversine
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return R * c


class OrchestratorFeatures:
    def __init__(self, repo: Optional[DataRepository] = None):
        self.repo = repo or DataRepository()
        self.repo.load_all()

    # 1) Compare product offers and specs
    def compare_product(self, product_id: str, ranking_weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Return structured comparison between offers for a product and compare with similar products' specs.
        ranking_weights: e.g. {'price':0.5, 'warranty':0.2, 'distance':0.3}
        """
        if ranking_weights is None:
            ranking_weights = {'price': 0.6, 'warranty': 0.2, 'distance': 0.2}
        p = self.repo.get_product(product_id)
        if not p:
            return {'error': 'product_not_found'}
        offers = self.repo.get_offers_by_product(product_id)
        # compute score per offer
        scored = []
        for o in offers:
            score = 0.0
            # price: lower is better. normalized later
            price = o.price_toman or 10**9
            # warranty: presence counts
            warranty_score = 1.0 if o.warranty else 0.0
            # distance: try to compute using shop coords if present
            shop = self.repo.get_shop(o.shop_id) if o.shop_id else None
            distance_km = None
            if shop and shop.lat and shop.lon:
                # no user location supplied here: distance unknown
                distance_km = None
            scored.append({'offer': o, 'price': price, 'warranty': warranty_score, 'distance_km': distance_km, 'shop': shop})
        # normalize price into 0..1 where 1 is best
        prices = [s['price'] for s in scored if s['price'] < 10**8]
        if prices:
            maxp = max(prices)
            minp = min(prices)
        else:
            maxp = minp = None
        for s in scored:
            if minp is None or maxp is None or maxp==minp:
                p_score = 0.5
            else:
                p_score = 1.0 - (s['price'] - minp) / (maxp - minp)
            d_score = 0.5 if s['distance_km'] is None else max(0.0, 1.0 - (s['distance_km']/50.0))
            w_score = s['warranty']
            s['composite_score'] = p_score * ranking_weights.get('price',0.6) + w_score * ranking_weights.get('warranty',0.2) + d_score * ranking_weights.get('distance',0.2)
        scored = sorted(scored, key=lambda x: x['composite_score'], reverse=True)
        # reason extraction: differences between top and others
        reasons = []
        if scored:
            top = scored[0]
            for s in scored[1:]:
                diff = []
                if s['price'] != top['price']:
                    diff.append(f"تفاوت قیمت: {top['price']} vs {s['price']}")
                if s['warranty'] != top['warranty']:
                    diff.append("وجود/نبود گارانتی متفاوت")
                reasons.append({'better': top['offer'].offer_id, 'worse': s['offer'].offer_id, 'diff': diff})
        # specs comparison: use product specs and find similar products in same category
        sim_products = [q for q in self.repo._products.values() if q.category_name == p.category_name and q.product_id != p.product_id]
        spec_diffs = []
        for sp in sim_products[:5]:
            # compare specs dict lists by key
            a = {item.get('k'): item.get('v') for item in (p.specs or []) if isinstance(item, dict)}
            b = {item.get('k'): item.get('v') for item in (sp.specs or []) if isinstance(item, dict)}
            common = set(a.keys()) & set(b.keys())
            diffs = {k: (a.get(k), b.get(k)) for k in common if a.get(k)!=b.get(k)}
            if diffs:
                spec_diffs.append({'product_id': sp.product_id, 'title': sp.persian_title, 'diffs': diffs})
        return {'product_id': p.product_id, 'title': p.persian_title, 'offers': [{'offer_id': s['offer'].offer_id, 'shop_id': s['shop'].shop_id if s['shop'] else None, 'price': s['price'], 'warranty': s['offer'].warranty} for s in scored], 'ranking_weights': ranking_weights, 'reasons': reasons, 'spec_diffs': spec_diffs}

    # 2) find substitutes
    def find_substitutes(self, product_id: str, top_n: int = 5, feature_weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        if feature_weights is None:
            feature_weights = {'category': 0.4, 'specs': 0.4, 'brand_preference': 0.2}
        p = self.repo.get_product(product_id)
        if not p:
            return {'error': 'product_not_found'}
        candidates = [q for q in self.repo._products.values() if q.category_name == p.category_name and q.product_id != p.product_id]
        scored = []
        base_specs = {item.get('k'): item.get('v') for item in (p.specs or []) if isinstance(item, dict)}
        for c in candidates:
            score = 0.0
            # category match is binary here
            cat_score = 1.0 if c.category_name == p.category_name else 0.0
            # specs overlap
            c_specs = {item.get('k'): item.get('v') for item in (c.specs or []) if isinstance(item, dict)}
            if not base_specs:
                spec_score = 0.0
            else:
                common_keys = set(base_specs.keys()) & set(c_specs.keys())
                if not common_keys:
                    spec_score = 0.0
                else:
                    matches = sum(1 for k in common_keys if base_specs.get(k) == c_specs.get(k))
                    spec_score = matches / len(base_specs)
            # brand pref (prefer same brand)
            brand_score = 1.0 if c.brand_name == p.brand_name else 0.0
            combined = feature_weights.get('category',0.4)*cat_score + feature_weights.get('specs',0.4)*spec_score + feature_weights.get('brand_preference',0.2)*brand_score
            scored.append({'product': c, 'score': combined, 'spec_match': spec_score})
        scored = sorted(scored, key=lambda x: x['score'], reverse=True)[:top_n]
        results = []
        for s in scored:
            # identify which main features differ
            diffs = {}
            for k in set(list(base_specs.keys()) + list({item.get('k') for item in (s['product'].specs or []) if isinstance(item, dict)})):
                a = base_specs.get(k)
                b = next((item.get('v') for item in (s['product'].specs or []) if item.get('k')==k), None)
                if a != b:
                    diffs[k] = {'requested': a, 'candidate': b}
            results.append({'product_id': s['product'].product_id, 'title': s['product'].persian_title, 'score': s['score'], 'diffs': diffs})
        return {'base_product': product_id, 'substitutes': results}

    # 3) visit planner
    def plan_visit(self, product_ids: List[str], user_location: Optional[Tuple[float,float]] = None, max_distance_km: Optional[float]=50.0) -> Dict[str, Any]:
        # collect shops that have these products
        shop_map = {}
        for pid in product_ids:
            offers = self.repo.get_offers_by_product(pid)
            for o in offers:
                s = self.repo.get_shop(o.shop_id) if o.shop_id else None
                if not s:
                    continue
                shop_map.setdefault(s.shop_id, {'shop': s, 'offers': []})
                shop_map[s.shop_id]['offers'].append({'product_id': pid, 'offer_id': o.offer_id, 'price': o.price_toman})
        shops = list(shop_map.values())
        # compute distance if user_location provided
        for entry in shops:
            s = entry['shop']
            if user_location and s.lat and s.lon:
                entry['distance_km'] = haversine_km(user_location[0], user_location[1], s.lat, s.lon)
            else:
                entry['distance_km'] = None
            # attach weekly hours if present
            entry['weekly_hours'] = s.weekly_hours
        # filter by max_distance_km
        if user_location and max_distance_km is not None:
            shops = [e for e in shops if e['distance_km'] is not None and e['distance_km']<=max_distance_km]
        # sort by distance if available else by number of requested products available
        shops = sorted(shops, key=lambda x: (x['distance_km'] if x['distance_km'] is not None else 9999, -len(x['offers'])))
        # construct visit plan: greedy route by distance
        plan = []
        remaining = shops.copy()
        current_loc = user_location
        while remaining:
            # pick nearest to current_loc
            if current_loc:
                remaining = sorted(remaining, key=lambda x: x['distance_km'] if x['distance_km'] is not None else 9999)
            else:
                remaining = sorted(remaining, key=lambda x: -len(x['offers']))
            next_shop = remaining.pop(0)
            plan.append({'shop_id': next_shop['shop'].shop_id, 'name': next_shop['shop'].name, 'distance_km': next_shop['distance_km'], 'offers': next_shop['offers'], 'weekly_hours': next_shop['weekly_hours']})
            # update current_loc to shop coords
            if next_shop['shop'].lat and next_shop['shop'].lon:
                current_loc = (next_shop['shop'].lat, next_shop['shop'].lon)
            else:
                current_loc = None
        return {'plan': plan, 'note': 'سفارش بازدید پیشنهادی بر اساس فاصله و تعداد محصولات موجود'}

    # 4) follow-up recording (architecture to attach to real channel later)
    def record_followup(self, state: ConversationState, offer_id: str, result: Dict[str, Any]):
        entry = {'timestamp': datetime.utcnow().isoformat(), 'offer_id': offer_id, 'result': result}
        state.followups.append(entry)
        # store into conversation_state via orchestrator.save_state externally
        return entry

    # 5) transparency helpers
    def tag_transparency(self, product_entry: Dict[str,Any]):
        # product_entry assumed to have offers with price_type and warranty_recorded
        for o in product_entry.get('offers', []):
            o['price_kind'] = 'recorded' if o.get('price_type','')=='recorded' else 'unknown'
            o['warranty_recorded'] = o.get('warranty_recorded')
            o['inventory_confirmed'] = o.get('followed_up', False)
        return product_entry


if __name__ == '__main__':
    of = OrchestratorFeatures()
    # print(of.compare_product('p1'))
