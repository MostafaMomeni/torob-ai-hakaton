from typing import List, Dict, Any, Optional
from repository import DataRepository
from text_utils import normalize_persian_text, tokenize_persian
import logging
import re

logger = logging.getLogger("shopping_assistant.product_search")


class MatchRecord:
    """Search evidence and rank metadata for one product."""
    def __init__(self, product_id):
        self.product_id = product_id
        self.stages = []
        self.matched_tokens = set()
        self.matched_phrases = []
        self.query_tokens = set()
        self.name_relevance_score = 0.0
        self.secondary_score = 0.0
        self.tier = 7
        self.final_rank = 0
        self.gap_words = None
        self.token_match_count = 0
        self.exact_title_match = False

    def add_match(self, stage, match_type, phrase, tokens, boost=0.0):
        item = (stage, match_type, phrase, list(tokens))
        if item not in self.stages:
            self.stages.append(item)
        if phrase and phrase not in self.matched_phrases:
            self.matched_phrases.append(phrase)
        self.matched_tokens.update(t for t in tokens if t)
        self.name_relevance_score += boost


class ProductSearcher:
    """Exact title > full phrase > full ordered query > partial phrases > token coverage."""
    def __init__(self, repo: DataRepository, config: Optional[Dict[str, Any]] = None):
        self.repo = repo
        self.config = config or {}
        self.max_gap_words = int(self.config.get("max_gap_words", 3))
        self.weights = {"exact_match": 1000.0, "full_phrase": 500.0,
                        "ordered_phrase": 100.0, "token_match": 20.0,
                        "brand_model": 60.0, "phrase_prefix": 80.0}

    def _product_text_fields(self, p):
        return {"title": p.persian_title or "", "brand": p.brand_name or "",
                "model": p.model or "", "category": getattr(p, "category_name", "") or ""}

    def _normalized_product_hay(self, p):
        return normalize_persian_text(" ".join(str(v) for v in self._product_text_fields(p).values() if v))

    def _title(self, p):
        return normalize_persian_text(getattr(p, "persian_title", "") or "")

    def _stage1_full_phrase(self, query):
        q = normalize_persian_text(query)
        exact, phrase = [], []
        if not q:
            return []
        for p in self.repo._products.values():
            title, hay = self._title(p), self._normalized_product_hay(p)
            if title == q:
                exact.append((p.product_id, q))
            elif re.search(r"(?<!\w)" + re.escape(q) + r"(?!\w)", hay):
                phrase.append((p.product_id, q))
        return exact + phrase

    def _stage1b_in_order(self, query):
        tokens = tokenize_persian(query)
        if len(tokens) < 2:
            return []
        results = []
        for p in self.repo._products.values():
            words, best_gap = self._normalized_product_hay(p).split(), None
            for start, word in enumerate(words):
                if word != tokens[0]:
                    continue
                pos, ok = start, True
                for token in tokens[1:]:
                    try:
                        pos = words.index(token, pos + 1)
                    except ValueError:
                        ok = False
                        break
                if ok:
                    gap = pos - start + 1 - len(tokens)
                    best_gap = gap if best_gap is None else min(best_gap, gap)
            if best_gap is not None and best_gap <= self.max_gap_words:
                results.append((p.product_id, best_gap))
        return sorted(results, key=lambda x: (x[1], x[0]))

    def _ngrams_ordered(self, tokens, min_len=2):
        phrases = {" ".join(tokens[i:i+n]) for n in range(min_len, len(tokens)+1)
                   for i in range(len(tokens)-n+1)}
        return sorted(phrases, key=lambda x: (-len(x.split()), -len(x), x))

    def _stage2_ordered_phrases(self, query, limit_per_phrase=10000):
        found = {}
        for phrase in self._ngrams_ordered(tokenize_persian(query)):
            for p in self.repo._products.values():
                if phrase in self._normalized_product_hay(p):
                    old = found.get(p.product_id)
                    if old is None or len(phrase.split()) > len(old.split()):
                        found[p.product_id] = phrase
        return sorted(found.items(), key=lambda x: (-len(x[1].split()), x[0]))[:limit_per_phrase]

    def _stage3_tokens(self, query, min_token_matches=1, top_k=10000):
        tokens = list(dict.fromkeys(tokenize_persian(query)))
        results = []
        for p in self.repo._products.values():
            words = set(self._normalized_product_hay(p).split())
            matched = [t for t in tokens if t in words]
            if len(matched) >= min_token_matches:
                results.append((p.product_id, matched))
        return sorted(results, key=lambda x: (-len(x[1]), x[0]))[:top_k]

    def retrieve_candidates(self, product_name=None, keywords=None, brand=None, model=None, limit=50):
        if not any([product_name, keywords, brand, model]):
            return []
        query_tokens = list(dict.fromkeys(tokenize_persian(product_name or "")))
        candidates = {}

        def rec_for(pid):
            if pid not in candidates:
                candidates[pid] = MatchRecord(pid)
                candidates[pid].query_tokens = set(query_tokens)
            return candidates[pid]

        if product_name:
            q = normalize_persian_text(product_name)
            for pid, phrase in self._stage1_full_phrase(product_name):
                p, rec = self.repo.get_product(pid), rec_for(pid)
                exact = self._title(p) == q
                rec.exact_title_match = exact
                rec.add_match(1, "exact_title" if exact else "full_phrase", phrase,
                              tokenize_persian(phrase, False),
                              self.weights["exact_match"] if exact else self.weights["full_phrase"])
            for pid, gap in self._stage1b_in_order(product_name):
                rec = rec_for(pid)
                rec.gap_words = gap
                rec.add_match(2, "in_order_phrase", q, query_tokens,
                              self.weights["full_phrase"] + max(0, self.max_gap_words-gap)*20)
            for pid, phrase in self._stage2_ordered_phrases(product_name):
                rec = rec_for(pid)
                rec.add_match(3, "ordered_phrase", phrase, tokenize_persian(phrase),
                              self.weights["ordered_phrase"] * len(tokenize_persian(phrase)))
            for pid, matched in self._stage3_tokens(product_name):
                rec = rec_for(pid)
                rec.token_match_count = max(rec.token_match_count, len(matched))
                rec.add_match(4, "token_match", None, matched, self.weights["token_match"]*len(matched))

        for value in (brand, model):
            if value:
                needle = normalize_persian_text(value)
                for p in self.repo._products.values():
                    if needle and needle in self._normalized_product_hay(p):
                        rec_for(p.product_id).add_match(5, "brand_model", needle,
                            tokenize_persian(needle), self.weights["brand_model"])
        for keyword in (keywords or []):
            needle = normalize_persian_text(keyword)
            if needle:
                for p in self.repo._products.values():
                    if needle in self._normalized_product_hay(p):
                        rec_for(p.product_id).add_match(6, "keyword", needle,
                            tokenize_persian(needle), self.weights["token_match"]/2)

        out = []
        for pid, rec in candidates.items():
            p = self.repo.get_product(pid)
            if p:
                rec.name_relevance_score = self.score_name_relevance(rec, p)
                rec.tier = self.compute_tier(rec, p)
                out.append({"product_id": pid, "persian_title": p.persian_title,
                            "brand_name": p.brand_name, "model": p.model,
                            "specs": p.specs, "match_record": rec})
        return sorted(out, key=lambda c: (c["match_record"].tier,
                    -c["match_record"].name_relevance_score,
                    -self.token_coverage(c["match_record"]),
                    normalize_persian_text(c["persian_title"] or "")))

    def score_name_relevance(self, rec, product_obj):
        title = self._title(product_obj)
        score = rec.name_relevance_score
        if rec.exact_title_match:
            score += 1000
        if any(title.startswith(ph + " ") for ph in rec.matched_phrases if ph):
            score += self.weights["phrase_prefix"]
        if rec.query_tokens:
            score += 100 * len(set(title.split()) & rec.query_tokens) / len(rec.query_tokens)
        return float(score)

    def compute_tier(self, rec, product_obj):
        if rec.exact_title_match:
            return 1
        types = {s[1] for s in rec.stages}
        if "full_phrase" in types:
            return 2
        if "in_order_phrase" in types:
            return 3
        if "ordered_phrase" in types:
            return 4
        if "token_match" in types:
            return 5
        if "brand_model" in types:
            return 6
        return 7

    def token_coverage(self, rec):
        return len(rec.matched_tokens & rec.query_tokens) / len(rec.query_tokens) if rec.query_tokens else 0.0


def _plain(value):
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    if hasattr(value, "dict"):
        return value.dict(exclude_none=True)
    return value if isinstance(value, dict) else {}


def _flatten(value):
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(_flatten(v) for v in value.values())
    if isinstance(value, (list, tuple, set)):
        return " ".join(_flatten(v) for v in value)
    return str(value)


def _shop_details(repo, shop_id):
    getter = getattr(repo, "get_shop_details", None)
    if callable(getter):
        details = getter(shop_id)
        if details:
            return details
    shop = repo.get_shop(shop_id) if shop_id else None
    if not shop:
        return {}
    return {key: getattr(shop, key, None) for key in (
        "shop_id", "name", "address", "category_names", "contacts", "instagram",
        "lat", "location_label", "lon", "payment_options",
        "supports_fast_delivery", "website_url", "weekly_hours"
    )}


def _shop_text(repo, offer):
    details = _shop_details(repo, getattr(offer, "shop_id", None))
    return normalize_persian_text(" ".join(str(details.get(key) or "") for key in
        ("name", "address", "location_label", "category_names")))


def _coordinates(location):
    location = _plain(location)
    lat = location.get("latitude", location.get("lat"))
    lon = location.get("longitude", location.get("lon", location.get("lng")))
    try:
        lat, lon = float(lat), float(lon)
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            return lat, lon
    except (TypeError, ValueError):
        pass
    return None


def _shop_coordinates(details):
    try:
        lat, lon = float(details.get("lat")), float(details.get("lon"))
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            return lat, lon
    except (TypeError, ValueError):
        pass
    return None


def _distance_km(a, b):
    """Haversine distance between (lat, lon) pairs."""
    import math
    lat1, lon1 = a
    lat2, lon2 = b
    radius = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    h = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return radius * 2 * math.asin(min(1.0, math.sqrt(h)))


def _shop_matches_place(details: Dict[str, Any], place_query: str) -> bool:
    """Match a street/area query against the shop address robustly.

    Persian addresses are inconsistent: they may use Arabic/Persian character
    variants, punctuation, ZWNJ, abbreviations such as «خ», or different spacing.
    A location query may be only a street name, not the complete address.
    """
    query = normalize_persian_text(place_query)
    if not query:
        return True

    address = normalize_persian_text(details.get("address") or "")
    label = normalize_persian_text(details.get("location_label") or "")
    candidates = [value for value in (address, label) if value]
    if not candidates:
        return False

    # Ignore generic address words; a query such as «خیابان» alone is too broad
    # to identify a street and must not match every address.
    generic = {
        "خیابان", "خ", "کوچه", "ک", "بلوار", "بل", "میدان", "چهارراه",
        "جنب", "روبروی", "روبرو", "پلاک", "واحد", "طبقه", "ساختمان",
        "شهر", "استان", "ایران", "در", "واقع", "واقعشده", "آدرس",
        "مغازه", "فروشگاه", "خیابانهای",
    }
    query_tokens = [t for t in query.split() if t not in generic and len(t) > 1]
    if not query_tokens:
        return False

    # First try a contiguous phrase match, including a compact form so that
    # «ولیعصر» and «ولی عصر» can match the same stored street name.
    query_compact = "".join(query_tokens)
    for candidate in candidates:
        candidate_compact = candidate.replace(" ", "")
        if " ".join(query_tokens) in candidate or query_compact in candidate_compact:
            return True

    # Require every remaining query token to match (AND, not loose OR).
    for candidate in candidates:
        address_tokens = set(candidate.split())
        compact_address = candidate.replace(" ", "")
        matched = 0
        for token in query_tokens:
            if token in address_tokens or token in compact_address:
                matched += 1
        if matched == len(query_tokens):
            return True
    return False


def _resolve_text_location(repo, location_text):
    """Find shop_ids whose address/location_label contain the place name.

    Also returns the centroid of those shops' coordinates when available, used
    only for secondary near-to-far ordering — never as a soft ranking substitute
    for a hard address filter.
    """
    query = normalize_persian_text(location_text)
    if not query:
        return None, set()
    matched_coords, matched_shop_ids = [], set()
    shops = getattr(repo, "_shops", {}) or {}
    for shop_id, shop in shops.items():
        details = _shop_details(repo, shop_id)
        if not _shop_matches_place(details, query):
            continue
        matched_shop_ids.add(str(shop_id))
        coords = _shop_coordinates(details)
        if coords:
            matched_coords.append(coords)
    if not matched_coords:
        return None, matched_shop_ids
    return (
        sum(x[0] for x in matched_coords) / len(matched_coords),
        sum(x[1] for x in matched_coords) / len(matched_coords),
    ), matched_shop_ids


def _brand_matches(requested: str, actual: str) -> bool:
    """Strict brand condition (AND).

    - If no brand was requested → pass.
    - If brand was requested but product has no brand → fail.
    - Otherwise requested brand must appear inside product brand_name
      (normalized), so «سامسونگ» matches «سامسونگ» / «گوشی سامسونگ» but not «اپل».
    """
    if not requested:
        return True
    if not actual:
        return False
    req = normalize_persian_text(requested)
    act = normalize_persian_text(actual)
    if not req or not act:
        return False
    return req == act or req in act


def _name_matches(requested_name: str, product) -> bool:
    """Strict product-name condition (AND).

    Every non-stopword token of the requested name must appear in the product
    title (and optionally brand/model). A single shared token is not enough.
    If no name was requested, this condition is not applied.
    """
    if not requested_name or not str(requested_name).strip():
        return True
    tokens = tokenize_persian(requested_name, remove_stopwords=True)
    if not tokens:
        # pure stopwords / noise → treat as no name filter
        return True
    title = normalize_persian_text(getattr(product, "persian_title", "") or "")
    brand = normalize_persian_text(getattr(product, "brand_name", "") or "")
    model = normalize_persian_text(getattr(product, "model", "") or "")
    hay = f"{title} {brand} {model}".strip()
    if not hay:
        return False
    # All query tokens must be present in the product text.
    return all(tok in hay for tok in tokens)


def _compact_shop(details: Dict[str, Any], distance_km=None) -> Dict[str, Any]:
    """Fixed shop shape for every offer in the response."""
    details = details or {}
    phones = []
    contacts = details.get("contacts")
    if isinstance(contacts, list):
        for c in contacts:
            if isinstance(c, dict):
                num = c.get("display_number") or c.get("number") or c.get("value")
                if num:
                    phones.append(str(num))
            elif c:
                phones.append(str(c))
    elif isinstance(contacts, str) and contacts.strip():
        phones.append(contacts.strip())

    dist = None
    if distance_km is not None:
        try:
            dist = round(float(distance_km), 3)
        except (TypeError, ValueError):
            dist = None

    return {
        "shop_id": details.get("shop_id") or None,
        "name": details.get("name") or None,
        "address": details.get("address") or None,
        "location_label": details.get("location_label") or None,
        "lat": details.get("lat") if details.get("lat") is not None else None,
        "lon": details.get("lon") if details.get("lon") is not None else None,
        "phones": phones[:3],
        "distance_km": dist,
    }


def _compact_offer(offer, shop_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Fixed offer shape for every response item."""
    shop_payload = shop_payload or _compact_shop({})
    return {
        "offer_id": getattr(offer, "offer_id", None),
        "price_toman": getattr(offer, "price_toman", None),
        "warranty": getattr(offer, "warranty", None) or None,
        "shop_id": getattr(offer, "shop_id", None),
        "distance_km": shop_payload.get("distance_km"),
        "shop": shop_payload,
    }


def _compact_product(product_id, persian_title, brand_name, model, offers, nearest_distance=None) -> Dict[str, Any]:
    """Fixed product shape — always the same keys."""
    return {
        "product_id": product_id,
        "persian_title": persian_title or None,
        "brand_name": brand_name or None,
        "model": model or None,
        "nearest_shop_distance_km": (
            round(nearest_distance, 3) if nearest_distance is not None else None
        ),
        "offers": offers or [],
    }


def run_structured_search(
    repo: DataRepository,
    spec: Dict[str, Any],
    limit: int = 50,
    offset: int = 0,
    return_total: bool = False,
):
    """AND search pipeline — every active condition must hold.

    Conditions (when provided):
      • product name  — all name tokens must appear in the product title
      • brand         — product brand_name must contain the requested brand
      • location text — every surviving offer's shop address/location_label
                        must contain the place name
      • GPS + radius  — offer shop must be within radius_km of (lat, lon)
      • min/max price — offer price must fall in range
      • features      — each feature term must appear in product text

    A product enters the result list only if it matches name+brand+features
    AND has at least one offer that satisfies price+location. Otherwise it is
    dropped. Empty list means nothing satisfied all conditions.

    Output shape is always the same for every product/offer/shop.

    Pagination:
      - ``offset`` / ``limit`` slice the ranked list after full ranking.
      - If ``return_total=True``, returns ``(page_results, total_matched)``
        so callers can build pagination metadata without a second pass.
    """
    spec = _plain(spec)
    product = _plain(spec.get("product", {}))
    price = _plain(spec.get("price", {}))
    location = _plain(spec.get("location", {}))
    options = _plain(spec.get("search_options", {}))

    name = product.get("name") or product.get("product_name") or spec.get("query")
    keywords = product.get("keywords") or spec.get("keywords") or []
    if isinstance(keywords, str):
        keywords = [keywords]
    brand = product.get("brand") or spec.get("brand")
    model = product.get("model") or spec.get("model")
    requested_brand = (brand or "").strip()

    # Name is a required AND condition when present: no name match → no results.
    if name and str(name).strip():
        ps = ProductSearcher(repo)
        # Broad retrieval first, then hard _name_matches filter below.
        candidates = ps.retrieve_candidates(
            name, keywords, brand=None, model=model, limit=max(limit * 30, 1000)
        )
    elif keywords or model:
        ps = ProductSearcher(repo)
        candidates = ps.retrieve_candidates(
            None, keywords, brand=None, model=model, limit=max(limit * 30, 1000)
        )
    else:
        # No product identity at all → cannot satisfy name condition.
        return []

    # Hard AND: name
    if name and str(name).strip():
        candidates = [
            c for c in candidates
            if (p := repo.get_product(c["product_id"])) is not None
            and _name_matches(name, p)
        ]
    if not candidates:
        return []

    # Hard AND: brand (only among name-matched products)
    if requested_brand:
        candidates = [
            c for c in candidates
            if (p := repo.get_product(c["product_id"])) is not None
            and _brand_matches(requested_brand, getattr(p, "brand_name", "") or "")
        ]
        if not candidates:
            return []

    min_price = price.get("min", spec.get("min_price"))
    max_price = price.get("max", spec.get("max_price"))

    # Street/address text is more specific than city. Do not let a generic
    # city value override the street name the user actually supplied.
    loc_value = (
        location.get("address")
        or location.get("text")
        or location.get("area")
        or location.get("city")
    )
    if not loc_value and isinstance(spec.get("location"), str):
        loc_value = spec["location"]
    location_query = normalize_persian_text(loc_value) if loc_value else ""
    address_hard_filter = bool(location_query)

    target_coords = _coordinates(location)
    matched_location_shop_ids: set = set()
    if location_query:
        centroid, matched_location_shop_ids = _resolve_text_location(repo, loc_value)
        # Place name with zero matching shops → nothing can satisfy location AND.
        if not matched_location_shop_ids:
            return []
        if target_coords is None:
            target_coords = centroid

    radius_km = location.get("radius_km") or location.get("max_distance_km")
    try:
        radius_km = float(radius_km) if radius_km is not None else None
    except (TypeError, ValueError):
        radius_km = None

    feature_terms = []
    for feature in (product.get("features") or product.get("attributes") or []):
        if isinstance(feature, str):
            feature_terms.append(normalize_persian_text(feature))
        elif isinstance(feature, dict):
            if feature.get("name"):
                feature_terms.append(normalize_persian_text(feature["name"]))
            value = feature.get("value")
            if value not in (None, True, False):
                feature_terms.append(normalize_persian_text(value))

    sort_by = str(options.get("sort_by") or spec.get("sort_by") or "relevance").lower()
    ranked = []
    ps = ProductSearcher(repo)

    for candidate in candidates:
        pid = candidate["product_id"]
        p = repo.get_product(pid)
        if not p:
            continue

        # Hard AND: features
        if feature_terms:
            searchable = normalize_persian_text(
                _flatten(getattr(p, "specs", None)) + " " +
                (getattr(p, "persian_title", "") or "")
            )
            if any(term and term not in searchable for term in feature_terms):
                continue

        offers = []
        for offer in repo.get_offers_by_product(pid):
            if offer.price_toman is None:
                continue
            try:
                amount = float(offer.price_toman)
            except (ValueError, TypeError):
                continue

            # Hard AND: price
            if min_price is not None and amount < float(min_price):
                continue
            if max_price is not None and amount > float(max_price):
                continue

            details = _shop_details(repo, getattr(offer, "shop_id", None))
            shop_id_str = str(details.get("shop_id") or offer.shop_id or "")

            # Hard AND: location text — shop must be in the resolved place set
            if address_hard_filter:
                if shop_id_str not in matched_location_shop_ids:
                    continue

            shop_coords = _shop_coordinates(details)
            distance_km = (
                _distance_km(target_coords, shop_coords)
                if target_coords and shop_coords
                else None
            )
            # Hard AND: radius when GPS (+ optional radius) is active
            if radius_km is not None:
                if distance_km is None or distance_km > radius_km:
                    continue

            shop_payload = _compact_shop(details, distance_km=distance_km)
            offers.append(_compact_offer(offer, shop_payload))

        # Product only survives if at least one offer passed every offer-level AND.
        if not offers:
            continue

        rec = candidate.get("match_record")
        valid_distances = [
            o.get("distance_km") for o in offers if o.get("distance_km") is not None
        ]
        nearest_distance = min(valid_distances) if valid_distances else None
        cheapest = min(
            (float(o["price_toman"]) for o in offers if o.get("price_toman") is not None),
            default=float("inf"),
        )
        tier = getattr(rec, "tier", 7) if rec else 7
        name_score = getattr(rec, "name_relevance_score", 0.0) if rec else 0.0
        coverage = ps.token_coverage(rec) if rec else 0.0

        if target_coords is not None:
            key = (
                nearest_distance if nearest_distance is not None else float("inf"),
                tier,
                -name_score,
                cheapest,
            )
        elif sort_by in ("price_asc", "price_low_to_high", "cheapest", "price"):
            key = (cheapest, tier, -name_score)
        elif sort_by in ("price_desc", "price_high_to_low", "most_expensive"):
            key = (-cheapest, tier, -name_score)
        else:
            key = (tier, -name_score, -coverage, cheapest)

        if target_coords is not None:
            offers.sort(key=lambda o: (
                o.get("distance_km") if o.get("distance_km") is not None else float("inf"),
                float(o.get("price_toman") or 0),
            ))
        else:
            offers.sort(key=lambda o: float(o.get("price_toman") or 0))

        ranked.append((key, candidate, offers, nearest_distance))

    ranked.sort(key=lambda x: x[0])
    total_matched = len(ranked)
    try:
        off = max(0, int(offset or 0))
    except (TypeError, ValueError):
        off = 0
    try:
        lim = max(1, int(limit))
    except (TypeError, ValueError):
        lim = 50
    page_slice = ranked[off: off + lim]
    results = []
    for _, candidate, offers, nearest_distance in page_slice:
        results.append(_compact_product(
            product_id=candidate["product_id"],
            persian_title=candidate.get("persian_title"),
            brand_name=candidate.get("brand_name"),
            model=candidate.get("model"),
            offers=offers,
            nearest_distance=nearest_distance,
        ))
    if return_total:
        return results, total_matched
    return results
