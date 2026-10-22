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


def _shop_text(repo, offer):
    shop = repo.get_shop(offer.shop_id) if offer.shop_id else None
    if not shop:
        return ""
    return normalize_persian_text(" ".join(str(x or "") for x in
        (shop.name, shop.address, shop.location_label)))


def run_structured_search(repo: DataRepository, spec: Dict[str, Any], limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieve by name, then apply price/location/features as hard filters and rank."""
    spec = _plain(spec)
    product = _plain(spec.get("product", {}))
    price = _plain(spec.get("price", {}))
    location = _plain(spec.get("location", {}))
    options = _plain(spec.get("search_options", {}))
    name = product.get("name") or product.get("product_name") or spec.get("query")
    keywords = product.get("keywords") or spec.get("keywords") or []
    if isinstance(keywords, str):
        keywords = [keywords]
    brand, model = product.get("brand") or spec.get("brand"), product.get("model") or spec.get("model")
    ps = ProductSearcher(repo)
    candidates = ps.retrieve_candidates(name, keywords, brand, model, limit=max(limit*20, 500))

    min_price = price.get("min", spec.get("min_price"))
    max_price = price.get("max", spec.get("max_price"))
    loc_value = location.get("city") or location.get("text")
    if not loc_value and isinstance(spec.get("location"), str):
        loc_value = spec["location"]
    location_query = normalize_persian_text(loc_value)
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

    for candidate in candidates:
        pid = candidate["product_id"]
        p = repo.get_product(pid)
        if not p:
            continue
        offers = []
        for offer in repo.get_offers_by_product(pid):
            if offer.price_toman is None:
                continue
            try:
                amount = float(offer.price_toman)
            except (ValueError, TypeError):
                continue
            if min_price is not None and amount < float(min_price):
                continue
            if max_price is not None and amount > float(max_price):
                continue
            if location_query and location_query not in _shop_text(repo, offer):
                continue
            offers.append({"offer_id": offer.offer_id, "price_toman": offer.price_toman,
                "shop_id": offer.shop_id,
                "last_price_changed_at": offer.last_price_changed_at.isoformat() if offer.last_price_changed_at else None,
                "warranty_recorded": getattr(offer, "warranty", None)})
        # If any offer-level constraint was requested, no matching offer means exclude product.
        if (min_price is not None or max_price is not None or location_query) and not offers:
            continue
        if feature_terms:
            searchable = normalize_persian_text(_flatten(getattr(p, "specs", None)) + " " + (p.persian_title or ""))
            if any(term and term not in searchable for term in feature_terms):
                continue
        rec = candidate["match_record"]
        cheapest = min((float(o["price_toman"]) for o in offers), default=float("inf"))
        if sort_by in ("price_asc", "price_low_to_high", "cheapest", "price"):
            key = (cheapest, rec.tier, -rec.name_relevance_score)
        elif sort_by in ("price_desc", "price_high_to_low", "most_expensive"):
            key = (-cheapest, rec.tier, -rec.name_relevance_score)
        else:
            key = (rec.tier, -rec.name_relevance_score, -ps.token_coverage(rec), cheapest)
        ranked.append((key, candidate, offers, cheapest))

    ranked.sort(key=lambda x: x[0])
    results = []
    for rank, (_, candidate, offers, cheapest) in enumerate(ranked[:max(1, int(limit))], 1):
        rec = candidate["match_record"]
        rec.final_rank = rank
        results.append({"product_id": candidate["product_id"],
            "persian_title": candidate["persian_title"], "brand_name": candidate.get("brand_name"),
            "model": candidate.get("model"), "specs": candidate.get("specs"),
            "offers": sorted(offers, key=lambda o: float(o["price_toman"])),
            "match_record": {"stages": rec.stages, "matched_phrases": rec.matched_phrases,
                "matched_tokens": sorted(rec.matched_tokens),
                "name_relevance_score": rec.name_relevance_score,
                "token_match_count": rec.token_match_count, "token_coverage": ps.token_coverage(rec),
                "tier": rec.tier, "sort_score": round(rec.name_relevance_score + 1000/max(1, rec.tier), 4),
                "final_rank": rank}})
    return results
