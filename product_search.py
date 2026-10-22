from typing import List, Dict, Any, Optional, Tuple
from repository import DataRepository
from difflib import SequenceMatcher
from text_utils import normalize_persian_text, tokenize_persian
import logging

logger = logging.getLogger('shopping_assistant.product_search')


class MatchRecord:
    """Tracks match metadata for a single product candidate across retrieval stages.

    Attributes:
        product_id: The product's unique identifier.
        stages: List of (stage_number, match_type, matched_phrase, matched_tokens).
        matched_tokens: Set of individual tokens that matched from the query.
        matched_phrases: List of phrase strings matched (across all stages).
        query_tokens: Full set of normalized query tokens for coverage computation.
        name_relevance_score: Accumulated relevance score.
        secondary_score: Secondary ranking score (e.g., price-based).
        tier: Lexicographic relevance tier (1=best, 5=weakest).
        final_rank: Final position in the sorted result list.
    """

    def __init__(self, product_id: str):
        self.product_id = product_id
        self.stages = []
        self.matched_tokens = set()
        self.matched_phrases = []
        self.query_tokens = set()
        self.name_relevance_score = 0.0
        self.secondary_score = 0.0
        self.tier = 6
        self.final_rank = 0
        # Number of filler words between the first and last in-order query token
        # when the match came from the in-order stage (0 = contiguous). None otherwise.
        self.gap_words: Optional[int] = None

    def add_match(self, stage: int, match_type: str, phrase: Optional[str], tokens: List[str], boost: float = 0.0):
        self.stages.append((stage, match_type, phrase, list(tokens)))
        if phrase:
            self.matched_phrases.append(phrase)
        for t in tokens:
            self.matched_tokens.add(t)
        self.name_relevance_score += boost


class ProductSearcher:
    """Multi-stage product searcher that uses normalized comparisons and staged retrieval.

    It expects a DataRepository with loaded products and offers. It returns candidate products
    with attached match metadata for traceability.

    The retrieval pipeline works in stages:
      1. Full-phrase search — the entire normalized query as a contiguous substring.
      1b. In-order full-coverage search — every query token present in the same order
          with a small number of filler words allowed between them (the "exact item"
          whose words are separated in the title, e.g. «سیم مسی 240» vs «سیم مسی نمره 240»).
      2. Ordered phrase variations — n-grams preserving word order, preferring long n-grams.
      3. Individual token search — meaningful non-stopword tokens.
      4. Brand/model boosting — match brand or model fields.
      5. Keyword match — weak signal from free-form keywords.

    After retrieval, a lexicographic tier system ranks results so that the whole user
    query always outranks its pieces, and pieces outrank single tokens:
      Tier 1 — Exact normalized title equals a matched phrase.
      Tier 2 — Full query appears as contiguous substring in product fields.
      Tier 3 — Every query token appears in order with a bounded gap (in-order coverage).
      Tier 4 — Ordered n-gram (a piece of the query, not the whole query).
      Tier 5 — Token-level or brand/model match.
      Tier 6 — Weak fallback / keyword-only match.

    Within each tier, the name_relevance_score and token coverage break ties.
    """

    def __init__(self, repo: DataRepository, config: Optional[Dict[str, Any]] = None):
        self.repo = repo
        self.config = config or {}
        # maximum number of filler words allowed between in-order query tokens
        self.max_gap_words = self.config.get('max_gap_words', 3)
        # scoring weights (configurable)
        self.weights = {
            'exact_match': self.config.get('w_exact', 100.0),
            'full_phrase': self.config.get('w_full_phrase', 80.0),
            'ordered_phrase': self.config.get('w_ordered', 40.0),
            'token_match': self.config.get('w_token', 10.0),
            'brand_model': self.config.get('w_brand_model', 30.0),
            'semantic': self.config.get('w_semantic', 5.0),
            'in_order_span': self.config.get('w_in_order_span', 15.0),
            'phrase_prefix': self.config.get('w_phrase_prefix', 20.0),
        }

    def _product_text_fields(self, p) -> Dict[str, str]:
        return {
            'title': p.persian_title or '',
            'brand': p.brand_name or '',
            'model': p.model or ''
        }

    def _normalized_product_hay(self, p) -> str:
        parts = [p.persian_title or '', p.brand_name or '', p.model or '']
        return normalize_persian_text(' '.join([x for x in parts if x]))

    def _stage1_full_phrase(self, query: str) -> List[Tuple[str, str]]:
        """Stage 1: search for the full normalized phrase as contiguous substring in product fields.

        Returns list of tuples (product_id, matched_phrase).
        """
        qn = normalize_persian_text(query)
        res = []
        for p in self.repo._products.values():
            hay = self._normalized_product_hay(p)
            if qn and qn in hay:
                res.append((p.product_id, qn))
        logger.debug('stage1 found %d candidates for %r', len(res), query)
        return res

    def _stage1b_in_order(self, query: str) -> List[Tuple[str, int]]:
        """Stage 1b: every query token present in the same order with bounded filler.

        This finds the "exact item" even when the query words are separated by other
        words in the title. For example, query «سیم مسی 240» should surface
        «سیم مسی نمره 240» (one filler word) before «سیم افشان ارت مسی 240»
        (two filler words). Returns (product_id, gap_words) sorted by tightest gap.
        """
        qn = normalize_persian_text(query)
        tokens = tokenize_persian(qn)
        n = len(tokens)
        if n < 2:
            return []
        res = []
        max_gap = self.max_gap_words
        for p in self.repo._products.values():
            words = self._normalized_product_hay(p).split()
            if len(words) < n:
                continue
            best_gap = None
            starts = [i for i, w in enumerate(words) if w == tokens[0]]
            for s in starts:
                pos = s
                for t in tokens[1:]:
                    try:
                        pos = words.index(t, pos + 1)
                    except ValueError:
                        pos = None
                        break
                if pos is None:
                    continue
                gap = pos - s + 1 - n
                if best_gap is None or gap < best_gap:
                    best_gap = gap
            if best_gap is not None and best_gap <= max_gap:
                res.append((p.product_id, best_gap))
        res.sort(key=lambda x: x[1])
        logger.debug('stage1b found %d in-order candidates for %r', len(res), query)
        return res

    def _ngrams_ordered(self, tokens: List[str], min_len: int = 2) -> List[str]:
        """Generate ordered n-grams of the token list, preferring longer phrases."""
        n = len(tokens)
        phrases = []
        for L in range(min_len, n + 1):
            for i in range(0, n - L + 1):
                phrases.append(' '.join(tokens[i: i + L]))
        # prefer longer phrases first
        phrases.sort(key=lambda x: len(x.split()), reverse=True)
        return phrases

    def _stage2_ordered_phrases(self, query: str, limit_per_phrase: int = 50) -> List[Tuple[str, str]]:
        """Search contiguous ordered n-grams of the query, prefer longer phrases."""
        qn = normalize_persian_text(query)
        tokens = tokenize_persian(qn)
        phrases = self._ngrams_ordered(tokens)
        res = []
        seen = set()
        for ph in phrases:
            for p in self.repo._products.values():
                hay = self._normalized_product_hay(p)
                if ph in hay:
                    if p.product_id not in seen:
                        res.append((p.product_id, ph))
                        seen.add(p.product_id)
                        if len(res) >= limit_per_phrase:
                            break
            if len(res) >= limit_per_phrase:
                break
        logger.debug('stage2 found %d candidates', len(res))
        return res

    def _stage3_tokens(self, query: str, min_token_matches: int = 2, top_k: int = 200) -> List[Tuple[str, List[str]]]:
        """Search by individual meaningful tokens (exclude stopwords).

        To avoid returning noisy results for arbitrary or unknown queries, require at least
        two token matches by default. Tests that need token-only matches can call with lower min.
        """
        qn = normalize_persian_text(query)
        tokens = tokenize_persian(qn)
        res = []
        for p in self.repo._products.values():
            hay = self._normalized_product_hay(p)
            matched = [t for t in tokens if t in hay]
            if len(matched) >= min_token_matches:
                res.append((p.product_id, matched))
        # sort by number of matched tokens desc
        res.sort(key=lambda x: len(x[1]), reverse=True)
        logger.debug('stage3 found %d candidates', len(res))
        return res[:top_k]

    def retrieve_candidates(
        self,
        product_name: Optional[str],
        keywords: List[str] = None,
        brand: Optional[str] = None,
        model: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Run multi-stage retrieval and merge deduplicated candidate list with match metadata.

        Stages are executed in order. Each subsequent stage is skipped if there are already
        enough *quality* candidates (phrase-level matches) to fill the limit, so that noisy
        token-only matches do not dilute a strong phrase-match result set.

        Returns candidate dicts with a 'match_record' key holding a MatchRecord instance.
        """
        if not product_name and not keywords and not brand and not model:
            return []

        # Pre-compute query tokens for coverage computation
        qn = normalize_persian_text(product_name or '')
        query_tokens = set(tokenize_persian(qn))

        candidates: Dict[str, MatchRecord] = {}

        # Stage 1: full-phrase search
        if product_name:
            s1 = self._stage1_full_phrase(product_name)
            for pid, phrase in s1:
                m = candidates.get(pid) or MatchRecord(pid)
                m.query_tokens = query_tokens
                m.add_match(
                    stage=1, match_type='full_phrase', phrase=phrase,
                    tokens=tokenize_persian(phrase),
                    boost=self.weights['full_phrase']
                )
                if pid not in candidates:
                    candidates[pid] = m

        # Quality check: count how many candidates came from phrase-level stages (1, 1b or 2)
        def _quality_phrase_count() -> int:
            """Count candidates that have at least one phrase-level stage (stage 1, 1b or 2)."""
            count = 0
            for rec in candidates.values():
                if any(s[0] in (1, 2) for s in rec.stages):
                    count += 1
            return count

        # Stage 1b: in-order full-query coverage (the exact item, words in order with a small gap)
        # Runs before n-gram pieces so the whole query outranks partial pieces.
        if product_name and _quality_phrase_count() < limit:
            for pid, gap in self._stage1b_in_order(product_name):
                if pid in candidates:
                    continue
                m = MatchRecord(pid)
                m.query_tokens = query_tokens
                m.gap_words = gap
                m.add_match(
                    stage=1, match_type='in_order_phrase', phrase=qn or None,
                    tokens=tokenize_persian(qn or ''),
                    boost=self.weights['full_phrase']
                    + self.weights['in_order_span'] * (self.max_gap_words - gap)
                )
                candidates[pid] = m

        # Stage 2: ordered phrases — skip if we already have enough quality candidates
        if product_name and _quality_phrase_count() < limit:
            s2 = self._stage2_ordered_phrases(product_name)
            ntok = max(len(query_tokens), 1)
            for pid, phrase in s2:
                if pid in candidates:
                    continue
                m = MatchRecord(pid)
                m.query_tokens = query_tokens
                # Boost scaled by how much of the query the piece covers: a longer
                # piece of the query outranks a shorter piece at the same tier.
                piece_len = len(tokenize_persian(phrase))
                m.add_match(
                    stage=2, match_type='ordered_phrase', phrase=phrase,
                    tokens=tokenize_persian(phrase),
                    boost=self.weights['ordered_phrase'] * (piece_len / ntok)
                )
                candidates[pid] = m

        # Stage 3: individual tokens — skip if quality candidates already satisfy the limit
        if product_name and _quality_phrase_count() < limit:
            s3 = self._stage3_tokens(product_name)
            for pid, matched in s3:
                if pid in candidates:
                    continue
                m = MatchRecord(pid)
                m.query_tokens = query_tokens
                m.add_match(
                    stage=3, match_type='token_match', phrase=None,
                    tokens=matched,
                    boost=self.weights['token_match'] * len(matched)
                )
                candidates[pid] = m

        # Stage 4: Brand/model boosting
        if brand or model:
            bm_q = ' '.join([x for x in [brand, model] if x])
            bm_norm = normalize_persian_text(bm_q)
            for p in self.repo._products.values():
                hay = self._normalized_product_hay(p)
                if bm_norm and bm_norm in hay:
                    if p.product_id not in candidates:
                        m = MatchRecord(p.product_id)
                        m.query_tokens = query_tokens
                        m.add_match(
                            stage=4, match_type='brand_model', phrase=bm_norm,
                            tokens=tokenize_persian(bm_norm),
                            boost=self.weights['brand_model']
                        )
                        candidates[p.product_id] = m
                    else:
                        candidates[p.product_id].add_match(
                            stage=4, match_type='brand_model', phrase=bm_norm,
                            tokens=tokenize_persian(bm_norm),
                            boost=self.weights['brand_model']
                        )

        # Stage 5: Keywords (weak signal)
        if keywords:
            for kw in keywords:
                kn = normalize_persian_text(kw)
                for p in self.repo._products.values():
                    hay = self._normalized_product_hay(p)
                    if kn in hay:
                        if p.product_id not in candidates:
                            m = MatchRecord(p.product_id)
                            m.query_tokens = query_tokens
                            m.add_match(
                                stage=5, match_type='keyword', phrase=kn,
                                tokens=tokenize_persian(kn),
                                boost=self.weights['token_match'] / 2
                            )
                            candidates[p.product_id] = m
                        else:
                            candidates[p.product_id].add_match(
                                stage=5, match_type='keyword', phrase=kn,
                                tokens=tokenize_persian(kn),
                                boost=self.weights['token_match'] / 2
                            )

        # Build final candidate list with metadata
        out = []
        for pid, rec in candidates.items():
            p = self.repo.get_product(pid)
            if not p:
                continue
            out.append({
                'product_id': pid,
                'persian_title': p.persian_title,
                'brand_name': p.brand_name,
                'model': p.model,
                'specs': p.specs,
                'match_record': rec,
            })
        return out

    def score_name_relevance(self, rec: MatchRecord, product_obj) -> float:
        """Compute name relevance from match record and product fields.

        Avoids double-counting: boosts were applied incrementally at match time.
        Additional heuristics applied here:
          - Exact title match bonus (when normalized title == matched phrase).
          - Token coverage bonus (fraction of query tokens present in the title).
        """
        base = rec.name_relevance_score

        # Exact title match bonus: if any matched phrase equals the entire normalized title
        title_norm = normalize_persian_text(product_obj.persian_title or '')
        extra = 0.0
        for ph in rec.matched_phrases:
            if ph and title_norm == ph:
                extra += self.weights['exact_match']

        # Phrase-prefix bonus: the product title starts with a matched phrase at a word
        # boundary, i.e. the item IS the searched phrase followed by more detail
        # («سیم مسی نمره ۲۴۰» for query «سیم مسی»). Stronger than a mid-title occurrence.
        prefix_bonus = 0.0
        for ph in rec.matched_phrases:
            if ph and title_norm.startswith(ph + ' '):
                prefix_bonus = max(prefix_bonus, self.weights['phrase_prefix'])

        # Token coverage bonus: fraction of *all query tokens* present in the product title
        coverage_bonus = 0.0
        if rec.query_tokens:
            present = 0
            for t in rec.query_tokens:
                if t and t in title_norm:
                    present += 1
            cov = present / len(rec.query_tokens)
            coverage_bonus = cov * self.weights['full_phrase']

        score = base + extra + coverage_bonus + prefix_bonus
        return float(score)

    def compute_tier(self, rec: MatchRecord, product_obj) -> int:
        """Compute a lexicographic relevance tier for the given match record.

        Tiers (1 = best, 6 = weakest), mirroring the user's priority ladder —
        the whole query first, then pieces, then individual words:
          1. Exact title match — the product's normalized title equals a matched phrase exactly.
          2. Full-phrase in product fields — the entire query appears as contiguous
             substring (stage 1, 'full_phrase').
          3. In-order full coverage — every query token appears in the title in the
             same order with a bounded gap (stage 1, 'in_order_phrase'): the exact
             item whose words are separated by filler words.
          4. Ordered n-gram — a piece of the query matched contiguously (stage 2).
          5. Token-level or brand/model match — individual tokens (stage 3) or
             brand/model (stage 4) with no higher-level phrase evidence.
          6. Weak fallback — keyword-only (stage 5) or no meaningful evidence.
        """
        if not rec.stages:
            return 6

        title_norm = normalize_persian_text(product_obj.persian_title or '')
        stage_nums = set(s[0] for s in rec.stages)
        has_type = lambda t: any(s[1] == t for s in rec.stages)

        # Tier 1: Exact normalized title matches a matched phrase
        for ph in rec.matched_phrases:
            if ph and title_norm == ph:
                return 1

        # Tier 2: Full query phrase found as contiguous substring (stage 1, full_phrase)
        if has_type('full_phrase'):
            return 2

        # Tier 3: Full query in order with bounded gap (stage 1, in_order_phrase)
        if has_type('in_order_phrase'):
            return 3

        # Tier 4: Ordered n-gram piece (stage 2)
        if 2 in stage_nums:
            return 4

        # Tier 5: Token-level (stage 3) or brand/model (stage 4)
        if 3 in stage_nums or 4 in stage_nums:
            return 5

        # Tier 6: Keyword or fallback
        return 6

    def token_coverage(self, rec: MatchRecord) -> float:
        """Compute fraction of query tokens present in matched_tokens."""
        if not rec.query_tokens:
            return 0.0
        matched = rec.matched_tokens & rec.query_tokens
        return len(matched) / len(rec.query_tokens)


# ---------------------------------------------------------------------------
# Convenience function used by orchestrator
# ---------------------------------------------------------------------------

def run_structured_search(repo: DataRepository, spec: Dict[str, Any], limit: int = 50) -> List[Dict[str, Any]]:
    """Run the full search pipeline: retrieve, score, rank, filter, and return sorted results.

    Pipeline stages:
      1. Parse the SearchSpec-like dict.
      2. Multi-stage candidate retrieval (ProductSearcher.retrieve_candidates).
      3. Name-relevance scoring and tier computation.
      4. Filter noisy stopword-only matches.
      5. Lexicographic sort by (tier, -name_relevance_score, -token_coverage).
      6. Apply price constraints (hard filters — products with no valid offers are dropped).
      7. Return top-N result dicts with match record metadata.

    The lexicographic sort ensures name relevance dominates ranking:
      - Primary key: tier (lower = better)
      - Secondary key: name_relevance_score (higher = better)
      - Tertiary key: token_coverage (higher = better)
    """
    ps = ProductSearcher(repo)

    # Parse spec dict
    prod_name = None
    keywords = []
    brand = None
    model = None
    if spec and isinstance(spec, dict):
        prod = spec.get('product', {})
        prod_name = prod.get('name')
        keywords = prod.get('keywords', [])
        brand = prod.get('brand')
        model = prod.get('model')

    # Stage A & B: normalize and retrieve candidates
    candidates = ps.retrieve_candidates(
        product_name=prod_name, keywords=keywords,
        brand=brand, model=model, limit=limit
    )

    # Stage D: score each candidate
    for c in candidates:
        rec = c.get('match_record')
        p = repo.get_product(c.get('product_id'))
        if p and rec:
            rec.name_relevance_score = ps.score_name_relevance(rec, p)
            rec.tier = ps.compute_tier(rec, p)

    # Filter noisy token-only matches (stopword-only)
    from text_utils import PERSIAN_STOPWORDS
    filtered = []
    for c in candidates:
        rec = c.get('match_record')
        has_phrase = any(s[0] in (1, 2) for s in rec.stages)
        if has_phrase:
            filtered.append(c)
            continue
        # Token-only matches: keep if at least one non-stopword token matched
        non_stop = [t for t in rec.matched_tokens if t and t not in PERSIAN_STOPWORDS]
        if non_stop:
            filtered.append(c)
        else:
            logger.debug('Dropping noisy candidate %s — stopword-only token matches', c.get('product_id'))

    # Stage F: lexicographic sort — tier first, then score, then coverage
    filtered.sort(key=lambda x: (
        x['match_record'].tier,
        -x['match_record'].name_relevance_score,
        -ps.token_coverage(x['match_record'])
    ))

    # Apply price constraints (hard filters)
    price_min = None
    price_max = None
    if isinstance(spec, dict):
        price = spec.get('price', {})
        price_min = price.get('min')
        price_max = price.get('max')

    out = []
    for c in filtered[:limit]:
        pid = c.get('product_id')
        p = repo.get_product(pid)
        offers = repo.get_offers_by_product(pid)
        offer_dicts = []
        for o in offers:
            if price_min is not None and o.price_toman is not None and o.price_toman < price_min:
                continue
            if price_max is not None and o.price_toman is not None and o.price_toman > price_max:
                continue
            offer_dicts.append({
                'offer_id': o.offer_id,
                'price_toman': o.price_toman,
                'shop_id': o.shop_id,
            })
        if not offer_dicts:
            logger.debug('Product %s excluded due to price constraints or no offers', pid)
            continue

        # Assign final rank
        rec = c['match_record']
        rec.final_rank = len(out) + 1

        out.append({
            'product_id': pid,
            'persian_title': p.persian_title,
            'offers': sorted(offer_dicts, key=lambda x: x['price_toman'] if x.get('price_toman') is not None else 10**18),
            'match_record': {
                'stages': rec.stages,
                'matched_phrases': rec.matched_phrases,
                'matched_tokens': list(rec.matched_tokens),
                'name_relevance_score': rec.name_relevance_score,
                'tier': rec.tier,
                'final_rank': rec.final_rank,
            }
        })
    return out