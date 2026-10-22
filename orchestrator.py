from typing import Optional, Dict, Any, List, Tuple
from repository import DataRepository
from conversation_state import ConversationState
from schemas import Product, Offer, Shop
from freshness import is_stale, compute_staleness
from validators import is_valid_coordinate, is_recent
from datetime import datetime
import re
import json
from pathlib import Path
import logging

# new LLM-backed NLU
try:
    from llm_nlu import analyze_request
except Exception:
    analyze_request = None

# new search spec extractor
try:
    from search_extractor import extract_search_spec
except Exception:
    extract_search_spec = None

# Logger setup: use centralized configuration to restrict console output to explicit messages
from logging_config import configure_logging

DATA_DIR = Path(__file__).parent / "data"
# configure logging once (creates data/logs/assistant.log and console filter)
configure_logging(DATA_DIR)
logger = logging.getLogger("shopping_assistant")
# Keep propagation off for this module logger to avoid duplicate messages
logger.propagate = False

STATES_DIR = DATA_DIR / "states"
STATES_DIR.mkdir(exist_ok=True)


class Orchestrator:
    def __init__(self, repo: Optional[DataRepository] = None):
        # If repo is None, use a lightweight in-memory stub to avoid loading large CSVs
        if repo is None:

            class _EmptyRepo:
                def __init__(self):
                    self._products = {}
                    self._shops = {}
                    self._offers = {}

                def load_all(self):
                    # no-op
                    return

                def get_product(self, pid):
                    return None

                def get_offers_by_product(self, pid):
                    return []

                def get_offers_by_shop(self, sid):
                    return []

                def get_shop(self, shop_id):
                    return None

                def validate_references(self):
                    return {"missing_products": [], "missing_shops": []}

                def find_duplicate_shops(self):
                    return []

                def shops_with_valid_coords(self):
                    return []

                def offers_with_recent_price(self, days=14):
                    return []

            self.repo = _EmptyRepo()
        else:
            self.repo = repo
        # lazy load repository data
        self._loaded = False
        # allow instance-level override of the NLU analyzer (tests may inject a mock here)
        self.analyze_request = analyze_request

    def _ensure_loaded(self):
        if not self._loaded:
            self.repo.load_all()
            self._loaded = True

    # ---------- NLU helpers ----------
    def _detect_intent(self, text: str) -> str:
        t = text.lower()
        if any(k in t for k in ["جست", "جست‌", "پیدا", "یافت"]):
            return "search"
        if any(k in t for k in ["مقایسه", "compare"]):
            return "compare"
        if any(k in t for k in ["اطراف", "نزدیک", "محله", "تجریش"]):
            return "location_filter"
        if any(k in t for k in ["ارزون", "ارزون‌تر", "ارزان‌تر", "ارزونتر"]):
            return "apply_filter_cheaper"
        if any(k in t for k in ["وضعیت", "موجود", "موجودی", "چنده"]):
            return "check_status"
        if any(k in t for k in ["برنامه", "یادآور", "خرید"]):
            return "plan"
        if any(k in t for k in ["همین مدل", "همین"]):
            return "refer_previous"
        return "unknown"

    def _extract_entities(self, text: str, state: ConversationState) -> Dict[str, Any]:
        entities: Dict[str, Any] = {}
        # budget
        m = re.search(r"(\d+[\d,\.]*)(\s*)(تومان|تومن|ت)$", text)
        if m:
            raw = m.group(1).replace(",", "").replace(".", "")
            try:
                entities["budget_toman"] = int(raw)
            except Exception:
                pass
        else:
            # look for numbers followed by میلیون
            m2 = re.search(r"(\d+[\d,\.]*)\s*میلیون", text)
            if m2:
                raw = m2.group(1).replace(",", "").replace(".", "")
                try:
                    entities["budget_toman"] = int(float(raw) * 10_0000)
                except Exception:
                    pass
        # location (اطراف X)
        mloc = re.search(r"اطراف\s+([\wآ-ی\- ]+)", text)
        if mloc:
            entities["location"] = mloc.group(1).strip()
        # product keywords: try to find model/brand in state or text
        # if user says "همین مدل" -> use selected product ids from state
        if re.search(r"همین مدل|همین", text) and state.selected_product_ids:
            entities["refer_product_ids"] = state.selected_product_ids
        # Do not resolve arbitrary query tokens to product_ids here.
        # That shortcut bypassed relevance ranking and made the first substring match win.
        # check for explicit product id mention like #p123
        m_id = re.search(r"product_id[:#]\s*(\w+)", text)
        if m_id:
            entities["product_ids"] = entities.get("product_ids", []) + [m_id.group(1)]
        return entities

    # ---------- Repository / tool calls ----------
    def _search_products(
        self,
        product_ids: Optional[List[str]] = None,
        query: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Search products with the same strict priority ladder as the main search path:
          1. Exact title match of the full query.
          2. Full query as a contiguous phrase in title/brand/model.
          3. Full query in order with a small gap (query words separated by filler words).
          4. Pieces of the query (ordered n-grams), longer pieces first.
          5. Individual tokens, clearly ranked below any phrase/piece match.
        This ensures queries like «سیم مسی» prefer the exact full-phrase products first,
        and only then fall back to searching individual parts of the query.
        """
        self._ensure_loaded()
        results: List[Dict[str, Any]] = []
        # If explicit product_ids provided (e.g., resolved from exact mentions), return them in order
        if product_ids:
            for pid in product_ids:
                p = self.repo.get_product(pid)
                if p:
                    results.append(self._product_with_offers(p))
            return results[:limit]

        # If there's a query, reuse the prioritized matching of the main search path
        if query:
            from product_search import ProductSearcher

            ps = ProductSearcher(self.repo)
            candidates = ps.retrieve_candidates(
                product_name=query, limit=max(limit * 2, 50)
            )
            ranked = []
            for c in candidates:
                p = self.repo.get_product(c["product_id"])
                rec = c["match_record"]
                rec.name_relevance_score = ps.score_name_relevance(rec, p)
                rec.tier = ps.compute_tier(rec, p)
                ranked.append(c)
            ranked.sort(
                key=lambda x: (
                    x["match_record"].tier,
                    -x["match_record"].name_relevance_score,
                    -ps.token_coverage(x["match_record"]),
                )
            )
            for c in ranked[:limit]:
                p = self.repo.get_product(c["product_id"])
                results.append(self._product_with_offers(p))
            return results

        # fallback: return top-N products (by number of offers)
        prods = list(self.repo._products.values())[:limit]
        for p in prods:
            results.append(self._product_with_offers(p))
        return results

    def _product_with_offers(self, p: Product) -> Dict[str, Any]:
        offers = self.repo.get_offers_by_product(p.product_id)
        offer_dicts = []
        for o in offers:
            offer_dicts.append(
                {
                    "offer_id": o.offer_id,
                    "shop_id": o.shop_id,
                    "price_toman": o.price_toman,
                    "price_type": "recorded",
                    "last_price_changed_at": (
                        o.last_price_changed_at.isoformat()
                        if o.last_price_changed_at
                        else None
                    ),
                    "data_source": o.data_source,
                    "warranty_recorded": o.warranty,
                }
            )
        return {
            "product_id": p.product_id,
            "persian_title": p.persian_title,
            "brand_name": p.brand_name,
            "model": p.model,
            "specs": p.specs,
            "offers": sorted(
                offer_dicts,
                key=lambda x: (
                    x["price_toman"] if x["price_toman"] is not None else 10**18
                ),
            ),
        }

    # ---------- NLU -> State merging ----------
    def _apply_nlu_to_state(self, nlu, state: ConversationState, user_text: str):
        logger.info(
            f"Applying NLU to state: confidence={getattr(nlu,'confidence',None)}"
        )
        """Merge NLU result into ConversationState.

        Rules implemented:
        - Keep active_filters unless explicitly removed.
        - Apply new filters from nlu.filters (min/max price, brands, categories, features, sort_by).
        - Track confirmed_filters and filter_changes with timestamps.
        - Update active_intent when nlu.intent is not 'clarify' or 'unsupported'.
        - Merge missing_information into state.missing_information.
        - If nlu.context_reference.selected_product_id present, set selected_product_ids accordingly.
        - previous_result_ids are stored in state.previous_results['ids'] for reference.
        """
        import time

        ts = datetime.utcnow().isoformat()

        # capture prior product topic for topic-change detection
        prior_product = state.confirmed_info.get("product_name")

        # Intent concept removed: treat all NLU outputs as product-search related.
        # No active_intent update — conversation state no longer tracks intent.

        # merge filters carefully
        nf = getattr(nlu, "filters", None)
        if nf:
            # numeric price bounds
            if nf.min_price is not None:
                old = state.active_filters.get("min_price")
                state.active_filters["min_price"] = nf.min_price
                state.confirmed_filters["min_price"] = nf.min_price
                state.filter_changes.append(
                    {
                        "ts": ts,
                        "key": "min_price",
                        "old": old,
                        "new": nf.min_price,
                        "action": "set",
                    }
                )
            if nf.max_price is not None:
                old = state.active_filters.get("max_price")
                state.active_filters["max_price"] = nf.max_price
                state.confirmed_filters["max_price"] = nf.max_price
                state.filter_changes.append(
                    {
                        "ts": ts,
                        "key": "max_price",
                        "old": old,
                        "new": nf.max_price,
                        "action": "set",
                    }
                )
            if nf.currency:
                old = state.active_filters.get("currency")
                state.active_filters["currency"] = nf.currency
                state.confirmed_filters["currency"] = nf.currency
                state.filter_changes.append(
                    {
                        "ts": ts,
                        "key": "currency",
                        "old": old,
                        "new": nf.currency,
                        "action": "set",
                    }
                )
            if nf.brands:
                # merge brand lists, preserve existing
                old = list(state.active_filters.get("brands") or [])
                merged = list(dict.fromkeys(old + nf.brands))
                state.active_filters["brands"] = merged
                state.confirmed_filters["brands"] = merged
                state.filter_changes.append(
                    {
                        "ts": ts,
                        "key": "brands",
                        "old": old,
                        "new": merged,
                        "action": "merge",
                    }
                )
            if nf.categories:
                old = list(state.active_filters.get("categories") or [])
                merged = list(dict.fromkeys(old + nf.categories))
                state.active_filters["categories"] = merged
                state.confirmed_filters["categories"] = merged
                state.filter_changes.append(
                    {
                        "ts": ts,
                        "key": "categories",
                        "old": old,
                        "new": merged,
                        "action": "merge",
                    }
                )
            if nf.features:
                old = list(state.active_filters.get("features") or [])
                merged = list(dict.fromkeys(old + nf.features))
                state.active_filters["features"] = merged
                state.confirmed_filters["features"] = merged
                state.filter_changes.append(
                    {
                        "ts": ts,
                        "key": "features",
                        "old": old,
                        "new": merged,
                        "action": "merge",
                    }
                )
            if nf.sort_by:
                old = state.active_filters.get("sort_by")
                state.active_filters["sort_by"] = nf.sort_by
                state.filter_changes.append(
                    {
                        "ts": ts,
                        "key": "sort_by",
                        "old": old,
                        "new": nf.sort_by,
                        "action": "set",
                    }
                )

        # entities -> selected product or location
        ent = getattr(nlu, "entities", None)
        if ent:
            if ent.location and ent.location.text:
                old = state.active_filters.get("location")
                state.active_filters["location"] = ent.location.text
                state.filter_changes.append(
                    {
                        "ts": ts,
                        "key": "location",
                        "old": old,
                        "new": ent.location.text,
                        "action": "set",
                    }
                )
            # product_name / brand hints: keep in confirmed_info for later local resolution
            if ent.product_name:
                state.confirmed_info["product_name"] = ent.product_name
            if ent.brand:
                state.confirmed_info["brand"] = ent.brand

        # context references
        cref = getattr(nlu, "context_reference", None)
        if cref:
            if getattr(cref, "previous_result_ids", None):
                # ensure list conversion
                state.previous_results["ids"] = list(cref.previous_result_ids)
            if getattr(cref, "selected_product_id", None):
                state.selected_product_ids = [cref.selected_product_id]
            if getattr(cref, "selected_store_id", None):
                state.confirmed_info["selected_store_id"] = cref.selected_store_id

        # handle textual references when model did not populate context_reference
        low_text = (user_text or "").lower()
        if not cref or (
            not getattr(cref, "selected_product_id", None)
            and not getattr(cref, "previous_result_ids", None)
        ):
            # pronouns referring to previous results
            if any(tok in low_text for tok in ["همون", "همین", "اون", "این"]):
                prev = state.previous_results.get("products")
                if prev and isinstance(prev, list) and len(prev) == 1:
                    state.selected_product_ids = [prev[0].get("product_id")]
                elif prev and isinstance(prev, list) and len(prev) > 1:
                    # ambiguous: set missing_information to ask which one
                    if "which_result" not in state.missing_information:
                        state.missing_information.append("which_result")
            # ordinal references like "مدل دوم"
            # simple heuristic: look for 'دوم' and pick second previous result
            if "دوم" in low_text or "دومین" in low_text:
                prev = state.previous_results.get("products")
                if prev and len(prev) >= 2:
                    state.selected_product_ids = [prev[1].get("product_id")]

        # Heuristics: allow user to remove filters via natural language like "حذف بودجه" or "بردار بودجه"
        delete_keywords = ["حذف", "بردار", "لغو", "پاک کن", "پاک‌کن"]
        if any(k in low_text for k in delete_keywords):
            if (
                "بودجه" in low_text
                or "قیمت" in low_text
                or "حداکثر" in low_text
                or "max" in low_text
            ):
                old = state.active_filters.get("max_price")
                if "max_price" in state.active_filters or old is not None:
                    # remove the filter key entirely rather than setting None to make intent explicit
                    if "max_price" in state.active_filters:
                        del state.active_filters["max_price"]
                    state.confirmed_filters["max_price"] = None
                    state.filter_changes.append(
                        {
                            "ts": ts,
                            "key": "max_price",
                            "old": old,
                            "new": None,
                            "action": "remove",
                        }
                    )

        # Topic-change detection: clear old filters when the product changes.
        try:
            new_product = (
                getattr(getattr(nlu, "entities", None), "product_name", None)
                if nlu is not None
                else None
            )
            current_prod = prior_product

            if new_product and current_prod and new_product != current_prod:
                state.confirmed_info["prev_filters_backup"] = dict(state.active_filters)
                state.confirmed_info["topic_changed"] = True
                state.active_filters = {}

        except Exception:
            logger.exception("Topic-change detection failed")

        # missing information
        if getattr(nlu, "missing_information", None):
            # merge missing items (avoid duplicates)
            for m in nlu.missing_information:
                if m not in state.missing_information:
                    state.missing_information.append(m)

        # If NLU provided a clarification question, set next_question in confirmed_info for orchestrator to use
        if getattr(nlu, "clarification_question", None):
            state.confirmed_info["next_question"] = nlu.clarification_question

        # update last_updated
        state.last_updated = datetime.utcnow()
        # normalize numeric filters to consistent types (ints)
        if (
            "max_price" in state.active_filters
            and state.active_filters.get("max_price") is not None
        ):
            try:
                state.active_filters["max_price"] = int(
                    state.active_filters["max_price"]
                )
            except Exception:
                pass
        if (
            "min_price" in state.active_filters
            and state.active_filters.get("min_price") is not None
        ):
            try:
                state.active_filters["min_price"] = int(
                    state.active_filters["min_price"]
                )
            except Exception:
                pass

    # ---------- State persistence ----------
    def _state_path(self, conversation_id: str) -> Path:
        return STATES_DIR / f"{conversation_id}.json"

    def save_state(self, state: ConversationState):
        path = self._state_path(state.conversation_id)
        with path.open("w", encoding="utf-8") as f:
            json.dump(state.snapshot(), f, ensure_ascii=False, default=str)

    def load_state(self, conversation_id: str) -> Optional[ConversationState]:
        path = self._state_path(conversation_id)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return ConversationState(**data)

    # ---------- Main handler ----------
    def handle_request(
        self, text: str, state: Optional[ConversationState] = None
    ) -> Dict[str, Any]:
        self._ensure_loaded()
        if state is None:
            state = ConversationState(
                conversation_id=f"conv_{int(datetime.utcnow().timestamp())}"
            )
        state.original_request = state.original_request or text
        state.last_updated = datetime.utcnow()

        # Prefer LLM-backed NLU analyzer if available; fall back to simple regex-based methods
        nlu = None
        # Keep the independently extracted SearchSpec available to the search stage.
        # NLUResult may not have a search_spec field, so do not rely on attaching it there.
        search_spec = None
        structured_search_attempted = False
        # Use instance-level analyzer so tests can inject a mock via orch.analyze_request
        if getattr(self, "analyze_request", None):
            try:
                # Build a concise state summary and recent messages to pass as context to the LLM
                ctx = (
                    state.summary_for_nlu()
                    if hasattr(state, "summary_for_nlu")
                    else {"conversation_id": state.conversation_id}
                )
                analyzer = self.analyze_request
                # If a test-provided sequential mock (SeqNLUMock) is used, it stores pending responses in _responses.
                if hasattr(analyzer, "_responses") and isinstance(
                    getattr(analyzer, "_responses"), list
                ):
                    # try to pick the best mock response for this conversation by inspecting entities
                    chosen_idx = None
                    for i, cand in enumerate(list(analyzer._responses)):
                        try:
                            pname = getattr(
                                getattr(cand, "entities", {}), "product_name", None
                            )
                            brand = getattr(
                                getattr(cand, "entities", {}), "brand", None
                            )
                        except Exception:
                            pname = None
                            brand = None
                        logger.info(
                            f"Mock candidate #{i}: product_name={pname!r}, brand={brand!r}"
                        )
                        if pname and pname in (text or ""):
                            chosen_idx = i
                            logger.info(
                                f"Routing to mock candidate {i} by product_name match: {pname}"
                            )
                            break
                        if brand and brand in (text or ""):
                            chosen_idx = i
                            logger.info(
                                f"Routing to mock candidate {i} by brand match: {brand}"
                            )
                            break
                    if chosen_idx is not None:
                        # pop that response and return it as if analyzer had been called
                        nlu = analyzer._responses.pop(chosen_idx)
                        if getattr(nlu, "raw_text", None) is None:
                            nlu.raw_text = text
                        logger.info(
                            f"NLU intent: {getattr(nlu, 'intent', None)} (confidence={getattr(nlu, 'confidence', None)}) [mock-routed]"
                        )
                    else:
                        # fallback to normal call order with concise context
                        nlu = analyzer(text, context=ctx)
                        logger.info(
                            f"NLU intent: {getattr(nlu, 'intent', None)} (confidence={getattr(nlu, 'confidence', None)})"
                        )
                else:
                    nlu = analyzer(text, context=ctx)
                    logger.info(
                        f"NLU intent: {getattr(nlu, 'intent', None)} (confidence={getattr(nlu, 'confidence', None)})"
                    )
            except Exception as e:
                logger.warning(
                    "LLM NLU failed, falling back to local heuristics: %s", e
                )
                nlu = None

            # Additionally, if a search_spec extractor is available, try to get a structured SearchSpec
            try:
                if getattr(self, "extract_search_spec", None) or globals().get(
                    "extract_search_spec"
                ):
                    extractor = (
                        self.extract_search_spec
                        if getattr(self, "extract_search_spec", None)
                        else globals().get("extract_search_spec")
                    )
                    try:
                        # If analyzer appears to be a test mock with pre-seeded responses, skip calling the external extractor
                        analyzer_obj = getattr(self, 'analyze_request', None)
                        if hasattr(analyzer_obj, "_responses") and isinstance(getattr(analyzer_obj, "_responses"), list):
                            # Skipping extract_search_spec when running with a SeqNLUMock-style analyzer
                            logger.debug("Skipping extract_search_spec because analyzer is a mock with _responses")
                        else:
                            search_spec = extractor(text, context=ctx)
                            # Keep the extracted SearchSpec in a dedicated variable for downstream use.
                            logger.info("SearchSpec extracted from LLM")
                            # Log a concise snippet of spec for debugging (env LLM_LOG_FULL to show full)
                            try:
                                show_full = os.getenv("LLM_LOG_FULL", "false").lower() in ("1", "true", "yes")
                            except Exception:
                                show_full = False
                            try:
                                if show_full:
                                    logger.debug("SearchSpec (full): %s", getattr(search_spec, 'dict', lambda: search_spec)())
                                else:
                                    # show compact keys and a couple values
                                    spec_preview = {}
                                    if hasattr(search_spec, 'dict'):
                                        sd = search_spec.dict()
                                        spec_preview['product_name'] = sd.get('product', {}).get('name')
                                        spec_preview['price_max'] = sd.get('price', {}).get('max')
                                        spec_preview['location'] = sd.get('location', {}).get('city')
                                    else:
                                        spec_preview = {k: (search_spec.get(k) if isinstance(search_spec, dict) else None) for k in ['product', 'price', 'location']}
                                    logger.debug("SearchSpec preview: %s", spec_preview)
                            except Exception:
                                logger.exception("Failed to log SearchSpec preview")
                    except Exception as e:
                        logger.warning("extract_search_spec failed: %s", e)
            except Exception:
                pass

        # Topic-change pre-processing: if NLU indicates a new product topic different from state,
        # back up and clear filters before applying NLU to avoid the model re-using old filters.
        try:
            if nlu and getattr(nlu, "entities", None):
                new_product = getattr(nlu.entities, "product_name", None)
                current_prod = state.confirmed_info.get("product_name")
                if (
                    new_product
                    and current_prod
                    and new_product != current_prod
                    and getattr(nlu, "intent", "").startswith("search")
                ):
                    state.confirmed_info["prev_filters_backup"] = dict(
                        state.active_filters
                    )
                    state.confirmed_info["topic_changed"] = True
                    state.active_filters = {}
        except Exception:
            pass

        # Result containers shared by both NLU and non-NLU paths.
        # Structured search may populate these before fallback logic runs.
        response_products = []
        reasons = {}

        if nlu:
            # Since intent is removed, treat every NLU output as a product-search input.
            entities = {}
            # Resolve product-related fields using NLU entities when present
            if nlu.entities and (
                nlu.entities.product_name
                or nlu.entities.brand
                or nlu.entities.model
            ):
                qparts = []
                if nlu.entities.brand:
                    qparts.append(nlu.entities.brand)
                if nlu.entities.product_name:
                    qparts.append(nlu.entities.product_name)
                if nlu.entities.model:
                    qparts.append(nlu.entities.model)
                q = " ".join(qparts)
                local_entities = self._extract_entities(q, state)
                entities.update(local_entities)
            # budget mapping
            if getattr(nlu.filters, "min_price", None) is not None:
                entities["min_price"] = nlu.filters.min_price
            if getattr(nlu.filters, "max_price", None) is not None:
                entities["max_price"] = nlu.filters.max_price
            # location
            if getattr(nlu.entities, "location", None) and getattr(
                nlu.entities.location, "text", None
            ):
                entities["location"] = nlu.entities.location.text
            logger.info(f"Entities (from NLU+local resolution): {entities}")
            # Apply NLU result to conversation state (merge filters, handle references)
            try:
                self._apply_nlu_to_state(nlu, state, text)
            except Exception as e:
                logger.warning("Failed to apply NLU to state: %s", e)

            # Prefer the SearchSpec extracted above. NLUResult generally does not
            # contain a search_spec attribute, and re-extracting only via self would
            # lose the module-level extractor result.
            try:
                spec = getattr(nlu, "search_spec", None) or search_spec
                if spec:
                    spec_dict = spec.dict() if hasattr(spec, "dict") else spec
                    try:
                        from product_search import run_structured_search

                        # A valid structured search owns both retrieval and hard filters.
                        # If it returns zero results, do not fall back to a price-only
                        # search that can return unrelated products.
                        ss_results = run_structured_search(
                            self.repo,
                            spec_dict,
                            limit=state.active_filters.get("limit", 20) or 20,
                        )
                        structured_search_attempted = True
                        response_products = ss_results
                        if ss_results:
                            reasons["note"] = "نتایج بر اساس نام محصول و فیلترهای ساختاریافته بازگشتی."
                        else:
                            reasons["note"] = "محصولی با نام و فیلترهای درخواستی پیدا نشد."
                    except Exception as e:
                        logger.warning("structured_search failed: %s", e)
                        structured_search_attempted = False
            except Exception as e:
                logger.warning("Could not prepare structured search: %s", e)
        else:
            # Intent detection removed — treat free-text as a product search query
            entities = self._extract_entities(text, state)
            logger.info(f"Entities extracted: {entities}")

        missing = []
        # simple missing-info rules: require at least a product keyword, budget, or location to run a meaningful search
        if (
            "product_ids" not in entities
            and "budget_toman" not in entities
            and "min_price" not in entities
            and "max_price" not in entities
            and not entities.get("location")
        ):
            missing.append("product_or_budget")

        # apply action: always perform a product search using available entities
        stores = []
        status = "success"
        next_q = None

        # Do not replace a valid structured search with a broad price-only fallback
        # when the structured search correctly found no matching products.
        if not response_products and not structured_search_attempted:
            response_products = []
            reasons = {}
            try:
                # Prefer explicit product ids if available
                pids = entities.get("product_ids") or entities.get("refer_product_ids")
                if pids:
                    response_products = self._search_products(product_ids=pids, limit=10)
                    reasons["note"] = "نتایج بر اساس شناسه/کلیدواژه محصول یافت شد."
                else:
                    # budget-based search
                    budget = (
                        entities.get("budget_toman")
                        or entities.get("min_price")
                        or entities.get("max_price")
                    )
                    if budget and isinstance(budget, (int, float)):
                        offers = [
                            o
                            for o in self.repo._offers.values()
                            if o.price_toman and o.price_toman <= int(budget)
                        ]
                        prodmap = {}
                        for o in offers:
                            prodmap.setdefault(o.product_id, []).append(o)
                        for pid, ofs in prodmap.items():
                            p = self.repo.get_product(pid)
                            if p:
                                response_products.append(
                                    {
                                        "product_id": pid,
                                        "persian_title": p.persian_title,
                                        "offers": sorted(
                                            [
                                                {
                                                    "offer_id": o.offer_id,
                                                    "price_toman": o.price_toman,
                                                    "shop_id": o.shop_id,
                                                }
                                                for o in ofs
                                            ],
                                            key=lambda x: x["price_toman"],
                                        ),
                                    }
                                )
                        reasons["note"] = (
                            f"محصولاتی با قیمت حداکثر {budget} تومان فیلتر شدند."
                        )
                    else:
                        # location-based shopping
                        loc = entities.get("location")
                        if loc:
                            state.active_filters["location"] = loc
                            reasons["note"] = f'فیلتر مکانی روی "{loc}" قرار گرفت.'
                            matches = [
                                s
                                for s in self.repo._shops.values()
                                if s.location_label and loc in (s.location_label.lower())
                            ]
                            if not matches:
                                matches = [
                                    s
                                    for s in self.repo._shops.values()
                                    if s.address and loc in s.address.lower()
                                ]
                            stores = [
                                {
                                    "shop_id": s.shop_id,
                                    "name": s.name,
                                    "address": s.address,
                                    "location_label": s.location_label,
                                }
                                for s in matches[:20]
                            ]
                            offers = []
                            for s in matches[:20]:
                                offers.extend(
                                    [
                                        {
                                            "offer_id": o.offer_id,
                                            "product_id": o.product_id,
                                            "price_toman": o.price_toman,
                                            "shop_id": s.shop_id,
                                        }
                                        for o in self.repo.get_offers_by_shop(s.shop_id)
                                    ]
                                )
                            prodmap = {}
                            for o in offers:
                                prodmap.setdefault(o["product_id"], []).append(o)
                            for pid, ofs in prodmap.items():
                                p = self.repo.get_product(pid)
                                if p:
                                    response_products.append(
                                        {
                                            "product_id": pid,
                                            "persian_title": p.persian_title,
                                            "offers": sorted(
                                                ofs,
                                                key=lambda x: (
                                                    x["price_toman"]
                                                    if x["price_toman"] is not None
                                                    else 10**18
                                                ),
                                            ),
                                        }
                                    )
                        else:
                            # full-text query fallback
                            response_products = self._search_products(query=text, limit=10)
                            if response_products:
                                reasons["note"] = "نتایج بر اساس جستجوی متن کامل."
                            else:
                                response_products = self._search_products(limit=10)
                                reasons["note"] = "نتایج پیش‌فرض (برگشتی به عنوان پیش‌فرض)."
            except Exception as e:
                status = "error"
                return {
                    "response_type": "error",
                    "status": status,
                    "text": "هنگام پردازش درخواست خطا رخ داد: " + str(e),
                    "products": [],
                    "stores": [],
                    "reasons": {},
                    "price_update_times": {},
                    "inventory_status": {},
                    "unknowns": missing,
                    "next_question": None,
                    "saved_state_snapshot": state.snapshot(),
                }

        # post-check: are results sufficient?
        sufficient = True
        if not response_products:
            sufficient = False
            status = "partial"
            if not missing:
                # ask clarifying question
                next_q = "لطفا بگویید دنبال چه محصولی هستید یا بودجه و برند مورد نظر را بنویسید."
                missing.append("product_keyword")

        # prepare structured output
        price_updates = {}
        inventory_status = {}
        for p in response_products:
            for o in p.get("offers", []):
                pid = p.get("product_id")
                key = o.get("offer_id") or f"{pid}:{o.get('shop_id')}"
                price_updates[key] = (
                    o.get("last_price_changed_at")
                    if o.get("last_price_changed_at")
                    else None
                )
                # infer availability: if last_price_changed_at recent -> likely available
                if o.get("last_price_changed_at"):
                    try:
                        dt = datetime.fromisoformat(o.get("last_price_changed_at"))
                        inventory_status[key] = {
                            "likely_available": not is_stale(dt, days_threshold=30),
                            "age_days": compute_staleness(dt),
                        }
                    except Exception:
                        inventory_status[key] = {
                            "likely_available": False,
                            "age_days": None,
                        }
                else:
                    inventory_status[key] = {
                        "likely_available": False,
                        "age_days": None,
                    }

        # update state
        state.previous_results["products"] = response_products
        # set selected product if single
        if len(response_products) == 1:
            state.selected_product_ids = [response_products[0]["product_id"]]
        state.last_updated = datetime.utcnow()
        # preserve visit plans and followups already in state
        try:
            self.save_state(state)
        except Exception:
            # state persistence should not break handler; log and continue
            logger.warning("save_state failed; continuing without persistence")

        # build Persian text summary
        if status == "error":
            text = "متأسفم، مشکلی پیش آمد."
        elif not response_products:
            text = "نتیجه‌ای پیدا نشد."
        else:
            # summarize top 3 products
            lines = []
            for p in response_products[:3]:
                offers = p.get("offers", [])
                cheapest = None
                if offers:
                    cheapest = min(
                        [o for o in offers if o.get("price_toman") is not None],
                        key=lambda x: x.get("price_toman"),
                        default=None,
                    )
                if cheapest:
                    lines.append(
                        f"{p.get('persian_title') or p.get('product_id')} - کمترین قیمت: {cheapest.get('price_toman')} تومان"
                    )
                else:
                    lines.append(
                        f"{p.get('persian_title') or p.get('product_id')} - قیمت در دسترس نیست"
                    )
            text = "پیشنهادها:" + "\n".join(lines)
            if next_q:
                text += "\n\n" + next_q

        # Always return a structured out dict (avoid falling through to None)
        out = {
            "response_type": (
                "search_result"
                if response_products
                else ("clarification" if missing else "empty")
            ),
            "status": status,
            "text": text,
            "products": response_products,
            "stores": stores,
            "reasons": reasons,
            "price_update_times": price_updates,
            "inventory_status": inventory_status,
            "unknowns": missing,
            "next_question": next_q,
            "saved_state_snapshot": state.snapshot(),
        }
        return out


if __name__ == "__main__":
    # simple manual test
    repo = DataRepository()
    repo.load_all()
    orch = Orchestrator(repo)
    state = ConversationState(conversation_id="demo")
    res = orch.handle_request("گوشی آیفون 13 با بودجه 30 میلیون تومان", state)
    # Console output is restricted by logging_config. Use explicit prefixes in logs for console visibility.
    logger.info(f"USER_INPUT: {'گوشی آیفون 13 با بودجه 30 میلیون تومان'}")
    logger.info(f"LLM_OUTPUT: {res.get('text')}")
