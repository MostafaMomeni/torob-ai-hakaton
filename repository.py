import csv
from pathlib import Path
from typing import Dict, List, Optional
from schemas import ShopRaw, ProductRaw, OfferRaw, Shop, Product, Offer
import json
from datetime import datetime, timedelta

DATA_DIR = Path(__file__).parent / "data"
# Also look next to the package (project root) so CSVs dropped alongside
# the Python modules are found without a mandatory data/ subfolder.
_ROOT_DIR = Path(__file__).parent


def _resolve_csv(name: str, data_dir: Path) -> Path:
    """Prefer data_dir/name, then project-root/name."""
    primary = data_dir / name
    if primary.exists():
        return primary
    fallback = _ROOT_DIR / name
    if fallback.exists():
        return fallback
    return primary  # default path (may not exist yet; loaders handle missing files)


class DataRepository:
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or DATA_DIR
        self.shops_csv = _resolve_csv("shops.csv", self.data_dir)
        self.products_csv = _resolve_csv("products.csv", self.data_dir)
        self.offers_csv = _resolve_csv("offers.csv", self.data_dir)
        self._shops: Dict[str, Shop] = {}
        self._products: Dict[str, Product] = {}
        self._offers: Dict[str, Offer] = {}
        self._offers_by_product: Dict[str, List[Offer]] = {}
        self._offers_by_shop: Dict[str, List[Offer]] = {}
        # Full shops.csv rows, including columns not represented by the Shop schema.
        self._shop_details: Dict[str, Dict[str, object]] = {}

    def load_all(self):
        self._shops.clear()
        self._products.clear()
        self._offers.clear()
        self._offers_by_product.clear()
        self._offers_by_shop.clear()
        self._shop_details.clear()
        self._load_shops()
        self._load_products()
        self._load_offers()

    @staticmethod
    def _parse_shop_cell(key, value):
        if value is None:
            return None
        value = value.strip() if isinstance(value, str) else value
        if value == "":
            return None
        if key in ("lat", "lon"):
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
        if key == "supports_fast_delivery":
            if isinstance(value, bool):
                return value
            return str(value).strip().lower() in ("1", "true", "yes", "y", "بله")
        if key in ("category_names", "contacts", "payment_options", "weekly_hours"):
            try:
                return json.loads(value) if isinstance(value, str) else value
            except (ValueError, TypeError):
                return value
        return value

    def _load_shops(self):
        if not self.shops_csv.exists():
            import logging
            logging.getLogger("shopping_assistant.repository").warning(
                "shops.csv not found at %s", self.shops_csv
            )
            return
        with self.shops_csv.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                details = {key: self._parse_shop_cell(key, value) for key, value in r.items()}
                shop_id = str(details.get("shop_id") or "").strip()
                if shop_id:
                    self._shop_details[shop_id] = details
                try:
                    # Pass known ShopRaw fields; extras live in _shop_details.
                    raw_fields = (
                        "shop_id", "name", "address", "lat", "lon",
                        "location_label", "category_names", "contacts",
                        "instagram", "payment_options", "supports_fast_delivery",
                        "website_url", "weekly_hours",
                    )
                    raw = ShopRaw(**{key: r.get(key) for key in raw_fields})
                except Exception as e:
                    # skip invalid row but log
                    # use logging instead of printing to avoid noisy console output
                    import logging
                    logging.getLogger('shopping_assistant.repository').warning('Invalid shop row: %s', e)
                    continue
                # parse fields
                cat = None
                try:
                    if raw.category_names:
                        cat = json.loads(raw.category_names)
                except Exception:
                    cat = None
                contacts = None
                try:
                    if raw.contacts:
                        contacts = json.loads(raw.contacts)
                except Exception:
                    contacts = None
                hours = None
                try:
                    if raw.weekly_hours:
                        hours = json.loads(raw.weekly_hours)
                except Exception:
                    hours = None
                shop = Shop(
                    shop_id=raw.shop_id,
                    name=raw.name,
                    address=raw.address,
                    lat=raw.lat,
                    lon=raw.lon,
                    location_label=raw.location_label,
                    category_names=cat,
                    contacts=contacts,
                    website_url=raw.website_url,
                    weekly_hours=hours,
                    data_observed_at=None
                )
                self._shops[shop.shop_id] = shop

    def _load_products(self):
        if not self.products_csv.exists():
            import logging
            logging.getLogger("shopping_assistant.repository").warning(
                "products.csv not found at %s", self.products_csv
            )
            return
        with self.products_csv.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                try:
                    raw = ProductRaw(**r)
                except Exception as e:
                    import logging
                    logging.getLogger('shopping_assistant.repository').warning('Invalid product row: %s', e)
                    continue
                imgs = None
                specs = None
                try:
                    if raw.image_urls:
                        imgs = json.loads(raw.image_urls)
                except Exception:
                    imgs = None
                try:
                    if raw.specs:
                        specs = json.loads(raw.specs)
                except Exception:
                    specs = None
                p = Product(
                    product_id=raw.product_id,
                    persian_title=raw.persian_title,
                    brand_name=raw.brand_name,
                    category_name=raw.category_name,
                    model=raw.model,
                    image_urls=imgs,
                    specs=specs,
                    data_observed_at=None
                )
                self._products[p.product_id] = p

    def _load_offers(self):
        if not self.offers_csv.exists():
            import logging
            logging.getLogger("shopping_assistant.repository").warning(
                "offers.csv not found at %s", self.offers_csv
            )
            return
        with self.offers_csv.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                try:
                    raw = OfferRaw(**r)
                except Exception as e:
                    import logging
                    logging.getLogger('shopping_assistant.repository').warning('Invalid offer row: %s', e)
                    continue
                offer = Offer(
                    offer_id=raw.offer_id,
                    product_id=raw.product_id,
                    shop_id=raw.shop_id,
                    price_toman=raw.price_toman,
                    last_price_changed_at=raw.last_price_changed_at,
                    seller_title=raw.seller_title,
                    warranty=raw.warranty,
                    data_observed_at=raw.last_price_changed_at
                )
                self._offers[offer.offer_id] = offer
                self._offers_by_product.setdefault(offer.product_id, []).append(offer)
                if offer.shop_id:
                    self._offers_by_shop.setdefault(offer.shop_id, []).append(offer)

    # Query APIs
    def get_shop(self, shop_id: str) -> Optional[Shop]:
        return self._shops.get(str(shop_id)) if shop_id is not None else None

    def get_shop_details(self, shop_id: str) -> Optional[Dict[str, object]]:
        """Return normalized full shops.csv row, including extra metadata columns."""
        if shop_id is None:
            return None
        details = self._shop_details.get(str(shop_id))
        if details is not None:
            return dict(details)
        # Graceful fallback for repos/tests that populate Shop objects directly.
        shop = self.get_shop(shop_id)
        if shop is None:
            return None
        fields = ("shop_id", "name", "address", "category_names", "contacts",
                  "instagram", "lat", "location_label", "lon", "payment_options",
                  "supports_fast_delivery", "website_url", "weekly_hours")
        return {key: getattr(shop, key, None) for key in fields}

    def all_shops(self):
        return list(self._shops.values())

    def get_product(self, product_id: str) -> Optional[Product]:
        return self._products.get(product_id)

    def get_offers_by_product(self, product_id: str) -> List[Offer]:
        return list(self._offers_by_product.get(product_id, []))

    def get_offers_by_shop(self, shop_id: str) -> List[Offer]:
        return list(self._offers_by_shop.get(shop_id, []))

    def validate_references(self) -> Dict[str, List[str]]:
        """Check for offers pointing to missing products or shops"""
        missing_products = []
        missing_shops = []
        for o in self._offers.values():
            if o.product_id not in self._products:
                missing_products.append(o.offer_id)
            if o.shop_id and o.shop_id not in self._shops:
                missing_shops.append(o.offer_id)
        return {"missing_products": missing_products, "missing_shops": missing_shops}

    def find_duplicate_shops(self):
        # naive duplicate detection by address+name
        seen = {}
        dups = []
        for s in self._shops.values():
            key = (s.name, s.address)
            if key in seen:
                dups.append((seen[key], s.shop_id))
            else:
                seen[key] = s.shop_id
        return dups

    def shops_with_valid_coords(self):
        res = []
        for s in self._shops.values():
            if s.lat is not None and s.lon is not None:
                try:
                    lat = float(s.lat)
                    lon = float(s.lon)
                    if -90 <= lat <= 90 and -180 <= lon <= 180:
                        res.append(s)
                except Exception:
                    continue
        return res

    def offers_with_recent_price(self, days=14):
        cutoff = datetime.utcnow() - timedelta(days=days)
        res = []
        for o in self._offers.values():
            if o.last_price_changed_at and o.last_price_changed_at >= cutoff:
                res.append(o)
        return res


if __name__ == '__main__':
    repo = DataRepository()
    repo.load_all()
    import logging
    log = logging.getLogger('shopping_assistant.repository')
    log.info('USER_INPUT: repo_stats'); log.info(f"LLM_OUTPUT: Shops: {len(repo._shops)} Products: {len(repo._products)} Offers: {len(repo._offers)}")
    log.debug('Reference issues: %s', repo.validate_references())
    log.debug('Valid coords shops: %s', len(repo.shops_with_valid_coords()))
