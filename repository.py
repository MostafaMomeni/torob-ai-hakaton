import csv
from pathlib import Path
from typing import Dict, List, Optional
from schemas import ShopRaw, ProductRaw, OfferRaw, Shop, Product, Offer
import json
from datetime import datetime

DATA_DIR = Path(__file__).parent / "data"


class DataRepository:
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or DATA_DIR
        self.shops_csv = self.data_dir / "shops.csv"
        self.products_csv = self.data_dir / "products.csv"
        self.offers_csv = self.data_dir / "offers.csv"
        self._shops: Dict[str, Shop] = {}
        self._products: Dict[str, Product] = {}
        self._offers: Dict[str, Offer] = {}

    def load_all(self):
        self._load_shops()
        self._load_products()
        self._load_offers()

    def _load_shops(self):
        with self.shops_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                try:
                    raw = ShopRaw(**r)
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
        with self.products_csv.open("r", encoding="utf-8") as f:
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
        with self.offers_csv.open("r", encoding="utf-8") as f:
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

    # Query APIs
    def get_shop(self, shop_id: str) -> Optional[Shop]:
        return self._shops.get(shop_id)

    def get_product(self, product_id: str) -> Optional[Product]:
        return self._products.get(product_id)

    def get_offers_by_product(self, product_id: str) -> List[Offer]:
        return [o for o in self._offers.values() if o.product_id == product_id]

    def get_offers_by_shop(self, shop_id: str) -> List[Offer]:
        return [o for o in self._offers.values() if o.shop_id == shop_id]

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
