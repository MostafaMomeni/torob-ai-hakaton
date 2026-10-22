from pydantic import BaseModel, Field, validator
from typing import List, Optional, Any, Dict
from datetime import datetime


def normalize_price(value):
    if value is None:
        return None

    if isinstance(value, str):
        value = value.strip().replace(",", "").replace("٬", "")
        value = value.translate(
            str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")
        )

        if value.lower() in ("null", "none", ""):
            return None

    try:
        return int(value)
    except (ValueError, TypeError):
        return None


class ShopRaw(BaseModel):
    address: Optional[str]
    category_names: Optional[str]
    contacts: Optional[str]
    instagram: Optional[str]
    lat: Optional[float]
    location_label: Optional[str]
    lon: Optional[float]
    name: Optional[str]
    payment_options: Optional[str]
    shop_id: str
    supports_fast_delivery: Optional[str]
    website_url: Optional[str]
    weekly_hours: Optional[str]


class ProductRaw(BaseModel):
    brand_name: Optional[str]
    category_name: Optional[str]
    image_urls: Optional[str]
    model: Optional[str]
    persian_title: Optional[str]
    product_id: str
    specs: Optional[str]


class OfferRaw(BaseModel):
    description: Optional[str] = None
    last_price_changed_at: Optional[datetime] = None
    offer_id: str
    price_toman: Optional[int] = None
    product_id: str
    seller_title: Optional[str] = None
    shop_id: Optional[str] = None
    warranty: Optional[str] = None

    @validator("price_toman", pre=True)
    def parse_price(cls, value):
        return normalize_price(value)

    @validator("last_price_changed_at", pre=True)
    def parse_dt(cls, value):
        if value is None or (
            isinstance(value, str)
            and value.strip().lower() in ("", "null", "none")
        ):
            return None

        if isinstance(value, datetime):
            return value

        if not isinstance(value, str):
            return None

        try:
            return datetime.fromisoformat(
                value.replace("Z", "+00:00")
            )
        except (ValueError, TypeError):
            return None


# Internal models

class Shop(BaseModel):
    shop_id: str
    name: Optional[str]
    address: Optional[str]
    lat: Optional[float]
    lon: Optional[float]
    location_label: Optional[str]
    category_names: Optional[List[str]]
    contacts: Optional[List[Dict[str, Any]]]
    website_url: Optional[str]
    weekly_hours: Optional[Dict[str, Any]]
    data_source: str = "starcoach_dataset"
    data_observed_at: Optional[datetime]


class Product(BaseModel):
    product_id: str
    persian_title: Optional[str]
    brand_name: Optional[str]
    category_name: Optional[str]
    model: Optional[str]
    image_urls: Optional[List[str]]
    specs: Optional[List[Dict[str, Any]]]
    data_source: str = "starcoach_dataset"
    data_observed_at: Optional[datetime]


class Offer(BaseModel):
    offer_id: str
    product_id: str
    shop_id: Optional[str]
    price_toman: Optional[int]
    last_price_changed_at: Optional[datetime]
    seller_title: Optional[str]
    warranty: Optional[str]
    data_source: str = "starcoach_dataset"
    data_observed_at: Optional[datetime]
