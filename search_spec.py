from pydantic import BaseModel, Field, validator
from typing import Optional, List, Any


class FeatureSpec(BaseModel):
    name: str
    operator: str = 'eq'
    value: Any


class ProductSpec(BaseModel):
    name: Optional[str] = None
    brand: Optional[str] = None
    model: Optional[str] = None
    category: Optional[str] = None
    features: List[FeatureSpec] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    exclude_keywords: List[str] = Field(default_factory=list)


class PriceSpec(BaseModel):
    min: Optional[int] = None
    max: Optional[int] = None
    currency: str = 'IRR'
    sort: Optional[str] = 'price_asc'


class LocationSpec(BaseModel):
    city: Optional[str] = None
    area: Optional[str] = None
    address: Optional[str] = None  # free-text place name e.g. تجریش, ونک
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    radius_km: Optional[float] = None
    max_distance_km: Optional[float] = None


class StoreFiltersSpec(BaseModel):
    store_name: Optional[str] = None
    business_category: Optional[str] = None
    mall_name: Optional[str] = None
    floor: Optional[str] = None
    unit: Optional[str] = None
    open_at: Optional[str] = None
    open_now: Optional[bool] = None


class PurchaseConditionsSpec(BaseModel):
    warranty_required: Optional[bool] = None
    warranty_provider: Optional[str] = None
    installment_payment: Optional[bool] = None
    torob_pay: Optional[bool] = None
    instant_delivery: Optional[bool] = None
    commodity_coupon: Optional[bool] = None


class OfferFiltersSpec(BaseModel):
    availability: Optional[str] = 'any'
    max_price_age_days: Optional[int] = None
    require_recent_price: Optional[bool] = False


class SearchOptionsSpec(BaseModel):
    sort_by: Optional[str] = 'relevance'
    limit: int = 10
    include_alternatives: bool = True
    semantic_search: bool = True


class ConversationSpec(BaseModel):
    session_id: Optional[str] = None
    # previous_intent removed — conversation should not track intent
    referenced_product_id: Optional[str] = None
    referenced_store_id: Optional[str] = None
    preserve_previous_filters: bool = True


class ClarificationSpec(BaseModel):
    required: bool = False
    missing_fields: List[str] = Field(default_factory=list)
    question: Optional[str] = None


class SearchSpec(BaseModel):
    # intent removed — SearchSpec now assumes product search
    user_goal: Optional[str] = None
    product: ProductSpec = Field(default_factory=ProductSpec)
    price: PriceSpec = Field(default_factory=PriceSpec)
    location: LocationSpec = Field(default_factory=LocationSpec)
    store_filters: StoreFiltersSpec = Field(default_factory=StoreFiltersSpec)
    purchase_conditions: PurchaseConditionsSpec = Field(default_factory=PurchaseConditionsSpec)
    offer_filters: OfferFiltersSpec = Field(default_factory=OfferFiltersSpec)
    search_options: SearchOptionsSpec = Field(default_factory=SearchOptionsSpec)
    conversation: ConversationSpec = Field(default_factory=ConversationSpec)
    clarification: ClarificationSpec = Field(default_factory=ClarificationSpec)

    class Config:
        extra = 'forbid'


__all__ = ['SearchSpec', 'ProductSpec', 'PriceSpec', 'LocationSpec']
