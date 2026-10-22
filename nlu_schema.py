from pydantic import BaseModel, Field
from typing import Optional, List


class LocationField(BaseModel):
    text: Optional[str] = None  # free-text place / address e.g. تجریش
    address: Optional[str] = None  # alias for text when LLM returns "address"
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    radius_km: Optional[float] = None


class EntitiesField(BaseModel):
    product_name: Optional[str] = None
    brand: Optional[str] = None
    category: Optional[str] = None
    model: Optional[str] = None
    features: List[str] = Field(default_factory=list)
    location: LocationField = Field(default_factory=LocationField)


class FiltersField(BaseModel):
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    currency: Optional[str] = None
    brands: List[str] = Field(default_factory=list)
    categories: List[str] = Field(default_factory=list)
    features: List[str] = Field(default_factory=list)
    sort_by: Optional[str] = 'relevance'


class ContextReference(BaseModel):
    previous_result_ids: List[str] = Field(default_factory=list)
    selected_product_id: Optional[str] = None
    selected_store_id: Optional[str] = None


class NLUResult(BaseModel):
    schema_version: str = '1.0'
    # NOTE: intent field removed — all inputs are treated as product search
    language: str = 'fa'
    user_goal: Optional[str] = None
    entities: EntitiesField = Field(default_factory=EntitiesField)
    filters: FiltersField = Field(default_factory=FiltersField)
    context_reference: ContextReference = Field(default_factory=ContextReference)
    missing_information: List[str] = Field(default_factory=list)
    clarification_question: Optional[str] = None
    confidence: float = 0.0
    raw_text: Optional[str] = None

    class Config:
        extra = 'forbid'
