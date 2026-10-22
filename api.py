from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

from orchestrator import Orchestrator
from repository import DataRepository
from conversation_state import ConversationState

app = FastAPI(title="Shopping Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

repo = DataRepository()
orch = Orchestrator(repo=repo)


class QueryRequest(BaseModel):
    text: str = ""
    conversation_id: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    page: Optional[int] = Field(default=1, ge=1)
    page_size: Optional[int] = Field(default=10, ge=1, le=50)
    result_id: Optional[str] = None


class PaginationMeta(BaseModel):
    page: int = 1
    page_size: int = 10
    total: int = 0
    total_pages: int = 0
    has_next: bool = False
    has_prev: bool = False


class QueryResponse(BaseModel):
    """Minimal client-facing search response."""
    status: str  # success | partial | error
    products: List[Dict[str, Any]] = []
    pagination: PaginationMeta = Field(default_factory=PaginationMeta)
    result_id: Optional[str] = None
    expires_at: Optional[str] = None


@app.on_event("startup")
def startup_event():
    pass


@app.post("/v1/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    state = None
    if req.conversation_id:
        try:
            state = orch.load_state(req.conversation_id)
        except Exception:
            state = None
    try:
        out = orch.handle_request(
            req.text or "",
            state,
            lat=req.lat,
            lon=req.lon,
            page=req.page or 1,
            page_size=req.page_size or 10,
            result_id=req.result_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return out


@app.get("/v1/results/{result_id}", response_model=QueryResponse)
def get_paginated_results(
    result_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=50),
):
    try:
        out = orch.paginate_results(result_id=result_id, page=page, page_size=page_size)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    if out is None:
        raise HTTPException(
            status_code=404,
            detail="result_id not found or expired; please run a new search",
        )
    return out


@app.get("/v1/product/{product_id}")
def get_product(product_id: str):
    orch._ensure_loaded()
    p = repo.get_product(product_id)
    if not p:
        raise HTTPException(status_code=404, detail="product not found")
    offers = repo.get_offers_by_product(product_id)
    return {
        "product": p.dict() if hasattr(p, "dict") else p.__dict__,
        "offers": [o.dict() if hasattr(o, "dict") else o.__dict__ for o in offers],
    }


@app.get("/v1/shop/{shop_id}")
def get_shop(shop_id: str):
    orch._ensure_loaded()
    s = repo.get_shop(shop_id)
    if not s:
        raise HTTPException(status_code=404, detail="shop not found")
    offers = repo.get_offers_by_shop(shop_id)
    return {
        "shop": s.dict() if hasattr(s, "dict") else s.__dict__,
        "offers": [o.dict() if hasattr(o, "dict") else o.__dict__ for o in offers],
    }


@app.get("/v1/offers")
def list_offers(
    product_id: Optional[str] = None,
    shop_id: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    limit: Optional[int] = None,
):
    orch._ensure_loaded()
    offers = list(repo._offers.values())
    if product_id:
        offers = [o for o in offers if o.product_id == product_id]
    if shop_id:
        offers = [o for o in offers if o.shop_id == shop_id]
    total = len(offers)
    effective_size = page_size
    if limit is not None and page_size == 20:
        effective_size = max(1, min(int(limit), 100))
    offset = (page - 1) * effective_size
    page_items = offers[offset: offset + effective_size]
    total_pages = (total + effective_size - 1) // effective_size if effective_size else 0
    return {
        "status": "success",
        "products": [],
        "pagination": {
            "page": page,
            "page_size": effective_size,
            "total": total,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_prev": page > 1,
        },
        "offers": [o.dict() if hasattr(o, "dict") else o.__dict__ for o in page_items],
        "result_id": None,
        "expires_at": None,
    }


@app.get("/v1/state/{conversation_id}")
def load_state(conversation_id: str):
    s = orch.load_state(conversation_id)
    if not s:
        raise HTTPException(status_code=404, detail="state not found")
    return {"state": s.snapshot()}


@app.post("/v1/state")
def save_state(state_snapshot: Dict[str, Any]):
    try:
        state = ConversationState(**state_snapshot)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid state payload: {e}")
    orch.save_state(state)
    return {"saved": True, "conversation_id": state.conversation_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
