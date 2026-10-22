from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from orchestrator import Orchestrator
from repository import DataRepository
from conversation_state import ConversationState

app = FastAPI(title="Shopping Assistant API")

# allow local testing from browsers/tools
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# instantiate repository and orchestrator (lazy-loading inside orchestrator)
repo = DataRepository()
orch = Orchestrator(repo=repo)

# --- API models ---
class QueryRequest(BaseModel):
    text: str
    conversation_id: Optional[str] = None

class QueryResponse(BaseModel):
    response_type: str
    status: str
    text: str
    products: List[Dict[str, Any]]
    stores: List[Dict[str, Any]]
    reasons: Dict[str, Any]
    price_update_times: Dict[str, Any]
    inventory_status: Dict[str, Any]
    unknowns: List[str]
    next_question: Optional[str]
    saved_state_snapshot: Dict[str, Any]

# --- Endpoints ---
@app.on_event("startup")
def startup_event():
    # do not load data until first request but ensure files exist by trying to access the data dir
    try:
        # repository will lazily load when orchestrator._ensure_loaded() is called
        pass
    except Exception:
        pass


@app.post("/v1/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    """Run a conversational query (text). Provide an optional conversation_id to continue a session.

    The API returns the orchestrator structured response and saves the conversation state on disk.
    """
    # load state if provided
    state = None
    if req.conversation_id:
        try:
            state = orch.load_state(req.conversation_id)
        except Exception:
            # if loading fails, continue with fresh state
            state = None
    # call orchestrator
    try:
        out = orch.handle_request(req.text, state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return out


@app.get("/v1/product/{product_id}")
def get_product(product_id: str):
    """Return a product and its offers."""
    orch._ensure_loaded()
    p = repo.get_product(product_id)
    if not p:
        raise HTTPException(status_code=404, detail="product not found")
    offers = repo.get_offers_by_product(product_id)
    return {
        "product": p.dict() if hasattr(p, 'dict') else p.__dict__,
        "offers": [o.dict() if hasattr(o, 'dict') else o.__dict__ for o in offers]
    }


@app.get("/v1/shop/{shop_id}")
def get_shop(shop_id: str):
    orch._ensure_loaded()
    s = repo.get_shop(shop_id)
    if not s:
        raise HTTPException(status_code=404, detail="shop not found")
    offers = repo.get_offers_by_shop(shop_id)
    return {"shop": s.dict() if hasattr(s, 'dict') else s.__dict__, "offers": [o.dict() if hasattr(o, 'dict') else o.__dict__ for o in offers]}


@app.get("/v1/offers")
def list_offers(product_id: Optional[str] = None, shop_id: Optional[str] = None, limit: Optional[int] = 100):
    orch._ensure_loaded()
    offers = list(repo._offers.values())
    if product_id:
        offers = [o for o in offers if o.product_id == product_id]
    if shop_id:
        offers = [o for o in offers if o.shop_id == shop_id]
    return {"count": len(offers[:limit]), "offers": [o.dict() if hasattr(o, 'dict') else o.__dict__ for o in offers[:limit]]}


@app.get("/v1/state/{conversation_id}")
def load_state(conversation_id: str):
    s = orch.load_state(conversation_id)
    if not s:
        raise HTTPException(status_code=404, detail="state not found")
    return {"state": s.snapshot()}


@app.post("/v1/state")
def save_state(state_snapshot: Dict[str, Any]):
    # accept a conversation state snapshot (as produced by saved_state_snapshot)
    try:
        state = ConversationState(**state_snapshot)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid state payload: {e}")
    orch.save_state(state)
    return {"saved": True, "conversation_id": state.conversation_id}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
