from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class ConversationState(BaseModel):
    conversation_id: str
    user_id: Optional[str] = "default"
    original_request: Optional[str] = None
    # Active filters that are currently applied (persist across turns unless explicitly removed)
    active_filters: Dict[str, Any] = Field(default_factory=dict)
    # Confirmed filters (those the user explicitly confirmed)
    confirmed_filters: Dict[str, Any] = Field(default_factory=dict)
    # History of filter changes: list of {ts, key, old, new, action}
    filter_changes: List[Dict[str, Any]] = Field(default_factory=list)
    selected_product_ids: List[str] = Field(default_factory=list)
    previous_results: Dict[str, Any] = Field(default_factory=dict)
    # confirmed_info is generic key->value of confirmed extracted info
    confirmed_info: Dict[str, Any] = Field(default_factory=dict)
    # missing information the system still needs to ask for
    missing_information: List[str] = Field(default_factory=list)
    visit_plans: List[Dict[str, Any]] = Field(default_factory=list)
    followups: List[Dict[str, Any]] = Field(default_factory=list)
    last_updated: Optional[datetime] = None
    # conversation messages (keeps recent user/assistant turns for local context; not sent fully to LLM)
    messages: List[Dict[str, Any]] = Field(default_factory=list)

    def snapshot(self):
        return self.dict()

    def push_message(self, role: str, text: str, ts: Optional[datetime] = None):
        """Append a message to the conversation history.

        role: 'user' or 'assistant'
        We keep messages bounded locally; this is not sent in full to the LLM.
        """
        ts = ts or datetime.utcnow()
        self.messages.append({'role': role, 'text': text, 'ts': ts.isoformat()})
        # keep only the last N messages to avoid unbounded growth
        if len(self.messages) > 20:
            self.messages = self.messages[-20:]

    def summary_for_nlu(self, max_prev: int = 5) -> Dict[str, Any]:
        """Return a concise summary safe to send to the LLM NLU component.

        This includes only high-level structured fields (no full history or PII):
        - conversation_id, user_id
        - active_filters (confirmed subset)
        - confirmed_filters
        - selected_product_ids
        - a short previous_results_summary (product_id + title) up to max_prev
        - recent user messages (texts only, up to 3) to help disambiguation
        - missing_information
        """
        prev = []
        products = self.previous_results.get('products') or []
        for p in products[:max_prev]:
            prev.append({'product_id': p.get('product_id'), 'persian_title': p.get('persian_title')})
        recent_user_texts = [m['text'] for m in self.messages if m.get('role') == 'user'][-3:]
        return {
            'conversation_id': self.conversation_id,
            'user_id': self.user_id,
            'active_filters': self.active_filters,
            'confirmed_filters': self.confirmed_filters,
            'selected_product_ids': self.selected_product_ids,
            'previous_results_summary': prev,
            'recent_user_messages': recent_user_texts,
            'missing_information': self.missing_information,
        }

