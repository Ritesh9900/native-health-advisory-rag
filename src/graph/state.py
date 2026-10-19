from typing import TypedDict, List, Dict, Any, Optional

class HealthAdvisoryState(TypedDict):
    """
    State definition for the Native Language Health Advisory LangGraph pipeline.
    All state is ephemeral, scrubbed of PII, and bounded to the session.
    """
    raw_query: str
    scrubbed_query: str
    redacted_pii: List[Dict[str, Any]]
    
    detected_language: str
    language_confidence: float
    
    is_emergency: bool
    emergency_info: Optional[Dict[str, Any]]
    
    intent: str  # "health_inquiry", "symptom_guidance", "prescription_diagnosis_refusal", "out_of_scope"
    refusal_reason: Optional[str]
    
    rewritten_query: str
    retrieved_docs: List[Dict[str, Any]]
    
    generated_text: str
    citations: List[Dict[str, str]]
    disclaimer: str
    final_response: str
