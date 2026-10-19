import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# Ensure UTF-8 output on Windows
if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from src.graph.workflow import health_advisory_app
from src.config import Config

app = FastAPI(
    title="Native Language Health Advisory API",
    description="Production-grade, Zero-PHI Healthcare RAG with deterministic triage and multilingual grounding.",
    version="1.0.0"
)

# CORS Middleware (Enable for frontend integrations)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AdvisoryRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=2000, description="The user's health inquiry in any supported language")

class CitationItem(BaseModel):
    id: str
    title: str
    section: str
    source: str

class AdvisoryResponse(BaseModel):
    scrubbed_query: str
    detected_language: str
    is_emergency: bool
    emergency_info: Optional[Dict[str, Any]] = None
    intent: str
    citations: List[CitationItem] = []
    final_response: str
    disclaimer: str

@app.get("/healthz", status_code=status.HTTP_200_OK)
def health_check():
    """Liveness probe for cloud orchestrators (K8s, ECS, HF Spaces)."""
    return {"status": "healthy", "service": Config.PROJECT_NAME}

@app.post("/v1/advisory", response_model=AdvisoryResponse)
def get_health_advisory(request: AdvisoryRequest):
    """
    Main Advisory Endpoint:
    Executes the full LangGraph state machine with Zero-PHI sanitization,
    deterministic triage, and grounded generation.
    """
    try:
        initial_state = {
            "raw_query": request.query,
            "scrubbed_query": "",
            "redacted_pii": [],
            "detected_language": "en",
            "language_confidence": 0.0,
            "is_emergency": False,
            "emergency_info": None,
            "intent": "health_inquiry",
            "refusal_reason": None,
            "rewritten_query": "",
            "retrieved_docs": [],
            "generated_text": "",
            "citations": [],
            "disclaimer": "",
            "final_response": ""
        }

        result = health_advisory_app.invoke(initial_state)

        citations_formatted = [
            CitationItem(
                id=c.get("id", ""),
                title=c.get("title", ""),
                section=c.get("section", ""),
                source=c.get("source", "")
            )
            for c in result.get("citations", [])
        ]

        return AdvisoryResponse(
            scrubbed_query=result.get("scrubbed_query", ""),
            detected_language=result.get("detected_language", "en"),
            is_emergency=result.get("is_emergency", False),
            emergency_info=result.get("emergency_info"),
            intent=result.get("intent", "health_inquiry"),
            citations=citations_formatted,
            final_response=result.get("final_response", ""),
            disclaimer=result.get("disclaimer", "")
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline execution error: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
