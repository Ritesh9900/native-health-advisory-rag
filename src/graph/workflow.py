from langgraph.graph import StateGraph, END
from src.graph.state import HealthAdvisoryState
from src.graph.nodes import (
    pii_scrub_node,
    language_detect_node,
    emergency_triage_node,
    intent_router_node,
    retrieval_node,
    generator_node,
    safety_postprocessor_node
)

def route_after_emergency(state: HealthAdvisoryState) -> str:
    """Conditional router: If life-threatening red-flag detected, skip retrieval & LLM entirely."""
    if state.get("is_emergency", False):
        return "safety_postprocessor"
    return "intent_router"

def route_after_intent(state: HealthAdvisoryState) -> str:
    """Conditional router: If query requests prescriptions/diagnosis, refuse safely."""
    if state.get("intent") == "prescription_diagnosis_refusal":
        return "safety_postprocessor"
    return "retrieval"

def build_health_advisory_graph():
    """Constructs and compiles the safety-critical LangGraph state machine."""
    workflow = StateGraph(HealthAdvisoryState)

    # 1. Register Nodes
    workflow.add_node("pii_scrub", pii_scrub_node)
    workflow.add_node("language_detect", language_detect_node)
    workflow.add_node("emergency_triage", emergency_triage_node)
    workflow.add_node("intent_router", intent_router_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("generator", generator_node)
    workflow.add_node("safety_postprocessor", safety_postprocessor_node)

    # 2. Define Edges
    workflow.set_entry_point("pii_scrub")
    workflow.add_edge("pii_scrub", "language_detect")
    workflow.add_edge("language_detect", "emergency_triage")

    # Conditional Branch 1: Emergency Check
    workflow.add_conditional_edges(
        "emergency_triage",
        route_after_emergency,
        {
            "safety_postprocessor": "safety_postprocessor",
            "intent_router": "intent_router"
        }
    )

    # Conditional Branch 2: Scope & Prescription Check
    workflow.add_conditional_edges(
        "intent_router",
        route_after_intent,
        {
            "safety_postprocessor": "safety_postprocessor",
            "retrieval": "retrieval"
        }
    )

    # Core RAG path
    workflow.add_edge("retrieval", "generator")
    workflow.add_edge("generator", "safety_postprocessor")
    workflow.add_edge("safety_postprocessor", END)

    app = workflow.compile()
    return app

# Singleton compiled app
health_advisory_app = build_health_advisory_graph()
