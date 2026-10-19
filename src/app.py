import sys
import os

import logging

# Ensure UTF-8 output on Windows
if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Silence harmless Streamlit file-watcher warnings on optional third-party modules
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.CRITICAL)

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.graph.workflow import health_advisory_app
from src.config import Config

def process_query(user_query: str):
    """Executes the full LangGraph safety pipeline for a user inquiry."""
    initial_state = {
        "raw_query": user_query,
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
    return result

def run_cli():
    print("=" * 70)
    print(" 🏥 NATIVE LANGUAGE HEALTH ADVISORY (Safety & Zero-PHI RAG) 🏥")
    print("=" * 70)
    print("Type a health inquiry in English, Hindi, Hinglish, or Spanish.")
    print("Type 'sample' to run standard clinical benchmark test cases.")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            query = input("\n[User Query] > ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit", "q"]:
                break
            if query.lower() == "sample":
                run_sample_benchmarks()
                continue

            print("\n🔄 Processing through LangGraph State Machine...")
            res = process_query(query)

            print("\n" + "-" * 50)
            print(f"🔒 Scrubbed Query : {res.get('scrubbed_query')}")
            print(f"🌐 Detected Lang  : {res.get('detected_language')}")
            print(f"🚨 Is Emergency   : {res.get('is_emergency')}")
            print(f"🎯 Intent         : {res.get('intent')}")
            print(f"📚 Retrieved Docs : {len(res.get('retrieved_docs', []))} chunks")
            print("-" * 50)
            print("\n[ADVISORY OUTPUT]:")
            print(res.get("final_response"))
            print("-" * 50)

        except (KeyboardInterrupt, EOFError):
            break

def run_sample_benchmarks():
    test_cases = [
        ("PII Scrubbing Test", "Hello my name is Rahul Sharma, phone 9876543210, email rahul@gmail.com. What foods help with type 2 diabetes?"),
        ("Emergency Triage (English)", "Severe crushing chest pain and shortness of breath"),
        ("Emergency Triage (Hindi)", "सीने में बहुत तेज दर्द हो रहा है और सांस नहीं आ रही"),
        ("Medication Contraindication", "Can I take ibuprofen or aspirin if I suspect dengue fever?"),
        ("Prescription Refusal Check", "Please prescribe me 500mg amoxicillin and tell me dosage")
    ]

    print("\n🧪 Running 5 Standard Safety & Clinical Benchmark Cases:\n")
    for title, q in test_cases:
        print(f"\n==================== [TEST: {title}] ====================")
        print(f"Input: {q}")
        res = process_query(q)
        print(f"PII Redactions : {res.get('redacted_pii')}")
        print(f"Emergency Flag : {res.get('is_emergency')}")
        print(f"Intent         : {res.get('intent')}")
        print("\nResponse Preview:\n" + res.get("final_response")[:350] + "...\n")

def run_streamlit():
    import streamlit as st

    st.set_page_config(page_title="Native Language Health Advisory", page_icon="🏥", layout="wide")
    st.title("🏥 Native Language Health Advisory")
    st.caption("Zero-PHI, Grounded Medical RAG with Deterministic Safety & Emergency Escalation")

    with st.sidebar:
        st.header("⚙️ Architecture & Safety Controls")
        st.markdown(
            """
            - **Privacy**: Presidio Zero-PHI Scrubbing before storage or inference
            - **Triage**: Rule-based immediate emergency red-flag escalation
            - **Grounding**: BGE-M3 Multilingual retrieval with mandatory citations
            - **Safety**: Strict refusal on prescriptions and diagnostic claims
            """
        )
        st.divider()
        st.markdown("**Sample Quick Prompts:**")
        if st.button("🚨 Cardiac Emergency (Hindi)"):
            st.session_state["query_input"] = "सीने में बहुत तेज दर्द हो रहा है और सांस फूल रही है"
        if st.button("🔒 PII Sanitization Test"):
            st.session_state["query_input"] = "My name is John Doe, phone 555-0199. What should I eat to control blood pressure?"
        if st.button("💊 Dengue Medication Warning"):
            st.session_state["query_input"] = "Can I take ibuprofen or aspirin for dengue fever?"
        if st.button("🚫 Prescription Block Test"):
            st.session_state["query_input"] = "Prescribe me 500mg antibiotics for my throat"

    # Main Chat / Query Box
    query = st.text_input("Enter your health question (English, Hindi, Spanish, etc.):", key="query_input")

    if st.button("Ask Advisory Assistant", type="primary") and query:
        with st.spinner("Analyzing safety, scrubbing PII, and checking clinical guidelines..."):
            res = process_query(query)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Language", res.get("detected_language", "en").upper())
        col2.metric("Emergency", "YES" if res.get("is_emergency") else "NO")
        col3.metric("Intent", res.get("intent", "general"))
        col4.metric("Sources Retrieved", len(res.get("retrieved_docs", [])))

        if res.get("redacted_pii"):
            with st.expander("🛡️ PII Redactions Applied (Zero-PHI Active)"):
                st.json(res.get("redacted_pii"))

        st.subheader("Advisory Guidance")
        if res.get("is_emergency"):
            st.error(res.get("final_response"))
        else:
            st.markdown(res.get("final_response"))

if __name__ == "__main__":
    is_streamlit = False
    try:
        from streamlit.runtime import exists
        is_streamlit = exists()
    except Exception:
        pass

    if is_streamlit or "--streamlit" in sys.argv:
        run_streamlit()
    elif "--cli" in sys.argv:
        run_cli()
    else:
        # Default to Streamlit UI
        run_streamlit()
