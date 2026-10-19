import re
import os
from typing import Dict, Any, List
from src.config import Config
from src.safety.pii_scrubber import PIIScrubber
from src.safety.emergency_triage import EmergencyTriageDetector
from src.rag.retriever import MedicalRetriever

# Lazy singletons for memory efficiency
_pii_scrubber = None
_emergency_detector = None
_retriever = None

def get_pii_scrubber() -> PIIScrubber:
    global _pii_scrubber
    if _pii_scrubber is None:
        _pii_scrubber = PIIScrubber()
    return _pii_scrubber

def get_emergency_detector() -> EmergencyTriageDetector:
    global _emergency_detector
    if _emergency_detector is None:
        _emergency_detector = EmergencyTriageDetector()
    return _emergency_detector

def get_retriever() -> MedicalRetriever:
    global _retriever
    if _retriever is None:
        _retriever = MedicalRetriever()
    return _retriever


def pii_scrub_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node 1: Scrubs PII from user query to maintain Zero-PHI."""
    raw = state.get("raw_query", "")
    scrubber = get_pii_scrubber()
    clean_text, redacted = scrubber.scrub(raw)
    return {
        "scrubbed_query": clean_text,
        "redacted_pii": redacted
    }


def language_detect_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node 2: Detects user's native language.
    Features an English stopword heuristic and Devanagari script detection
    to prevent short queries from falsely mapping to Norwegian ('no') or Dutch ('nl').
    """
    text = state.get("scrubbed_query", "")
    text_lower = text.lower()

    # 1. Devanagari script detection (Hindi)
    if re.search(r"[\u0900-\u097F]", text):
        return {"detected_language": "hi", "language_confidence": 0.99}

    # 2. English functional markers heuristic (handles short medical inquiries)
    english_markers = {
        "can", "i", "take", "what", "should", "my", "having", "have", "is", 
        "the", "with", "how", "do", "does", "why", "when", "which", "fever", 
        "pain", "headache", "blood", "doctor", "medicine", "pill"
    }
    words = set(re.findall(r"\b[a-z]+\b", text_lower))
    if len(words.intersection(english_markers)) >= 2:
        return {"detected_language": "en", "language_confidence": 0.98}

    # 3. Probabilistic detection with Scandinavian/Dutch guard
    try:
        from langdetect import detect_langs
        langs = detect_langs(text)
        top = langs[0]
        candidate_codes = [l.lang for l in langs]
        
        # If langdetect guesses obscure/Germanic false-positives on English text, bias to English
        if "en" in candidate_codes and top.lang in ["no", "af", "so", "da", "nl", "sv"]:
            lang = "en"
        else:
            lang = top.lang
    except Exception:
        lang = "en"
            
    return {"detected_language": lang, "language_confidence": 0.95}


def emergency_triage_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node 3: Deterministic check for life-threatening emergencies.
    Checks both raw and scrubbed query to ensure English NER models
    do not inadvertently redact non-English symptom descriptions (e.g., Hinglish/Spanish).
    """
    raw_text = state.get("raw_query", "")
    scrubbed_text = state.get("scrubbed_query", "")
    detector = get_emergency_detector()
    result = detector.check(raw_text) or detector.check(scrubbed_text)

    if result:
        return {
            "is_emergency": True,
            "emergency_info": result
        }
    return {
        "is_emergency": False,
        "emergency_info": None
    }


def intent_router_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node 4: Classifies intent and blocks diagnosis or prescription demands."""
    text = state.get("scrubbed_query", "").lower()

    # Refusal triggers: requesting specific prescription drugs or personal diagnosis
    rx_patterns = [
        r"\b(prescribe|write me a prescription|which antibiotic should i buy|dosage of (xanax|adderall|steroid|amoxicillin))\b",
        r"\b(diagnose me|do i have cancer|tell me exactly what disease i have)\b",
        r"(दवा लिख दो|पर्चा बना दो|कौन सी एंटीबायोटिक खाऊं)"
    ]

    for p in rx_patterns:
        if re.search(p, text):
            return {
                "intent": "prescription_diagnosis_refusal",
                "refusal_reason": (
                    "I cannot diagnose conditions or prescribe/recommend specific medication dosages. "
                    "Only a qualified, licensed medical practitioner can prescribe medications after in-person clinical assessment."
                )
            }

    return {"intent": "health_inquiry", "refusal_reason": None}


def retrieval_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node 5: Hybrid cross-lingual semantic retrieval against Qdrant."""
    query = state.get("scrubbed_query", "")
    try:
        retriever = get_retriever()
        docs = retriever.retrieve(query, top_k=Config.TOP_K)
    except Exception as e:
        print(f"[Retrieval Node Error] {e}")
        docs = []

    return {"retrieved_docs": docs}


def generator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node 6: Grounded generation strictly from retrieved context in the user's native language."""
    docs = state.get("retrieved_docs", [])
    query = state.get("scrubbed_query", "")
    lang = state.get("detected_language", "en")

    if not docs:
        fallback_msg = (
            "I could not find sufficiently authoritative clinical guidelines in the database to answer your specific inquiry. "
            "Please consult a certified medical practitioner or local health authority for verified information."
        )
        if lang == "hi":
            fallback_msg = (
                "आपके इस प्रश्न के लिए आधिकारिक स्वास्थ्य दिशानिर्देश डेटाबेस में उपलब्ध नहीं हैं। "
                "कृपया उचित सलाह के लिए किसी प्रमाणित चिकित्सक या नजदीकी स्वास्थ्य केंद्र से परामर्श लें।"
            )
        return {
            "generated_text": fallback_msg,
            "citations": []
        }

    # Format context for grounding
    context_str = ""
    citations = []
    for idx, d in enumerate(docs):
        context_str += f"\n--- Source [{idx+1}]: {d['doc_title']} ({d['section_title']}) ---\n{d['content']}\n"
        citations.append({"id": f"[{idx+1}]", "title": d["doc_title"], "section": d["section_title"], "source": d["source"]})

    # Try calling configured free LLM (Gemini or Groq)
    generated = None
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    groq_key = os.getenv("GROQ_API_KEY", "")

    system_prompt = f"""You are a specialized Native Language Health Advisory assistant.
CRITICAL SAFETY & CLINICAL RULES:
1. Rely EXCLUSIVELY on the provided clinical context. Never speculate, diagnose diseases, or prescribe drug dosages.
2. If the user's question cannot be answered purely from the provided context, clearly state that you don't have sufficient verified guidance.
3. Every health statement MUST cite the source marker like [1], [2].
4. Answer in the user's native language (Language detected: '{lang}'). Keep explanations clear, empathetic, and plain-language (approx. 7th-grade reading level).
5. If dengue is discussed, emphasize avoiding NSAIDs (ibuprofen/aspirin).
"""

    user_prompt = f"""CLINICAL CONTEXT:
{context_str}

USER INQUIRY:
{query}

Provide a grounded, cited health advisory response in language code '{lang}':"""

    # 1. Attempt Gemini if key is provided
    if not generated and gemini_key and gemini_key != "your_gemini_api_key_here":
        try:
            from google import genai
            client = genai.Client(api_key=gemini_key)
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            # Robust model cascade: Try newest active flash models in order
            candidate_models = ["gemini-3.5-flash-lite", "gemini-3.8-flash", "gemini-2.5-flash", "gemini-1.5-flash"]
            for m in candidate_models:
                try:
                    response = client.models.generate_content(
                        model=m,
                        contents=full_prompt,
                    )
                    if response.text:
                        generated = response.text
                        break
                except Exception:
                    continue
        except Exception as e:
            print(f"[Generator] Gemini SDK error: {e}")

    # 2. Attempt Groq if key is provided
    if not generated and groq_key and groq_key != "your_groq_api_key_here":
        try:
            from langchain_groq import ChatGroq
            llm = ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=groq_key, temperature=0.1)
            response = llm.invoke([("system", system_prompt), ("human", user_prompt)])
            generated = response.content
        except Exception as e:
            print(f"[Generator] Groq API error: {e}")

    # 3. Deterministic Grounded Synthesizer Fallback (Runs offline without any API key)
    if not generated:
        summary_lines = []
        for d in docs[:2]:
            summary_lines.append(f"• According to [{d['doc_title']} - {d['section_title']}]: {d['content'][:250]}...")
        
        if lang == "hi":
            generated = (
                f"सत्यापित स्वास्थ्य दिशानिर्देशों के आधार पर संक्षेप:\n\n" +
                "\n\n".join(summary_lines) +
                "\n\n(विस्तृत जानकारी के लिए नीचे दिए गए संदर्भ देखें।)"
            )
        else:
            generated = (
                f"Summary based on official clinical guidelines:\n\n" +
                "\n\n".join(summary_lines) +
                "\n\n(Please refer to the source references below for full guidance.)"
            )

    return {
        "generated_text": generated,
        "citations": citations
    }


def safety_postprocessor_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node 7: Attaches mandatory clinical disclaimers and emergency helplines."""
    is_emergency = state.get("is_emergency", False)
    intent = state.get("intent", "")
    lang = state.get("detected_language", "en")

    if is_emergency:
        em = state.get("emergency_info", {})
        helplines = em.get("emergency_helplines", {})
        
        if lang == "hi":
            header = f"🚨 **आपातकालीन चेतावनी: {em.get('condition')}** 🚨\n\n"
            advice = f"**तत्काल सलाह:** {em.get('immediate_advice')}\n\n"
            contacts = "**तत्काल आपातकालीन नंबर:**\n"
        else:
            header = f"🚨 **MEDICAL EMERGENCY ALERT: {em.get('condition')}** 🚨\n\n"
            advice = f"**Immediate Action:** {em.get('immediate_advice')}\n\n"
            contacts = "**Emergency Contacts:**\n"

        for country, num in helplines.items():
            contacts += f"- **{country}**: {num}\n"

        return {
            "final_response": f"{header}{advice}{contacts}",
            "disclaimer": "Emergency escalation triggered: LLM bypassed."
        }

    if intent == "prescription_diagnosis_refusal":
        reason = state.get("refusal_reason", "Prescription or direct diagnosis requests are out of safety scope.")
        disclaimer = "⚠️ Out of Scope / Clinical Safety Policy."
        return {
            "final_response": f"### Advisory Refusal\n\n{reason}\n\n*Please seek an in-person consultation with a qualified doctor.*",
            "disclaimer": disclaimer
        }

    # Standard grounded response with disclaimer
    text = state.get("generated_text", "")
    citations = state.get("citations", [])

    citation_block = "\n\n**Sources & Provenance:**\n"
    for c in citations:
        citation_block += f"- {c['id']} *{c['title']}* ({c['section']}) - Source: {c['source']}\n"

    disclaimer = (
        "\n\n---\n"
        "ℹ️ **Health Advisory Disclaimer:** *This guidance is for educational and informational purposes only. "
        "It does not constitute medical diagnosis, treatment, or prescription. If you experience severe symptoms or sudden deterioration, "
        "contact emergency services (112 / 108 / 911) immediately.*"
    )

    return {
        "final_response": text + citation_block + disclaimer,
        "disclaimer": disclaimer
    }
