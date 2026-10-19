from typing import Optional, Dict, Any, List
import re

class EmergencyTriageDetector:
    """
    Deterministic Red-Flag Detection:
    Detects critical emergencies (chest pain, stroke, suicidal ideation, severe bleeding)
    in English, Hindi (Devanagari & Hinglish), and Spanish.
    Bypasses LLM generation to guarantee safety and immediate escalation.
    """

    RED_FLAGS: Dict[str, Dict[str, Any]] = {
        "CARDIAC_EMERGENCY": {
            "patterns": [
                r"\b(chest pain|heart attack|crushing chest|angina|cardiac arrest)\b",
                r"(?:सीने|छाती)\s*(?:में)?.*?(?:दर्द|जलन|दबाव)",
                r"(दिल का दौरा|हार्ट अटैक)",
                r"\b(chhati|seene)\s*(me)?.*?(dard|jalan|attack)\b",
                r"\b(dolor de pecho|infarto|ataque al corazon)\b"
            ],
            "severity": "CRITICAL",
            "condition": "Suspected Acute Cardiac Event / Heart Attack",
            "advice": "Please sit quietly and do not strain. Call emergency services immediately. If aspirin is available and you are not allergic or bleeding, medical operators may advise chewing one."
        },
        "STROKE_FAST": {
            "patterns": [
                r"\b(face drooping|arm weakness|slurred speech|stroke symptoms|paralysis)\b",
                r"(पक्षाघात|लकवा|मुंह टेढ़ा होना|हाथ सुन्न|बोलने में दिक्कत)",
                r"\b(chehra tedha|lakwa|bolne me pareshani|stroke)\b",
                r"\b(paralisis facial|derrame cerebral|apoplejia)\b"
            ],
            "severity": "CRITICAL",
            "condition": "Suspected Acute Stroke (FAST Protocol)",
            "advice": "Every minute counts. Note the exact time symptoms started and reach the nearest stroke-ready emergency hospital immediately."
        },
        "RESPIRATORY_DISTRESS": {
            "patterns": [
                r"\b(can't breathe|cannot breathe|suffocating|severe shortness of breath|gasping for air|turning blue)\b",
                r"(सांस नहीं आ रही|दम घुट रहा है|सांस फूल रही है|सांस लेने में बहुत तकलीफ)",
                r"\b(saans nahi aa rahi|dam ghut raha|saans fulna|severe saas ki takleef)\b",
                r"\b(no puedo respirar|asfixia|falta severa de aire)\b"
            ],
            "severity": "CRITICAL",
            "condition": "Severe Acute Respiratory Distress",
            "advice": "Stay in an upright sitting position. Loosen tight clothing. Seek immediate medical emergency care."
        },
        "SUICIDAL_OR_SELF_HARM": {
            "patterns": [
                r"\b(suicide|kill myself|want to die|end my life|self harm|hang myself)\b",
                r"(आत्महत्या|खुदकुशी|जान देना चाहता|मरना चाहता)",
                r"\b(aatmahatya|khudkushi|marna chahta|marne ka man)\b",
                r"\b(suicidio|quitarme la vida|matarme)\b"
            ],
            "severity": "CRITICAL_MENTAL_HEALTH",
            "condition": "Acute Crisis & Self-Harm Prevention",
            "advice": "You are not alone, and help is available right now. Please reach out to a trained crisis counselor immediately."
        },
        "SEVERE_HEMORRHAGE": {
            "patterns": [
                r"\b(severe bleeding|uncontrolled bleeding|gushing blood|vomiting blood|coughing blood)\b",
                r"(भारी रक्तस्राव|खून की उल्टी|खून बहना नहीं रुक रहा)",
                r"\b(khoon ki ulti|bohot khoon beh raha|severe bleeding)\b",
                r"\b(hemorragia severa|vomitando sangre|sangrado incontrolable)\b"
            ],
            "severity": "CRITICAL",
            "condition": "Uncontrolled Severe Hemorrhage",
            "advice": "Apply direct, continuous, firm pressure to the bleeding wound using a clean cloth. Elevate if possible and call emergency services immediately."
        }
    }

    def __init__(self):
        # Precompile regular expressions for microsecond performance
        self.compiled_rules = []
        for category, info in self.RED_FLAGS.items():
            for p in info["patterns"]:
                self.compiled_rules.append({
                    "category": category,
                    "condition": info["condition"],
                    "severity": info["severity"],
                    "advice": info["advice"],
                    "regex": re.compile(p, re.IGNORECASE)
                })

    def check(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Scans query for red-flag patterns.
        Returns emergency details if detected, else None.
        """
        for rule in self.compiled_rules:
            if rule["regex"].search(query):
                return {
                    "is_emergency": True,
                    "category": rule["category"],
                    "condition": rule["condition"],
                    "severity": rule["severity"],
                    "immediate_advice": rule["advice"],
                    "emergency_helplines": {
                        "India": "112 (National Emergency), 108 (Ambulance), 14416 (Tele-MANAS Mental Health)",
                        "USA": "911 (Emergency), 988 (Suicide & Crisis Lifeline)",
                        "UK / Europe": "999 / 112 (Emergency Services)"
                    }
                }
        return None
