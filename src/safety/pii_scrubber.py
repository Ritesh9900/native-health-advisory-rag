import re
from typing import Tuple, List, Dict

class PIIScrubber:
    """
    Zero-PHI Privacy Defense:
    Detects and scrubs personal identifiable information (PII) before any text
    reaches embeddings, LLMs, or persistent stores.
    Supports Presidio with a resilient regex fallback.
    """
    def __init__(self):
        self.presidio_available = False
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_anonymizer import AnonymizerEngine
            self.analyzer = AnalyzerEngine()
            self.anonymizer = AnonymizerEngine()
            self.presidio_available = True
        except Exception:
            self.presidio_available = False

        # Resilient Regex patterns for international and Indian PII identifiers
        # Note: Specific IDs (Aadhaar, SSN) MUST run before generic phone patterns
        self.patterns: List[Tuple[str, str, re.Pattern]] = [
            ("EMAIL", "[EMAIL_REDACTED]", re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')),
            ("AADHAAR", "[ID_REDACTED]", re.compile(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b')),
            ("SSN", "[ID_REDACTED]", re.compile(r'\b\d{3}-\d{2}-\d{4}\b')),
            ("INDIAN_PHONE", "[PHONE_REDACTED]", re.compile(r'\b[6-9]\d{9}\b')),
            ("PHONE", "[PHONE_REDACTED]", re.compile(r'(\+?\d{1,3}[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b')),
            ("AGE_IDENTIFIER", "[AGE_REDACTED]", re.compile(r'\b(?:i am|age|aged)\s+(\d{1,3})\s+(?:years old|yo|yrs)?\b', re.IGNORECASE)),
            ("DOB", "[DOB_REDACTED]", re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b')),
        ]

    def scrub(self, text: str) -> Tuple[str, List[Dict[str, str]]]:
        redacted_items = []
        clean_text = text

        # 1. Use Presidio if available
        if self.presidio_available:
            try:
                from presidio_anonymizer.entities import OperatorConfig
                operators = {
                    "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "[EMAIL_REDACTED]"}),
                    "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "[PHONE_REDACTED]"}),
                    "PERSON": OperatorConfig("replace", {"new_value": "[PERSON_REDACTED]"}),
                    "US_SSN": OperatorConfig("replace", {"new_value": "[ID_REDACTED]"})
                }
                results = self.analyzer.analyze(text=text, entities=["PHONE_NUMBER", "EMAIL_ADDRESS", "PERSON", "US_SSN"], language="en")
                if results:
                    anonymized_result = self.anonymizer.anonymize(text=text, analyzer_results=results, operators=operators)
                    clean_text = anonymized_result.text
                    for r in results:
                        redacted_items.append({"entity_type": r.entity_type, "start": r.start, "end": r.end})
            except Exception:
                pass  # Fallback to regex

        # 2. Always run regex pass for guaranteed local coverage
        for label, replacement, pattern in self.patterns:
            matches = list(pattern.finditer(clean_text))
            if matches:
                for match in matches:
                    redacted_items.append({"entity_type": label, "value": match.group(0)})
                clean_text = pattern.sub(replacement, clean_text)

        return clean_text, redacted_items
