import unittest
from src.safety.pii_scrubber import PIIScrubber
from src.safety.emergency_triage import EmergencyTriageDetector

class TestSafetyLayer(unittest.TestCase):
    def setUp(self):
        self.scrubber = PIIScrubber()
        self.triage = EmergencyTriageDetector()

    def test_pii_email_redaction(self):
        text = "Contact me at patient.test@hospital.org regarding my test report."
        clean, redacted = self.scrubber.scrub(text)
        self.assertIn("[EMAIL_REDACTED]", clean)
        self.assertNotIn("patient.test@hospital.org", clean)

    def test_pii_phone_redaction(self):
        text = "My contact number is 9876543210 please call soon."
        clean, redacted = self.scrubber.scrub(text)
        self.assertIn("[PHONE_REDACTED]", clean)
        self.assertNotIn("9876543210", clean)

    def test_pii_aadhaar_redaction(self):
        text = "Aadhaar ID: 1234 5678 9012"
        clean, redacted = self.scrubber.scrub(text)
        self.assertIn("[ID_REDACTED]", clean)
        self.assertNotIn("1234 5678 9012", clean)

    def test_emergency_cardiac_english(self):
        query = "I am experiencing severe chest pain and breathlessness"
        res = self.triage.check(query)
        self.assertIsNotNone(res)
        self.assertTrue(res["is_emergency"])
        self.assertEqual(res["category"], "CARDIAC_EMERGENCY")

    def test_emergency_cardiac_hindi(self):
        query = "मेरे सीने में बहुत तेज दर्द हो रहा है"
        res = self.triage.check(query)
        self.assertIsNotNone(res)
        self.assertTrue(res["is_emergency"])

    def test_emergency_stroke_hinglish(self):
        query = "Achanak chehra tedha ho gaya aur bolne me pareshani ho rahi hai"
        res = self.triage.check(query)
        self.assertIsNotNone(res)
        self.assertTrue(res["is_emergency"])
        self.assertEqual(res["category"], "STROKE_FAST")

    def test_emergency_suicide_detection(self):
        query = "I cannot take this anymore and want to end my life"
        res = self.triage.check(query)
        self.assertIsNotNone(res)
        self.assertTrue(res["is_emergency"])
        self.assertEqual(res["category"], "SUICIDAL_OR_SELF_HARM")

    def test_non_emergency_safe_query(self):
        query = "What are the common symptoms of type 2 diabetes?"
        res = self.triage.check(query)
        self.assertIsNone(res)

if __name__ == "__main__":
    unittest.main()
