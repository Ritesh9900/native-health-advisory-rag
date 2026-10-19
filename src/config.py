import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env file
load_dotenv(BASE_DIR / ".env")

# Ensure all downloaded HuggingFace models/cache stay on E: drive
os.environ.setdefault("HF_HOME", str(BASE_DIR / "hf_cache"))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(BASE_DIR / "hf_cache"))

class Config:
    PROJECT_NAME = "Native Language Health Advisory RAG"
    DATA_RAW_DIR = BASE_DIR / "data" / "raw"
    DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
    QDRANT_STORAGE_PATH = os.getenv("QDRANT_STORAGE_PATH", str(BASE_DIR / "qdrant_storage"))
    QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "health_guidelines")
    
    # Embedding Model: multilingual, runs locally on CPU/GPU
    # "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" is lightweight (~470MB) and fast.
    # "BAAI/bge-m3" is state-of-the-art for cross-lingual (~2.2GB).
    EMBEDDING_MODEL_NAME = os.getenv(
        "EMBEDDING_MODEL_NAME", 
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    # LLM configuration (Free tier: Gemini or Groq)
    DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "gemini").lower()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    
    # Retrieval parameters
    TOP_K = 4
    SIMILARITY_THRESHOLD = 0.30
    
    # Emergency Helplines (Deterministic fallbacks)
    EMERGENCY_HELPLINES = {
        "IN": {"General": "112", "Ambulance": "108", "Mental Health (Tele-MANAS)": "14416"},
        "US": {"Emergency": "911", "Suicide & Crisis": "988"},
        "UK": {"Emergency": "999", "Non-Emergency NHS": "111"},
        "GLOBAL": {"Note": "Please dial your local emergency services (112 / 911 / 108) immediately."}
    }
