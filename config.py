import os
from dotenv import load_dotenv

load_dotenv("api.env")

LLM_BACKEND          = os.getenv("LLM_BACKEND", "mistral")
MISTRAL_API_KEY       = os.getenv("MISTRAL_API_KEY", "")
MISTRAL_VISION_MODEL  = "pixtral-12b-2409"
MISTRAL_CHAT_MODEL    = "mistral-small-latest"
MISTRAL_EMBED_MODEL   = "mistral-embed"

OLLAMA_BASE_URL      = "http://localhost:11434"
OLLAMA_VISION_MODEL  = "llava"
OLLAMA_EMBED_MODEL   = "nomic-embed-text"
OLLAMA_CHAT_MODEL    = "mistral"

CHROMA_PATH       = "./chroma_db"
CHROMA_COLLECTION = "rag_db1"
BATCH_SIZE        = 5
PDF_DPI           = 150
OUTPUT_JSON       = "chunks.json"
FAILED_PAGES_LOG  = "failed_pages.json"
MAX_RETRIES       = 3
RETRY_DELAY       = 3
