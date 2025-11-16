
from dotenv import load_dotenv
import os

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_API_URL = os.getenv("LLM_API_URL")
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL") or 'http://arch-ideapadg3:11434'