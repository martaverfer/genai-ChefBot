# GenAI
import os
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# const
CHROMA_PATH = "../chroma"
DATA_PATH = "../data"
api_key = os.getenv("OPENAI_API_KEY")
