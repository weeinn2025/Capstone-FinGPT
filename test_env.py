# test_env.py
from dotenv import load_dotenv, dotenv_values
import os

load_dotenv('.env')
print("GEMINI_API_KEY:", os.getenv("GEMINI_API_KEY"))
print("GEMINI_URL:   ", os.getenv("GEMINI_URL"))
