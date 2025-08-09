# test_env.py
# from dotenv import load_dotenv, dotenv_values
# import os

# load_dotenv(".env")
# print("GEMINI_API_KEY:", os.getenv("GEMINI_API_KEY"))
# print("GEMINI_URL:   ", os.getenv("GEMINI_URL"))

from dotenv import dotenv_values
from pathlib import Path


def test_env_example_parses():
    assert Path(".env.example").exists()
    dotenv_values(".env.example")  # just ensure it parses
