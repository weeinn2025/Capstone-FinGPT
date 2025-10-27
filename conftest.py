# conftest.py
import sys
from pathlib import Path

# Ensure repository root is on sys.path for pytest
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
