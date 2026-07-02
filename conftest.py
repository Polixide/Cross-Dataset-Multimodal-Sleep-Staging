"""Make the repository root importable so tests can use ``from src import ...``.

pytest adds the directory containing this file to sys.path, which is all the
project needs — no packaging or extra configuration.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
