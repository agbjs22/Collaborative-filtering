"""Ensure the project root is on sys.path for test discovery."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

