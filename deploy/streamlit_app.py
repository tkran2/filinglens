"""Streamlit Cloud entry point for FilingLens."""

import os
import runpy
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Cloud credentials live in Streamlit's private settings.
if not os.getenv("OPENAI_API_KEY"):
    try:
        key = st.secrets.get("OPENAI_API_KEY")
    except FileNotFoundError:
        key = None
    if key:
        os.environ["OPENAI_API_KEY"] = key

runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
