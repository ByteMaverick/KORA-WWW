"""Puts the authors' folder on sys.path: their modules import each other by bare name (main, prompt,
few_shot_llm_tools_unified_helper). Imported for that side effect only."""
import os
import sys

UPSTREAM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "upstream")
if UPSTREAM not in sys.path:
    sys.path.insert(0, UPSTREAM)
