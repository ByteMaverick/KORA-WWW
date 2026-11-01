"""
baselines/official_loader.py
-----------------------------
Imports an authors' code folder whose modules import each other by bare name (`from utils import *`), without
letting those names leak: ToG and PoG both have `utils`, `freebase_func` and `prompt_list`. The modules are
imported from the folder, kept by reference, and removed from sys.modules again, so another folder's modules of
the same name import fresh. Nothing in the folder is changed.
"""
from __future__ import annotations

import importlib
import sys


def load(folder: str, names: list[str]) -> dict:
    """{name: module} for the authors' modules `names`, imported in that order from `folder`."""
    saved = {n: sys.modules.pop(n) for n in names if n in sys.modules}
    sys.path.insert(0, folder)
    try:
        return {n: importlib.import_module(n) for n in names}
    finally:
        sys.path.remove(folder)
        for n in names:
            sys.modules.pop(n, None)
        sys.modules.update(saved)
