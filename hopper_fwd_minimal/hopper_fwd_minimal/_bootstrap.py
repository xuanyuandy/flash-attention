"""Import helpers for using FA4 CuTe sources without building FA2 extensions."""

from __future__ import annotations

import importlib.machinery
import sys
import types
from pathlib import Path


def ensure_repo_flash_attn_cute() -> None:
    """Expose ``flash_attn.cute`` as a source package without running FA2 init.

    The repository's top-level ``flash_attn/__init__.py`` imports
    ``flash_attn_2_cuda``. That extension is unrelated to FA4/CuTe, and it is
    often not built in lightweight Modal or local source-tree workflows.
    """
    repo_root = Path(__file__).resolve().parents[2]
    flash_attn_root = repo_root / "flash_attn"
    cute_root = flash_attn_root / "cute"

    if "flash_attn" not in sys.modules:
        pkg = types.ModuleType("flash_attn")
        pkg.__path__ = [str(flash_attn_root)]
        pkg.__package__ = "flash_attn"
        pkg.__file__ = str(flash_attn_root / "__init__.py")
        pkg.__spec__ = importlib.machinery.ModuleSpec("flash_attn", loader=None, is_package=True)
        sys.modules["flash_attn"] = pkg

    if "flash_attn.cute" not in sys.modules:
        cute_pkg = types.ModuleType("flash_attn.cute")
        cute_pkg.__path__ = [str(cute_root)]
        cute_pkg.__package__ = "flash_attn.cute"
        cute_pkg.__file__ = str(cute_root / "__init__.py")
        cute_pkg.__spec__ = importlib.machinery.ModuleSpec(
            "flash_attn.cute", loader=None, is_package=True
        )
        sys.modules["flash_attn.cute"] = cute_pkg
