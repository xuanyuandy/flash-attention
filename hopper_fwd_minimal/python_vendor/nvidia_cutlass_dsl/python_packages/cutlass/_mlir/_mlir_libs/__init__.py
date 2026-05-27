# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any, Sequence

import os

_this_dir = os.path.dirname(__file__)

def get_lib_dirs() -> Sequence[str]:
    """Gets the lib directory for linking to shared libraries.

    On some platforms, the package may need to be built specially to export
    development libraries.
    """
    return [_this_dir]

def get_include_dirs() -> Sequence[str]:
    """Gets the include directory for compiling against exported C libraries.

    Depending on how the package was build, development C libraries may or may
    not be present.
    """
    return [os.path.join(_this_dir, "include")]

_dialect_registry = None
_load_on_create_dialects = None

def get_dialect_registry():
    global _dialect_registry

    if _dialect_registry is None:
        from ._cutlass_ir._mlir import ir

        _dialect_registry = ir.DialectRegistry()

    return _dialect_registry

def append_load_on_create_dialect(dialect: str):
    global _load_on_create_dialects
    if _load_on_create_dialects is None:
        _load_on_create_dialects = [dialect]
    else:
        _load_on_create_dialects.append(dialect)

def get_load_on_create_dialects():
    global _load_on_create_dialects
    if _load_on_create_dialects is None:
        _load_on_create_dialects = []
    return _load_on_create_dialects

def _site_initialize():
    import importlib
    import itertools
    import logging
    from ._cutlass_ir._mlir import ir

    logger = logging.getLogger(__name__)
    post_init_hooks = []
    disable_multithreading = False

    disable_load_all_available_dialects = False

    def process_initializer_module(module_name):
        nonlocal disable_multithreading
        nonlocal disable_load_all_available_dialects
        try:
            m = importlib.import_module(f".{module_name}", __name__)
        except ModuleNotFoundError:
            return False
        except ImportError:
            message = (
                f"Error importing mlir initializer {module_name}. This may "
                "happen in unclean incremental builds but is likely a real bug if "
                "encountered otherwise and the MLIR Python API may not function."
            )
            logger.warning(message, exc_info=True)
            return False

        logger.debug("Initializing MLIR with module: %s", module_name)
        if hasattr(m, "register_dialects"):
            logger.debug("Registering dialects from initializer %r", m)
            m.register_dialects(get_dialect_registry())
        if hasattr(m, "context_init_hook"):
            logger.debug("Adding context init hook from %r", m)
            post_init_hooks.append(m.context_init_hook)
        if hasattr(m, "disable_multithreading"):
            if bool(m.disable_multithreading):
                logger.debug("Disabling multi-threading for context")
                disable_multithreading = True
        if hasattr(m, "disable_load_all_available_dialects"):
            disable_load_all_available_dialects = True
        return True

    init_module = None
    if process_initializer_module("_mlirRegisterEverything"):
        init_module = importlib.import_module(f"._mlirRegisterEverything", __name__)

    if not process_initializer_module("_cutlass_ir"):
        raise RuntimeError("Could not successfully initialize _cutlass_ir module.")

    class Context(ir._BaseContext):
        def __init__(self, load_on_create_dialects=None, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.append_dialect_registry(get_dialect_registry())
            for hook in post_init_hooks:
                hook(self)
            if not disable_multithreading:
                self.enable_multithreading(True)
            if load_on_create_dialects is not None:
                logger.debug(
                    "Loading all dialects from load_on_create_dialects arg %r",
                    load_on_create_dialects,
                )
                for dialect in load_on_create_dialects:

                    _ = self.dialects[dialect]
            else:
                if disable_load_all_available_dialects:
                    dialects = get_load_on_create_dialects()
                    if dialects:
                        logger.debug(
                            "Loading all dialects from global load_on_create_dialects %r",
                            dialects,
                        )
                        for dialect in dialects:

                            _ = self.dialects[dialect]
                else:
                    logger.debug("Loading all available dialects")
                    self.load_all_available_dialects()
            if init_module:
                logger.debug(
                    "Registering translations from initializer %r", init_module
                )
                init_module.register_llvm_translations(self)

    ir.Context = Context

    class MLIRError(Exception):
        """
        An exception with diagnostic information. Has the following fields:
          message: str
          error_diagnostics: List[ir.DiagnosticInfo]
        """

        def __init__(self, message, error_diagnostics):
            self.message = message
            self.error_diagnostics = error_diagnostics
            super().__init__(message, error_diagnostics)

        def __str__(self):
            s = self.message
            if self.error_diagnostics:
                s += ":"
            for diag in self.error_diagnostics:
                s += (
                    "\nerror: "
                    + str(diag.location)[4:-1]
                    + ": "
                    + diag.message.replace("\n", "\n  ")
                )
                for note in diag.notes:
                    s += (
                        "\n note: "
                        + str(note.location)[4:-1]
                        + ": "
                        + note.message.replace("\n", "\n  ")
                    )
            return s

    ir.MLIRError = MLIRError

_site_initialize()