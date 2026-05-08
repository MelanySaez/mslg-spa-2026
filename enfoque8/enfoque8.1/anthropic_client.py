"""Cliente Anthropic — shim a enfoque7/anthropic_client.py."""

import importlib.util
import os

_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "..", "enfoque7", "anthropic_client.py")
_spec = importlib.util.spec_from_file_location("e7_anthropic_client", _path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

translate = _mod.translate
