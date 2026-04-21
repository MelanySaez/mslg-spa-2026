"""Post-procesamiento — reutiliza enfoque3/post_processor.py.

Claude no emite bloques <think> (a diferencia de DeepSeek-R1), así que no se
necesita la capa extra de enfoque6.
"""

import importlib.util
import os

_path = os.path.join(os.path.dirname(__file__), "..", "enfoque3", "post_processor.py")
_spec = importlib.util.spec_from_file_location("enfoque3_post_processor", _path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

clean = _mod.clean
