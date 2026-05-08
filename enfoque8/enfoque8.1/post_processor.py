"""Post-procesamiento para salidas SPA — shim a enfoque7.2/post_processor.py.

Preserva español natural: capitalización, puntuación, acentos.
No convierte a mayúsculas ni elimina artículos (eso es para glosas MSLG).
"""

import importlib.util
import os

_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "..", "enfoque7", "enfoque7.2", "post_processor.py")
_spec = importlib.util.spec_from_file_location("e72_post_processor", _path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

clean = _mod.clean
