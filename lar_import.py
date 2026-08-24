"""
Single, explicit place this service pulls Lar from.

The `lar` package is ALSO pip-installed globally (editable, version 2.2.0,
pulled in as a dependency of an unrelated pre-existing project called
"lar-dmn" under ~/Desktop/Lar_Main/DMN -- a different "DMN" concept, brain/
bicameral-memory cognitive architecture work, name collision only, nothing
to do with this service). That global install does NOT have
`checkpoint.py` / the HMAC-signing `AuditLogger.verify_signature` this
service depends on -- it's an older snapshot.

The real, current Lar source -- the one with `FileCheckpointStore`,
`ActionMarker`, `confirm_action`, `CheckpointIntegrityError`, and
`AuditLogger.verify_signature` -- lives at
``/Users/aadithya/Desktop/Lar_Main/lar/src/lar`` (version 2.3.0, 176+ tests
passing under ``PYTHONPATH=src pytest``). This module puts that exact path
at the FRONT of ``sys.path`` before anything imports ``lar``, so this
service always gets the intended 2.3.0 checkout regardless of what's on the
global site-packages path -- verified by checking ``lar.__file__`` /
``lar.__version__`` below at import time (fails loudly if the wrong one
loads).
"""
import sys
from pathlib import Path

_LAR_SRC = "/Users/aadithya/Desktop/Lar_Main/lar/src"

if _LAR_SRC not in sys.path:
    sys.path.insert(0, _LAR_SRC)

import lar  # noqa: E402

_resolved = Path(lar.__file__).resolve()
_expected = Path(_LAR_SRC, "lar", "__init__.py").resolve()
if _resolved != _expected:
    raise ImportError(
        f"Wrong 'lar' package resolved: got {_resolved}, expected {_expected}. "
        f"This usually means something already imported 'lar' from the global "
        f"site-packages (v2.2.0, no checkpoint.py) before this module ran. "
        f"Import lar_import (or dmn_service.storage) before any other 'lar' import."
    )
if not hasattr(lar, "FileCheckpointStore"):
    raise ImportError(
        "Resolved lar package has no FileCheckpointStore -- wrong version loaded."
    )

from lar import (  # noqa: E402
    FileCheckpointStore,
    Checkpoint,
    ActionMarker,
    UnconfirmedActionError,
    CheckpointIntegrityError,
    confirm_action,
    AuditLogger,
)

__all__ = [
    "lar",
    "FileCheckpointStore",
    "Checkpoint",
    "ActionMarker",
    "UnconfirmedActionError",
    "CheckpointIntegrityError",
    "confirm_action",
    "AuditLogger",
]
