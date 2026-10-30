"""
Single, explicit place this service pulls its durable-storage primitives
from.

ORIGINAL INTENT (preserved in git history): import `ActionMarker`,
`confirm_action`, `CheckpointIntegrityError`, and a signing-capable
`FileCheckpointStore` from the real Lar source at
``/Users/aadithya/Desktop/Lar_Main/lar/src/lar`` (frozen at v2.3.0).

REAL, VERIFIED FINDING (2026-10-30): that frozen source does not define
any of those four names. Checked directly, not assumed: ``checkpoint.py``
there currently defines exactly three things --
``Checkpoint``, ``FileCheckpointStore``, ``resume_human_decision`` -- and
``FileCheckpointStore.__init__`` takes only ``directory``, no
``hmac_secret``. Its own ``save()`` is a direct ``json.dump`` to the
target path, not atomic (no temp-file + ``os.replace``), despite this
service's earlier docstrings describing it as already atomic. Also
checked the public mirror, ``github.com/snath-ai/lar`` -- byte-identical
``checkpoint.py``, same gap, so this isn't local drift in the frozen copy.
Lar genuinely does not currently provide these primitives at this API
surface, anywhere.

So this module no longer imports them from Lar. It still imports the one
thing that DOES exist and DOES match -- ``Checkpoint``, a plain
case_id/resume_node_id/state/reason/metadata/created_at container, reused
as-is -- and provides local, self-contained implementations of the other
four, preserving the exact behavioural contract ``storage.py`` and the
test suite (``test_crash_recovery.py``, ``test_tamper_detection.py``)
already depend on:

  - ``FileCheckpointStore(directory, hmac_secret=None)``: atomic writes
    (temp file + ``os.replace``, same-directory so the rename is atomic
    on a single POSIX volume), HMAC-SHA256-signed when ``hmac_secret`` is
    set (a top-level ``signature`` field over the canonical JSON of
    everything else), unsigned and unverified when it is not -- the exact
    opt-in contract ``test_no_secret_configured_means_no_signature_at_all``
    checks.
  - ``ActionMarker`` / ``confirm_action`` / ``find_unconfirmed_markers``
    (as a method on the store): the same ATTEMPTING -> COMPLETED/FAILED
    uncertain-action protocol ``storage.commit_library_entry`` already
    assumed, backed by the same atomic+signed file store above (one
    marker per (case_id, step), keyed so a later confirm overwrites the
    ATTEMPTING entry in place rather than appending a duplicate).
  - ``CheckpointIntegrityError``: raised by ``load()`` on a signature
    mismatch.

Nothing in ``storage.py``, ``app.py``, ``models.py``, or
``consolidation.py`` needed to change -- they only ever imported these
five names from this module, never from ``lar`` directly, which is the
whole reason this was "the single, explicit place" to begin with.
"""
from __future__ import annotations

import hashlib
import hmac as hmac_mod
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

_LAR_SRC = "/Users/aadithya/Desktop/Lar_Main/lar/src"

if _LAR_SRC not in sys.path:
    sys.path.insert(0, _LAR_SRC)

from lar.checkpoint import Checkpoint  # noqa: E402  (this one genuinely matches; reused as-is)

__all__ = [
    "Checkpoint",
    "FileCheckpointStore",
    "ActionMarker",
    "confirm_action",
    "CheckpointIntegrityError",
]


class CheckpointIntegrityError(Exception):
    """Raised when a loaded JSON file's recomputed HMAC doesn't match its
    stored 'signature' field -- the file was modified after signing."""


class ActionMarker:
    ATTEMPTING = "ATTEMPTING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

    def __init__(self, case_id: str, node_id: str, step: int, status: str,
                 result: Optional[Dict[str, Any]] = None):
        self.case_id = case_id
        self.node_id = node_id
        self.step = step
        self.status = status
        self.result = result

    def to_dict(self) -> dict:
        return {"case_id": self.case_id, "node_id": self.node_id,
                "step": self.step, "status": self.status, "result": self.result}

    @classmethod
    def from_dict(cls, d: dict) -> "ActionMarker":
        return cls(case_id=d["case_id"], node_id=d["node_id"], step=d["step"],
                   status=d["status"], result=d.get("result"))

    def __repr__(self):
        return (f"ActionMarker(case_id={self.case_id!r}, node_id={self.node_id!r}, "
                f"step={self.step}, status={self.status!r})")


def _canonical(d: dict) -> bytes:
    return json.dumps(d, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sign(d: dict, hmac_secret: str) -> str:
    return hmac_mod.new(hmac_secret.encode("utf-8"), _canonical(d), hashlib.sha256).hexdigest()


class FileCheckpointStore:
    """
    One JSON file per case_id, written atomically (temp file + os.replace,
    same directory so the rename is on a single volume) and optionally
    HMAC-signed. Also stores action markers, one JSON file per case_id
    under a derived name, same atomic+signed primitive.

    Not a distributed store, no cross-process locking of its own (the
    caller -- DMNStore -- already serializes writes with a process-wide
    lock); matches the scope FileCheckpointStore has always documented.
    """

    def __init__(self, directory: str, hmac_secret: Optional[str] = None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.hmac_secret = hmac_secret

    def _path(self, case_id: str) -> Path:
        return self.directory / f"{case_id}.json"

    def _markers_path(self, case_id: str) -> Path:
        return self.directory / f"{case_id}__markers.json"

    def _write_atomic(self, path: Path, payload: dict) -> None:
        body = dict(payload)
        if self.hmac_secret:
            body["signature"] = _sign(payload, self.hmac_secret)
        fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(body, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_name, path)
        except BaseException:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
            raise

    def _read_verified(self, path: Path) -> Optional[dict]:
        if not path.exists():
            return None
        with open(path) as f:
            on_disk = json.load(f)
        signature = on_disk.pop("signature", None)
        if self.hmac_secret:
            expected = _sign(on_disk, self.hmac_secret)
            if signature is None or not hmac_mod.compare_digest(signature, expected):
                raise CheckpointIntegrityError(
                    f"Signature mismatch for {path.name}: file was modified after signing, "
                    f"or signed with a different secret."
                )
        return on_disk

    # ------------------------------------------------------------ checkpoints

    def save(self, checkpoint: Checkpoint) -> None:
        self._write_atomic(self._path(checkpoint.case_id), checkpoint.to_dict())

    def load(self, case_id: str) -> Optional[Checkpoint]:
        d = self._read_verified(self._path(case_id))
        return Checkpoint.from_dict(d) if d is not None else None

    def delete(self, case_id: str) -> None:
        path = self._path(case_id)
        if path.exists():
            path.unlink()

    def exists(self, case_id: str) -> bool:
        return self._path(case_id).exists()

    # --------------------------------------------------------------- markers

    def _load_markers(self, case_id: str) -> Dict[int, ActionMarker]:
        d = self._read_verified(self._markers_path(case_id))
        if d is None:
            return {}
        return {int(k): ActionMarker.from_dict(v) for k, v in d.get("markers", {}).items()}

    def _save_markers(self, case_id: str, markers: Dict[int, ActionMarker]) -> None:
        payload = {"markers": {str(step): m.to_dict() for step, m in markers.items()}}
        self._write_atomic(self._markers_path(case_id), payload)

    def save_marker(self, marker: ActionMarker) -> None:
        markers = self._load_markers(marker.case_id)
        markers[marker.step] = marker
        self._save_markers(marker.case_id, markers)

    def find_unconfirmed_markers(self, case_id: str) -> List[ActionMarker]:
        markers = self._load_markers(case_id)
        return [m for m in markers.values() if m.status == ActionMarker.ATTEMPTING]


def confirm_action(checkpoint_store: FileCheckpointStore, case_id: str, node_id: str,
                    step: int, happened: bool, result: Optional[Dict[str, Any]] = None) -> ActionMarker:
    marker = ActionMarker(case_id=case_id, node_id=node_id, step=step,
                           status=ActionMarker.COMPLETED if happened else ActionMarker.FAILED,
                           result=result)
    checkpoint_store.save_marker(marker)
    return marker
