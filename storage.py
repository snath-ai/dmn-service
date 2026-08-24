"""
Durable storage for the central DMN service, built ENTIRELY on Lar's
existing checkpoint primitives (``lar.checkpoint.FileCheckpointStore``,
``Checkpoint``, ``ActionMarker``, ``confirm_action``) -- no new storage
mechanism, no new signing scheme.

Two case_ids live in one ``FileCheckpointStore``:

  - ``events__<class_name>``: the durable, signed raw-event log for one
    class. Each submitted fleet event is appended here. This is the
    "episodic" tier in the LTL paper's three-tier DMN consolidation
    language (JEPA_DMN_Consolidation_Node's raw D_hard queue, HMAC-signed).

  - ``consolidated_library``: the single servable library -- one entry per
    class, each produced by ``consolidation.consolidate_class``. This is
    what GET /library serves. This is the ONE case_id whose writes go
    through the ActionMarker attempting/completed protocol (see
    ``commit_library_step1_attempt`` / ``commit_library_step2_write`` /
    ``commit_library_step3_confirm`` below) -- everything else (event
    ingest) uses plain ``FileCheckpointStore.save``, which is already
    atomic (temp file + os.replace) and needs nothing more, per Lar's own
    checkpoint.py docstring.

Why the library commit specifically gets the ActionMarker treatment
--------------------------------------------------------------------------
``FileCheckpointStore.save`` is already atomic at the filesystem level --
a reader can never observe a torn write. What atomicity does NOT give you
is a durable, inspectable RECORD of "a commit was in progress when the
process died" versus "the last commit cleanly finished." Without that
record, an operator restarting the service after a crash has no way to
tell "this library is the confirmed result of the last consolidation" from
"this library is the PREVIOUS good state because the new one never made it
to disk" -- both look identical (a valid, correctly-signed JSON file) from
the outside. That distinction matters here specifically because
consolidation is a multi-step process (read raw events -> compute new
entry -> merge into current library snapshot -> write) with real work
between "we decided to commit" and "the file changed on disk" -- exactly
the same "uncertain window" ToolNode(side_effecting=True) closes for an
arbitrary side-effecting function, per checkpoint.py's own docstring.

This module reuses that exact three-step contract, hand-driven (this
service isn't a Lar graph/ToolNode, so there's no executor auto-driving
it, but the primitives -- ActionMarker, confirm_action,
find_unconfirmed_markers -- are the same ones ToolNode uses internally):

  1. write an ActionMarker(status=ATTEMPTING) BEFORE the library file write
  2. perform the library file write (FileCheckpointStore.save, atomic)
  3. write ActionMarker(status=COMPLETED) (via confirm_action) AFTER

If the process dies between (1) and (3), ``find_unconfirmed_markers``
finds the ATTEMPTING marker on the NEXT store instantiation (e.g. after a
restart) and ``DMNStore.get_library`` refuses to serve until an operator
resolves it via ``resolve_unconfirmed_commit`` -- mirroring Lar's own
"no automatic path past UnconfirmedActionError" rule exactly.
"""
from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional

from lar_import import (
    ActionMarker,
    Checkpoint,
    CheckpointIntegrityError,
    FileCheckpointStore,
    confirm_action,
)

LIBRARY_CASE_ID = "consolidated_library"
LIBRARY_COMMIT_NODE_ID = "commit_consolidated_library"


def _events_case_id(class_name: str) -> str:
    # FileCheckpointStore._path just does f"{case_id}.json" under `directory`
    # -- sanitize class_name minimally so it can't escape the directory or
    # collide with the library's own case_id.
    safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in class_name)
    return f"events__{safe}"


class LibraryUnconfirmedCommitError(Exception):
    """
    Raised by ``DMNStore.get_library`` when one or more consolidated-library
    commits were left in ATTEMPTING status by a crash and have not yet been
    resolved via ``resolve_unconfirmed_commit``. This is the store-level
    surface of Lar's ``UnconfirmedActionError`` contract: the service will
    NOT silently guess whether the interrupted commit landed -- it refuses
    to serve the library until a human/operator resolves it, exactly the
    same "detectable, not auto-handled" posture ``checkpoint.py`` documents
    for ``ToolNode(side_effecting=True)``.
    """

    def __init__(self, markers: List[ActionMarker]):
        self.markers = markers
        steps = [m.step for m in markers]
        super().__init__(
            f"Consolidated library has {len(markers)} unresolved commit(s) "
            f"(step(s) {steps}) -- a previous consolidation commit was "
            f"attempting to write when the process stopped, and it was never "
            f"confirmed complete or failed. Refusing to serve the library "
            f"until this is resolved. Call DMNStore.resolve_unconfirmed_commit(...) "
            f"(or POST /admin/resolve_commit) after determining out-of-band "
            f"whether the write actually completed."
        )


class DMNStore:
    def __init__(self, directory: str, hmac_secret: Optional[str] = None):
        self.checkpoint_store = FileCheckpointStore(directory=directory, hmac_secret=hmac_secret)
        # FileCheckpointStore documents itself as unsafe for concurrent
        # writers to the SAME case_id. This service can receive concurrent
        # HTTP requests (uvicorn can run handlers concurrently), so a single
        # process-wide lock serializes all writes -- events across different
        # classes included, for simplicity; this is a single-process,
        # single-instance service exactly like FileCheckpointStore's own
        # documented scope ("not a distributed store, has no locking").
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- events

    def append_event(self, event: Dict[str, Any]) -> int:
        """
        Durably appends one raw fleet event to its class's signed event log.
        Returns the new total count of raw events stored for that class.
        """
        class_name = event["class_name"]
        case_id = _events_case_id(class_name)
        with self._lock:
            existing = self.checkpoint_store.load(case_id)
            events = list(existing.state["events"]) if existing else []
            events.append(event)
            cp = Checkpoint(
                case_id=case_id,
                resume_node_id="dmn_event_log",  # not used for graph resume; just a label
                state={"class_name": class_name, "events": events},
                reason="fleet event ingest",
            )
            self.checkpoint_store.save(cp)
            return len(events)

    def get_events(self, class_name: str) -> List[Dict[str, Any]]:
        cp = self.checkpoint_store.load(_events_case_id(class_name))
        return list(cp.state["events"]) if cp else []

    # ------------------------------------------------------- library commit

    def _load_library_state(self) -> Dict[str, Any]:
        cp = self.checkpoint_store.load(LIBRARY_CASE_ID)
        if cp:
            return dict(cp.state)
        return {"classes": {}, "commit_counter": 0}

    def commit_library_entry(
        self,
        class_name: str,
        entry: Dict[str, Any],
        _crash_after_marker: bool = False,
        _test_delay_before_write: float = 0.0,
        _on_marker_written=None,
    ) -> ActionMarker:
        """
        The ONE write path this service protects with Lar's ActionMarker
        pattern (see module docstring). Three durable steps:

          1. ATTEMPTING marker written (durable, signed if hmac_secret set)
          2. the actual library file write (atomic, via FileCheckpointStore)
          3. COMPLETED marker written via confirm_action

        ``_crash_after_marker`` is a TEST-ONLY hook: when True, this method
        writes the ATTEMPTING marker (step 1) and then raises immediately,
        never reaching steps 2/3 -- a fast in-process simulation of "process
        died right after step 1".

        ``_test_delay_before_write`` is a TEST-ONLY hook: sleeps for this
        many seconds AFTER the ATTEMPTING marker is durably written but
        BEFORE the actual library file write -- opens a real wall-clock
        window during which an external test harness can send a genuine
        ``SIGKILL`` to this process and land exactly between "marker
        recorded" and "file written", which is the real scenario
        requirement 4 asks to be tested (see
        ``tests/_crash_worker.py`` + ``tests/test_crash_recovery.py``).
        Zero by default -- no behavior change for the real service.

        ``_on_marker_written`` is a TEST-ONLY hook: an optional zero-arg
        callable invoked synchronously immediately after step 1 durably
        completes (marker written and flushed to disk), before the delay/
        crash hooks run -- lets an external test process synchronize on
        "the marker is now really on disk" (e.g. by printing a line the
        parent test's subprocess reader waits for) instead of guessing via
        a fixed sleep. ``None`` by default -- no behavior change.
        """
        with self._lock:
            state = self._load_library_state()
            step = int(state.get("commit_counter", 0)) + 1

            marker = ActionMarker(
                case_id=LIBRARY_CASE_ID,
                node_id=LIBRARY_COMMIT_NODE_ID,
                step=step,
                status=ActionMarker.ATTEMPTING,
            )
            self.checkpoint_store.save_marker(marker)

            if _on_marker_written is not None:
                _on_marker_written()

            if _crash_after_marker:
                raise RuntimeError("simulated crash immediately after ATTEMPTING marker")

            if _test_delay_before_write:
                import time
                time.sleep(_test_delay_before_write)

            new_state = dict(state)
            new_state["classes"] = dict(state.get("classes", {}))
            new_state["classes"][class_name] = entry
            new_state["commit_counter"] = step
            cp = Checkpoint(
                case_id=LIBRARY_CASE_ID,
                resume_node_id="dmn_library",
                state=new_state,
                reason=f"consolidate:{class_name}",
                metadata={"step": step, "class_name": class_name},
            )
            self.checkpoint_store.save(cp)

            confirm_action(
                self.checkpoint_store,
                case_id=LIBRARY_CASE_ID,
                node_id=LIBRARY_COMMIT_NODE_ID,
                step=step,
                happened=True,
                result={"class_name": class_name, "step": step},
            )
            return marker

    def find_unconfirmed_library_commits(self) -> List[ActionMarker]:
        return self.checkpoint_store.find_unconfirmed_markers(LIBRARY_CASE_ID)

    def resolve_unconfirmed_commit(self, step: int, happened: bool) -> ActionMarker:
        """
        The one sanctioned way past an unresolved ATTEMPTING marker --
        thin wrapper over ``lar.checkpoint.confirm_action``. ``happened``
        must be determined out-of-band (e.g. by inspecting whether
        ``consolidated_library.json``'s ``classes[class_name]`` already
        reflects the attempted commit's entry) -- this service does not
        guess.
        """
        return confirm_action(
            self.checkpoint_store,
            case_id=LIBRARY_CASE_ID,
            node_id=LIBRARY_COMMIT_NODE_ID,
            step=step,
            happened=happened,
        )

    def get_library(self, allow_unconfirmed: bool = False) -> Dict[str, Any]:
        unconfirmed = self.find_unconfirmed_library_commits()
        if unconfirmed and not allow_unconfirmed:
            raise LibraryUnconfirmedCommitError(unconfirmed)
        return self._load_library_state()

    def get_library_entry(self, class_name: str) -> Optional[Dict[str, Any]]:
        state = self.get_library()
        return state.get("classes", {}).get(class_name)
