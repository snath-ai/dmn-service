"""
Requirement 4: kill the process mid-consolidation-commit for real (a
genuine SIGKILL of a genuine child process, not an in-process simulation),
then confirm the service detects the unconfirmed state on restart rather
than silently serving a half-written library.

Uses Lar's own ``ActionMarker`` / ``find_unconfirmed_markers`` /
``confirm_action`` primitives throughout (via ``storage.DMNStore``) --
nothing here reimplements uncertain-action tracking.
"""
import signal
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
DMN_SERVICE_DIR = HERE.parent
sys.path.insert(0, str(DMN_SERVICE_DIR))

from lar_import import ActionMarker  # noqa: E402
from storage import DMNStore, LibraryUnconfirmedCommitError  # noqa: E402

CLASS_NAME = "environmental_transient"
HMAC_SECRET = "crash-test-secret"


def test_kill_mid_commit_leaves_detectable_unconfirmed_marker(tmp_path):
    storage_dir = tmp_path / "dmn_storage"

    worker = HERE / "_crash_worker.py"
    proc = subprocess.Popen(
        [sys.executable, str(worker), str(storage_dir), HMAC_SECRET, CLASS_NAME, "5"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # Wait for the worker's own production-code test hook to confirm the
    # ATTEMPTING marker is durably on disk, then kill for real.
    saw_marker_line = False
    for _ in range(200):  # up to ~20s at 0.1s poll via readline blocking below
        line = proc.stdout.readline()
        if not line:
            break
        line = line.strip()
        if line == "ATTEMPTING_MARKER_WRITTEN":
            saw_marker_line = True
            break
        if line == "COMMIT_COMPLETED_WITHOUT_BEING_KILLED":
            pytest.fail("worker completed the commit before we could kill it -- "
                        "increase the delay in this test")

    assert saw_marker_line, "never saw ATTEMPTING_MARKER_WRITTEN from the worker process"

    # The marker is now durably on disk (the worker printed the sync line
    # from inside commit_library_entry, right after checkpoint_store.save_marker
    # returned) but the worker is still sleeping before it writes the actual
    # library file. Kill it NOW, for real.
    proc.send_signal(signal.SIGKILL)
    proc.wait(timeout=10)
    assert proc.returncode != 0  # killed, not a clean exit

    # ---- "process restart": brand-new DMNStore instance, same directory ----
    store = DMNStore(directory=str(storage_dir), hmac_secret=HMAC_SECRET)

    unconfirmed = store.find_unconfirmed_library_commits()
    assert len(unconfirmed) == 1, (
        f"expected exactly one unresolved ATTEMPTING marker after the kill, got {unconfirmed}"
    )
    marker = unconfirmed[0]
    assert marker.status == ActionMarker.ATTEMPTING
    assert marker.node_id == "commit_consolidated_library"
    print(f"\n[test] confirmed unconfirmed marker after real SIGKILL: {marker!r}")

    # The service must refuse to silently serve the library while this is unresolved.
    with pytest.raises(LibraryUnconfirmedCommitError) as exc_info:
        store.get_library()
    print(f"[test] get_library() correctly refused: {exc_info.value}")

    # And the library file itself must NOT contain the half-committed class
    # (the worker was killed before FileCheckpointStore.save() for the
    # library ever ran -- so there is nothing partial, only "unconfirmed").
    raw_state = store._load_library_state()
    assert CLASS_NAME not in raw_state.get("classes", {}), (
        "the class should not be visible at all -- the write never happened, "
        "only the marker did; a class appearing here would mean a torn/partial "
        "write, which FileCheckpointStore's atomicity should make impossible"
    )

    # ---- operator resolves it out-of-band: confirmed the write never landed ----
    resolved = store.resolve_unconfirmed_commit(step=marker.step, happened=False)
    assert resolved.status == ActionMarker.FAILED
    print(f"[test] resolved via confirm_action(happened=False): {resolved!r}")

    # Now the library serves again, cleanly, still without the never-committed class.
    state = store.get_library()
    assert CLASS_NAME not in state.get("classes", {})
    print(f"[test] get_library() now serves cleanly: {state}")
