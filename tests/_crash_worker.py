"""
Standalone worker process for test_crash_recovery.py -- NOT a pytest file.

Invoked as: python3 _crash_worker.py <storage_dir> <hmac_secret> <class_name> <delay_seconds>

Appends a few real robotics events, builds a real consolidated entry, then
calls DMNStore.commit_library_entry(..., _test_delay_before_write=delay,
_on_marker_written=<print+flush a sync line>).

The sync line ("ATTEMPTING_MARKER_WRITTEN") is printed by the production
code's own test hook at the exact instant the ATTEMPTING ActionMarker has
been durably written to disk (step 1 of the real commit protocol) -- the
parent test process reads stdout line-by-line and sends SIGKILL the moment
it sees that line, so the kill is guaranteed to land AFTER the marker is
durably recorded and BEFORE the library file itself is written (the
``_test_delay_before_write`` sleep, still inside the real
``commit_library_entry`` call, is what keeps that window open long enough
for the parent to act).
"""
import sys
from pathlib import Path

DMN_SERVICE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DMN_SERVICE_DIR))

from consolidation import consolidate_class  # noqa: E402
from real_data import load_robotics_events  # noqa: E402
from storage import DMNStore  # noqa: E402


def main() -> None:
    storage_dir, hmac_secret, class_name, delay_seconds = sys.argv[1:5]
    delay_seconds = float(delay_seconds)

    store = DMNStore(directory=storage_dir, hmac_secret=hmac_secret)

    events = load_robotics_events(seed=13, unit_id="robot_a_seed13", n=3)
    for ev in events:
        store.append_event(ev)

    entry = consolidate_class(class_name, events, failure_class=class_name)

    def _sync() -> None:
        print("ATTEMPTING_MARKER_WRITTEN", flush=True)

    store.commit_library_entry(
        class_name,
        entry,
        _test_delay_before_write=delay_seconds,
        _on_marker_written=_sync,
    )
    # Reached only if NOT killed in time -- lets the parent test fail with
    # a clear message instead of a confusing timeout if timing is off.
    print("COMMIT_COMPLETED_WITHOUT_BEING_KILLED", flush=True)


if __name__ == "__main__":
    main()
