"""
Requirement 1: every stored event is genuinely signed and tamper-detectable
-- verified with a real corrupt-and-detect test, same standard as today's
other Lar work (mirrors ``Lar_Main/lar/tests/unit/test_checkpoint_integrity.py``'s
own approach: write via the real store, flip a byte on disk, reload via the
real store, assert ``CheckpointIntegrityError``).

Covers BOTH signed artifacts this service produces:
  - the per-class raw event log (``events__<class>.json``)
  - the consolidated library (``consolidated_library.json``)
"""
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
DMN_SERVICE_DIR = HERE.parent
sys.path.insert(0, str(DMN_SERVICE_DIR))

from consolidation import consolidate_class  # noqa: E402
from lar_import import CheckpointIntegrityError  # noqa: E402
from real_data import load_robotics_events  # noqa: E402
from storage import DMNStore  # noqa: E402

HMAC_SECRET = "tamper-test-secret"
CLASS_NAME = "environmental_transient"


@pytest.fixture
def store(tmp_path):
    return DMNStore(directory=str(tmp_path / "dmn_storage"), hmac_secret=HMAC_SECRET)


def test_event_log_is_actually_signed_not_just_labeled(store, tmp_path):
    events = load_robotics_events(seed=13, unit_id="robot_a_seed13", n=3)
    for ev in events:
        store.append_event(ev)

    path = tmp_path / "dmn_storage" / f"events__{CLASS_NAME}.json"
    assert path.exists()
    with open(path) as f:
        on_disk = json.load(f)
    assert "signature" in on_disk and on_disk["signature"], (
        "event log has no real 'signature' field -- would repeat the exact "
        "fleet_event_signed.json gap (labeled signed, actually plain json.dump)"
    )

    # Reading back through the real store with the correct secret works fine.
    assert len(store.get_events(CLASS_NAME)) == 3

    # Now corrupt ONE value on disk -- still syntactically valid JSON.
    raw = path.read_text()
    corrupted = raw.replace('"n_source_events": 1', '"n_source_events": 42', 1)
    assert corrupted != raw, "test setup bug: pattern not found to corrupt"
    path.write_text(corrupted)
    json.loads(corrupted)  # still parses -- this is NOT a torn-write scenario

    with pytest.raises(CheckpointIntegrityError) as exc_info:
        store.get_events(CLASS_NAME)
    print(f"\n[test] tampered event log correctly rejected: {exc_info.value}")


def test_consolidated_library_is_actually_signed_not_just_labeled(store, tmp_path):
    events = load_robotics_events(seed=42, unit_id="robot_a_seed42", n=4)
    for ev in events:
        store.append_event(ev)
    entry = consolidate_class(CLASS_NAME, events, failure_class=CLASS_NAME)
    store.commit_library_entry(CLASS_NAME, entry)

    path = tmp_path / "dmn_storage" / "consolidated_library.json"
    assert path.exists()
    with open(path) as f:
        on_disk = json.load(f)
    assert "signature" in on_disk and on_disk["signature"]

    assert store.get_library()["classes"][CLASS_NAME]["centroid"] == entry["centroid"]

    # Flip one number inside the consolidated centroid on disk.
    raw = path.read_text()
    original_val = str(entry["centroid"][0])
    corrupted = raw.replace(original_val, str(entry["centroid"][0] + 999.0), 1)
    assert corrupted != raw, "test setup bug: pattern not found to corrupt"
    path.write_text(corrupted)
    json.loads(corrupted)

    with pytest.raises(CheckpointIntegrityError) as exc_info:
        store.get_library()
    print(f"\n[test] tampered consolidated library correctly rejected: {exc_info.value}")


def test_no_secret_configured_means_no_signature_at_all(tmp_path):
    """
    Sanity check on the opt-in contract itself (documented in Lar's own
    checkpoint.py): a store with no hmac_secret writes no signature and
    verifies nothing -- this is Lar's existing, tested default behavior,
    checked here only to confirm this service didn't accidentally rely on
    signing being unconditionally on.
    """
    unsigned_store = DMNStore(directory=str(tmp_path / "unsigned"), hmac_secret=None)
    events = load_robotics_events(seed=7, unit_id="robot_a_seed7", n=2)
    for ev in events:
        unsigned_store.append_event(ev)
    path = tmp_path / "unsigned" / f"events__{CLASS_NAME}.json"
    with open(path) as f:
        on_disk = json.load(f)
    assert "signature" not in on_disk
