"""
End-to-end demo of the central DMN service, run over real HTTP against a
real uvicorn subprocess, using real anchor vectors from real prior work
(see real_data.py). Run:  python3 demo.py

Demonstrates, in order:
  1. Two real fleet units submitting real events (robotics environmental_transient
     D-hard events from two different seeds' robot_a runs).
  2. Consolidation actually happening once enough events accumulate.
  3. A third unit (never submitted anything) pulling the consolidated
     library and getting the benefit.
  4. A class with too few events staying unconsolidated (honest 404) --
     the pneumonia_pattern clinical event, submitted once.
  5. Signed tamper-detection: corrupt a stored file on disk, show the
     service refuses to serve it instead of silently returning bad data.

The crash-mid-commit ActionMarker test (requirement 4) is in
tests/test_crash_recovery.py -- it needs a real `kill -9` of a child
process and is run via pytest, not folded into this script.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import httpx  # noqa: E402

from consolidation import trust_weight  # noqa: E402
from real_data import load_clinical_event, load_ground_truth_centroid, load_robotics_events  # noqa: E402

PORT = 8799
BASE_URL = f"http://127.0.0.1:{PORT}"
STORAGE_DIR = HERE / "_demo_run" / "storage"
HMAC_SECRET = "demo-fleet-dmn-secret-do-not-use-in-prod"


def start_server() -> subprocess.Popen:
    if STORAGE_DIR.exists():
        shutil.rmtree(STORAGE_DIR)
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["DMN_STORAGE_DIR"] = str(STORAGE_DIR)
    env["DMN_HMAC_SECRET"] = HMAC_SECRET
    env["DMN_MIN_EVENTS_FOR_CONSOLIDATION"] = "3"
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app:app", "--port", str(PORT), "--host", "127.0.0.1"],
        cwd=str(HERE),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc


def wait_for_health(client: httpx.Client, timeout: float = 20.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = client.get("/health")
            if r.status_code == 200:
                print(f"  server up: {r.json()}")
                return
        except httpx.ConnectError:
            pass
        time.sleep(0.3)
    raise RuntimeError("server never became healthy")


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main() -> None:
    proc = start_server()
    try:
        with httpx.Client(base_url=BASE_URL, timeout=10.0) as client:
            section("0. Waiting for real uvicorn server to start")
            wait_for_health(client)

            section("1. TWO REAL FLEET UNITS SUBMIT REAL EVENTS (robotics, environmental_transient)")
            events_seed13 = load_robotics_events(seed=13, unit_id="robot_a_seed13", n=4)
            events_seed42 = load_robotics_events(seed=42, unit_id="robot_a_seed42", n=4)
            print(f"Loaded {len(events_seed13)} real events from ood_seed_13/robot_a_dhard.jsonl")
            print(f"Loaded {len(events_seed42)} real events from ood_seed_42/robot_a_dhard.jsonl")
            print(f"Sample real anchor (seed13, event0) z_proprio: "
                  f"{[round(v, 4) for v in events_seed13[0]['concept_anchor']]}")

            for ev in events_seed13 + events_seed42:
                r = client.post("/events", json=ev)
                r.raise_for_status()
                ack = r.json()
                print(f"  POST /events unit={ev['unit_id']:<16} class={ev['class_name']:<22} "
                      f"-> stored={ack['stored_events_for_class']:>2} "
                      f"consolidated={ack['consolidation_triggered']}")

            section("2. CONSOLIDATION ACTUALLY HAPPENED")
            r = client.get("/library/environmental_transient")
            r.raise_for_status()
            entry = r.json()
            print(json.dumps(entry, indent=2))

            gt = load_ground_truth_centroid(seed=42)
            import math
            our_c = entry["centroid"]
            gt_c = gt["centroid_proprio"]
            dot = sum(a * b for a, b in zip(our_c, gt_c))
            na = math.sqrt(sum(a * a for a in our_c))
            nb = math.sqrt(sum(b * b for b in gt_c))
            cos = dot / (na * nb)
            print(f"\nSanity check: cosine similarity between OUR consolidated centroid "
                  f"(from {entry['n_events']} sampled events) and ood_guard.py's OWN "
                  f"independently-computed centroid_proprio for seed 42 "
                  f"(from its full 30-event run): {cos:.4f}")
            print("(Not expected to be ~1.0 -- we used a small sample from two seeds; "
                  "this just confirms the consolidation math points the same direction "
                  "as ood_guard.py's own real, independently-derived centroid.)")

            section("3. A THIRD, NEVER-SUBMITTED UNIT PULLS THE LIBRARY AND BENEFITS")
            r = client.get("/library/environmental_transient")
            r.raise_for_status()
            pulled = r.json()
            w = trust_weight(pulled["consolidated_at"], pulled["decay_lambda"])
            print(f"  robot_c_seed7 (has submitted ZERO events) -> GET /library/environmental_transient")
            print(f"  received centroid (8 real numbers, no raw data): "
                  f"{[round(v, 4) for v in pulled['centroid']]}")
            print(f"  tau_sim (derived cosine-similarity floor): {pulled['tau_sim']}")
            print(f"  decay_lambda: {pulled['decay_lambda']}  w_min: {pulled['w_min']}  "
                  f"current trust weight W(dt): {w:.6f}  (fresh consolidation -> ~1.0, as expected)")
            print(f"  contributing_units: {pulled['contributing_units']} "
                  f"-- robot_c never contributed, purely receives the benefit")

            section("4. A CLASS WITH TOO FEW EVENTS STAYS HONESTLY UNCONSOLIDATED")
            clinical_event = load_clinical_event()
            print(f"Loaded 1 real event from Fleet_Learning/fleet_event_signed.json "
                  f"(class={clinical_event['class_name']}, "
                  f"n_source_events={clinical_event['n_source_events']})")
            r = client.post("/events", json=clinical_event)
            r.raise_for_status()
            print(f"  POST /events -> {r.json()}")
            r = client.get("/library/pneumonia_pattern")
            print(f"  GET /library/pneumonia_pattern -> HTTP {r.status_code}: {r.json()['detail']}")
            assert r.status_code == 404

            section("5. SIGNED TAMPER-DETECTION (real corrupt-and-detect)")
            events_file = STORAGE_DIR / "events__environmental_transient.json"
            with open(events_file) as f:
                raw = f.read()
            print(f"  Loaded real on-disk signed file: {events_file.name} ({len(raw)} bytes)")
            corrupted = raw.replace('"n_source_events": 1', '"n_source_events": 999', 1)
            assert corrupted != raw, "corruption did not change file content -- pattern not found"
            with open(events_file, "w") as f:
                f.write(corrupted)
            print("  Flipped one value on disk (n_source_events: 1 -> 999) -- "
                  "still syntactically valid JSON, signature now stale.")
            r = client.post("/consolidate/environmental_transient")
            print(f"  POST /consolidate/environmental_transient (forces a re-read of the "
                  f"tampered file) -> HTTP {r.status_code}")
            print(f"  {json.dumps(r.json(), indent=2)}")
            assert r.status_code == 500 and r.json()["error"] == "CheckpointIntegrityError"
            print("\n  CONFIRMED: tampered signed file was refused, not silently served.")

        print("\n" + "=" * 78)
        print("DEMO COMPLETE -- all 5 sections above ran against a real HTTP server,")
        print("real subprocess, real files on disk, real anchor vectors.")
        print("=" * 78)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        out, _ = proc.communicate()
        server_log = HERE / "_demo_run" / "server.log"
        server_log.parent.mkdir(parents=True, exist_ok=True)
        server_log.write_text(out or "")
        print(f"\n(full server stdout/stderr saved to {server_log})")


if __name__ == "__main__":
    main()
