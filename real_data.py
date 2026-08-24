"""
Loads REAL anchor vectors from today's/this project's actual prior work --
no synthetic placeholder numbers anywhere in this module.

Sources:
  - Robotics: ``experiments/fleet/runs/ood_seed_{13,42}/robot_a_dhard.jsonl``
    -- real D-hard events written by ``ood_guard.py`` runs (today,
    2026-08-20), each a real 8-dim ``z_proprio`` embedding from an actual
    Walker2d-v5 rollout, HMAC-signed per-line (``hmac_hex``), with
    ``failure_class: "environmental_transient"`` -- one of the LTL paper's
    three named failure classes.
  - Clinical: ``Fleet_Learning/fleet_event_signed.json`` -- the real,
    already-consolidated 5-dim BiomedCLIP concept anchor for
    "pneumonia_pattern" (Site A, 20 source hard-case chest X-rays).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

FLEET_DIR = Path(
    "/Users/aadithya/Desktop/Private/personal/Snath/JEPA_Playground/"
    "Snath Robotics/experiments/fleet"
)
CLINICAL_FILE = Path(
    "/Users/aadithya/Desktop/Private/personal/Fleet_Learning/fleet_event_signed.json"
)


def load_robotics_events(seed: int, unit_id: str, n: int = 4) -> List[Dict[str, Any]]:
    """
    Real environmental_transient D-hard events from one seed's robot_a run.
    Uses z_proprio as the compact concept anchor -- ood_guard.py's own
    events record "winner": "proprio" for these, i.e. proprioception is the
    diagnostic stream for this failure class in the real run.
    """
    path = FLEET_DIR / "runs" / f"ood_seed_{seed}" / "robot_a_dhard.jsonl"
    events = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            raw = json.loads(line)
            events.append({
                "unit_id": unit_id,
                "class_name": raw["failure_class"],
                "concept_anchor": raw["z_proprio"],
                "n_source_events": 1,
                "failure_class": raw["failure_class"],
                "schema_labels": [f"proprio_{i}" for i in range(len(raw["z_proprio"]))],
            })
    return events


def load_clinical_event(unit_id: str = "site_A_clinical") -> Dict[str, Any]:
    with open(CLINICAL_FILE) as f:
        raw = json.load(f)
    return {
        "unit_id": unit_id,
        "class_name": raw["class_name"],
        "concept_anchor": raw["concept_anchor"],
        "n_source_events": raw["n_source_events"],
        "failure_class": None,  # not one of the robotics failure classes -> default decay lambda
        "schema_labels": raw["findings_schema"],
    }


def load_ground_truth_centroid(seed: int) -> Dict[str, Any]:
    """
    The centroid ood_guard.py itself already computed for this seed's full
    30-event run (adapters_b/environmental_transient.json) -- used ONLY as
    an external sanity check in the demo (cosine similarity between our
    from-scratch consolidation and this independently-computed real
    centroid), not consumed by the service itself.
    """
    path = FLEET_DIR / "runs" / f"ood_seed_{seed}" / "adapters_b" / "environmental_transient.json"
    with open(path) as f:
        return json.load(f)
