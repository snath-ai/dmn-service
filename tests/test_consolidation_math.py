"""
Checks ``consolidation.consolidate_class`` against the LTL paper's own
formula, computed independently here with plain Python (not by calling the
module under test twice), plus a sanity check against ood_guard.py's own
independently-computed real centroid for the same failure class.
"""
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DMN_SERVICE_DIR = HERE.parent
sys.path.insert(0, str(DMN_SERVICE_DIR))

from consolidation import (  # noqa: E402
    FAILURE_CLASS_DECAY_LAMBDA,
    W_MIN,
    consolidate_class,
    trust_weight,
)
from real_data import load_ground_truth_centroid, load_robotics_events  # noqa: E402


def _manual_mean(vectors):
    dim = len(vectors[0])
    return [sum(v[i] for v in vectors) / len(vectors) for i in range(dim)]


def _manual_cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb)


def test_centroid_is_the_papers_cluster_mean():
    events = load_robotics_events(seed=13, unit_id="robot_a_seed13", n=6)
    entry = consolidate_class("environmental_transient", events, failure_class="environmental_transient")

    expected_centroid = _manual_mean([e["concept_anchor"] for e in events])
    assert entry["centroid"] == expected_centroid, (
        "centroid must be the plain arithmetic mean of member anchors -- "
        "the LTL paper's Delta_bar_c definition"
    )
    assert entry["n_events"] == 6
    assert entry["dim"] == len(events[0]["concept_anchor"]) == 8


def test_tau_sim_is_mean_minus_2std_of_member_cosine_similarities_clipped():
    events = load_robotics_events(seed=42, unit_id="robot_a_seed42", n=5)
    entry = consolidate_class("environmental_transient", events, failure_class="environmental_transient")

    centroid = _manual_mean([e["concept_anchor"] for e in events])
    sims = [_manual_cosine(e["concept_anchor"], centroid) for e in events]
    mean_sim = sum(sims) / len(sims)
    var = sum((s - mean_sim) ** 2 for s in sims) / len(sims)
    std_sim = math.sqrt(var)
    expected_tau = max(0.0, min(1.0, mean_sim - 2.0 * std_sim))

    # entry["tau_sim"] is rounded to 6 dp for a clean servable payload
    assert abs(entry["tau_sim"] - expected_tau) < 1e-6


def test_decay_lambda_lookup_matches_ltl_paper_table():
    events = load_robotics_events(seed=7, unit_id="robot_a_seed7", n=3)
    entry = consolidate_class("environmental_transient", events, failure_class="environmental_transient")
    assert entry["decay_lambda"] == FAILURE_CLASS_DECAY_LAMBDA["environmental_transient"] == 0.50
    assert entry["w_min"] == W_MIN == 0.40


def test_decay_lambda_falls_back_for_unknown_failure_class():
    events = load_robotics_events(seed=99, unit_id="robot_a_seed99", n=3)
    entry = consolidate_class("some_new_pattern", events, failure_class=None)
    # falls back to class_name lookup, which also misses the table -> default
    assert entry["decay_lambda"] == 0.20


def test_trust_weight_matches_papers_exponential_decay_formula():
    from datetime import datetime, timedelta, timezone
    lam = 0.50  # environmental_transient; half-life = ln(2)/lambda = 1.386 yr (paper: "1.4 yr")

    # At exactly one half-life, W should be ~0.5 -- matches the paper's own
    # "half-life" framing of this constant directly.
    half_life_years = math.log(2) / lam
    consolidated_at = (datetime.now(timezone.utc) - timedelta(days=half_life_years * 365.25)).isoformat()
    w_half = trust_weight(consolidated_at, lam)
    assert abs(w_half - 0.5) < 1e-3

    # W_min=0.40 gate is crossed at t = -ln(W_min)/lambda ~= 1.83 yr for this
    # lambda -- one year (< that) should still be trusted, two years should not.
    t_1yr = (datetime.now(timezone.utc) - timedelta(days=365.25)).isoformat()
    t_2yr = (datetime.now(timezone.utc) - timedelta(days=2 * 365.25)).isoformat()
    w_1yr = trust_weight(t_1yr, lam)
    w_2yr = trust_weight(t_2yr, lam)
    assert abs(w_1yr - math.exp(-lam * 1.0)) < 1e-3
    assert w_1yr >= W_MIN, "1 year at lambda=0.50 should still be above W_min=0.40"
    assert w_2yr < W_MIN, "2 years at lambda=0.50 should have decayed below W_min=0.40"


def test_consolidation_points_same_direction_as_ood_guard_own_real_centroid():
    """
    Not a bit-exact match (we consolidate a small sample from two seeds;
    ood_guard.py's own environmental_transient.json was computed from a
    full 30-event single-seed run) -- but our from-scratch mean-of-members
    centroid should point in a similar direction in real Delta-space to
    ood_guard.py's own independently-computed real centroid for the same
    failure class and seed.
    """
    events = load_robotics_events(seed=42, unit_id="robot_a_seed42", n=8)
    entry = consolidate_class("environmental_transient", events, failure_class="environmental_transient")
    gt = load_ground_truth_centroid(seed=42)
    cos = _manual_cosine(entry["centroid"], gt["centroid_proprio"])
    assert cos > 0.8, f"expected our centroid to point roughly the same direction as ood_guard.py's own real centroid, got cosine={cos}"
