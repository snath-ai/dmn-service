"""
Delta-space consolidation math -- the actual formalism from the LTL paper
(``academic_papers/snath_core/03_LTL/main.tex``, section
"Delta-Space Adapter Library", definitions around lines 897-981).

What the paper defines, and what we implement here 1:1
--------------------------------------------------------------------------
Definition (Delta-Space and Divergence Vector):
    Delta_base(x_I, x_T) = phi_I^0(x_I) - phi_T^0(x_T)   in R^d

    In this service we do not recompute this -- a fleet unit computes its
    own divergence/concept vector locally (exactly like
    ``fleet_clinical_pipeline.py``'s ``anchorA_pneumonia`` or
    ``ood_guard.py``'s ``z_proprio`` D-hard events) and submits only the
    compact vector. That is the whole point of the central-DMN pattern:
    raw data never crosses the wire, only the already-reduced anchor.

Definition (Adapter Library):
    Given K clusters in Delta-space with centroids {Delta_bar_1 .. Delta_bar_K},
    the library is L_phi = {(Delta_bar_c, phi_c^A, phi_c^B)}_{c=1}^K.
    At inference: c* = argmax_c cos(Delta_base, Delta_bar_c); if
    cos(Delta_base, Delta_bar_c*) >= tau_sim, the adapters for c* are used.

    In our setting, the "cluster" a raw event belongs to is already named
    by the submitting unit (``class_name`` / ``failure_class``, exactly the
    field every one of today's real artifacts already carries --
    ``fleet_event_signed.json``'s ``class_name: "pneumonia_pattern"``,
    ``environmental_transient.json``'s ``failure_class``). So consolidation
    here is: for a fixed class label c, take every raw anchor vector
    submitted under it and reduce it to ONE centroid, per the paper's own
    "centroid of the cluster" definition -- the arithmetic mean:

        Delta_bar_c = (1/n) * sum_i Delta_i          <- paper's centroid

    This is the exact operation ``fleet_clinical_pipeline.py`` already
    performs by hand (``anchorA_pneumonia = vA_hardcases.mean(axis=0)``)
    and the exact operation ``ood_guard.py`` already performs to produce
    ``adapters_b/environmental_transient.json``'s ``centroid_proprio`` --
    we are not inventing a new clustering scheme, just giving the same
    per-class mean a durable, signed, servable home.

Remark (Temporal Decay / Synaptic Depression):
    W(delta_t) = exp(-lambda * delta_t); load iff W >= W_min (W_min = 0.40
    in the paper's robotics instantiation, Table in ``main.tex`` around
    line 950-981). Failure-class-specific lambda from that same table:
        environmental_transient  lambda=0.50  (half-life 1.4 yr)
        sensor_drift              lambda=0.20  (half-life 3.5 yr)
        hardware_structural       lambda=0.02  (half-life 35 yr)
    This service stores ``consolidated_at`` and a ``decay_lambda`` (looked
    up from this table when the failure class matches, else a documented
    default) alongside each library entry so a RECEIVING unit can compute
    its own W(delta_t) and apply W_min itself -- this service does not
    decide trust (see the "boundary" docstring in app.py); it only carries
    the ingredients the paper's formula needs.

What is an ENGINEERING CHOICE, not from the paper, and is documented as such
--------------------------------------------------------------------------
The paper specifies the *selection* rule (cosine similarity to centroid
>= tau_sim) but does not specify how tau_sim itself should be derived from
a real cluster's member statistics for a production system -- Table
(robotics regression suite) treats tau_sim as a fixed hyperparameter of the
router, not something re-derived per cluster centroid.

For a *servable per-class* tau_sim (so a unit that only has a centroid, not
the whole raw history, still gets a usable similarity floor), we derive one
from the cluster's own member dispersion, using the same
"mean - k*std" gating shape ``fleet_clinical_pipeline.py`` already uses for
its (differently scaled) L1 divergence threshold
(``tau_A = dA_normal_self.mean() + 2 * dA_normal_self.std()``) -- adapted to
cosine-similarity space, where higher is better, so it's a floor
(mean - k*std) rather than a ceiling (mean + k*std):

    tau_sim_c = clip(mean(cos_sim(Delta_i, Delta_bar_c)) - 2*std(...), 0, 1)

This is clearly marked as a derived recommendation, not the paper's own
formula -- a consuming unit can equally choose to ignore it and set its own
tau_sim (the paper's mechanism explicitly allows a fixed router-level
tau_sim; the per-cluster value here is a convenience default).
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Sequence

# Table from LTL main.tex, Delta-Space Adapter Library section, "Temporal
# Decay and Synaptic Depression" remark (robotics instantiation).
FAILURE_CLASS_DECAY_LAMBDA: Dict[str, float] = {
    "environmental_transient": 0.50,
    "sensor_drift": 0.20,
    "hardware_structural": 0.02,
}
DEFAULT_DECAY_LAMBDA = 0.20  # paper doesn't define a default; sensor_drift's
# value is the middle of the three named classes, used when a submitted
# class isn't one of the three named ones.
W_MIN = 0.40  # paper's injection gate (Sec. Delta-Space Adapter Library remark)

TAU_SIM_STD_MULTIPLIER = 2.0  # matches fleet_clinical_pipeline.py's own
# "mean +/- 2*std" gating convention (documented above)


def _mean_vector(vectors: Sequence[Sequence[float]]) -> List[float]:
    n = len(vectors)
    dim = len(vectors[0])
    return [sum(v[i] for v in vectors) / n for i in range(dim)]


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = sum(values) / len(values)
    var = sum((v - m) ** 2 for v in values) / len(values)
    return math.sqrt(var)


def consolidate_class(
    class_name: str,
    events: List[Dict[str, Any]],
    failure_class: str | None = None,
) -> Dict[str, Any]:
    """
    Reduce every raw event submitted under ``class_name`` to one library
    entry: centroid (Delta_bar_c, the paper's cluster centroid = mean of
    members), a derived per-cluster cosine-similarity floor (tau_sim, see
    module docstring for why this is an engineering derivation, not the
    paper's own formula), and the temporal-decay ingredients (decay_lambda,
    consolidated_at) a receiving unit needs to compute W(delta_t) itself.

    ``events`` must be a non-empty list of dicts each containing at least
    ``concept_anchor`` (the raw Delta_i vector) -- all vectors must share
    the same dimensionality.
    """
    if not events:
        raise ValueError(f"cannot consolidate class {class_name!r} with zero events")

    anchors = [e["concept_anchor"] for e in events]
    dim = len(anchors[0])
    if any(len(a) != dim for a in anchors):
        raise ValueError(
            f"class {class_name!r} has anchors of inconsistent dimensionality "
            f"-- cannot average vectors from different concept spaces"
        )

    centroid = _mean_vector(anchors)
    member_sims = [_cosine_similarity(a, centroid) for a in anchors]
    tau_sim = max(0.0, min(1.0, (sum(member_sims) / len(member_sims))
                            - TAU_SIM_STD_MULTIPLIER * _std(member_sims)))

    n_source_events_total = sum(int(e.get("n_source_events", 1)) for e in events)
    fc = failure_class or class_name
    decay_lambda = FAILURE_CLASS_DECAY_LAMBDA.get(fc, DEFAULT_DECAY_LAMBDA)

    return {
        "class_name": class_name,
        "failure_class": fc,
        "centroid": centroid,
        "dim": dim,
        "tau_sim": round(tau_sim, 6),
        "member_cosine_similarities": [round(s, 6) for s in member_sims],
        "n_events": len(events),
        "n_source_events_total": n_source_events_total,
        "decay_lambda": decay_lambda,
        "w_min": W_MIN,
        "consolidated_at": datetime.now(timezone.utc).isoformat(),
        "contributing_units": sorted({e["unit_id"] for e in events if "unit_id" in e}),
    }


def trust_weight(consolidated_at_iso: str, decay_lambda: float, now: datetime | None = None) -> float:
    """
    W(delta_t) = exp(-lambda * delta_t), delta_t in years -- the paper's own
    temporal-decay formula. Exposed as a convenience for callers/tests; the
    HTTP service does not use this to gate anything itself (see the
    boundary note in app.py) -- it is here so a receiving unit's local
    client code (or our own demo) doesn't have to hand-roll the formula.
    """
    consolidated_at = datetime.fromisoformat(consolidated_at_iso)
    if consolidated_at.tzinfo is None:
        consolidated_at = consolidated_at.replace(tzinfo=timezone.utc)
    now = now or datetime.now(timezone.utc)
    delta_t_years = (now - consolidated_at).total_seconds() / (365.25 * 24 * 3600)
    return math.exp(-decay_lambda * max(0.0, delta_t_years))
