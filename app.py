"""
Central DMN service -- a real, running HTTP service a fleet of units can
connect to, built on Lar's checkpoint/audit infrastructure as its durable,
signed backbone.

WHAT THIS SERVICE IS
--------------------------------------------------------------------------
A durable, signed, honest STORE-AND-FORWARD for the Delta-space adapter
library defined in the LTL paper (Section "Delta-Space Adapter Library",
``academic_papers/snath_core/03_LTL/main.tex``): fleet units submit compact
anchor events under a named class; once enough events accumulate for a
class, this service consolidates them into a single named centroid
(``consolidation.consolidate_class`` -- the paper's Delta_bar_c = mean of
cluster members) and serves the resulting library back to any unit that
asks. One unit's hard-won correction becomes available to every other unit
without a single byte of raw data ever leaving the unit that originated it
-- only the already-reduced anchor vector is ever stored or served, same
"19-number event, never raw data" pattern as the FLEET paper and
``fleet_clinical_pipeline.py``.

WHAT THIS SERVICE DELIBERATELY IS **NOT** -- READ BEFORE WIRING A CLIENT
--------------------------------------------------------------------------
This service does NOT verify whether a submitted or consolidated anchor is
GOOD -- i.e. it does not check whether trusting it would actually help, or
whether it's a bad/confounded anchor (e.g. equipment drift misclassified as
a novel pattern, the exact failure mode ``fleet_clinical_pipeline.py``
Test 2 demonstrates). It CANNOT do that job: verification requires testing
a candidate anchor against real local data, and this service, by design,
never has any raw local data from any unit -- only the compact anchors
units chose to submit. Attempting verification here would either be
meaningless (no data to verify against) or would require units to submit
raw data to a central point, which defeats the entire privacy/bandwidth
rationale of the architecture.

Verification-before-trust is, and must remain, a RECEIVING UNIT'S LOCAL
responsibility, run against that unit's own local data, before it acts on
anything pulled from GET /library. That is exactly the PERSIST-style gate
already demonstrated client-side in ``fleet_clinical_pipeline.py``
(``GATE_TOLERANCE`` / the "WITH verification gating" block): a receiving
site tests a transferred anchor against a small local held-out
known-normal validation set and rejects it if its false-fire rate exceeds
a tolerance, BEFORE adding it to its own trusted local library. Do that on
the client. This service's only jobs are: accept a submission durably and
signed, consolidate per the paper's own math, and serve the result honestly
-- including being honest when it CAN'T yet serve a trustworthy answer
(see the unconfirmed-commit 503 below), which is a durability/integrity
guarantee, not a content-quality guarantee.

LAR PRIMITIVES REUSED (NOT REBUILT)
--------------------------------------------------------------------------
  - ``lar.checkpoint.FileCheckpointStore(hmac_secret=...)``: durable,
    HMAC-signed storage for both the raw per-class event logs and the
    consolidated library. Every stored file is genuinely signed (see
    ``storage.py``); tampering is detected on load via
    ``CheckpointIntegrityError``, not merely labeled as signed the way
    ``fleet_event_signed.json`` was (a known, previously-flagged gap this
    service does not repeat).
  - ``lar.checkpoint.ActionMarker`` / ``confirm_action`` /
    ``find_unconfirmed_markers``: the uncertain-action protocol, applied to
    exactly one write path -- committing a consolidated update to the
    servable library (see ``storage.DMNStore.commit_library_entry`` for the
    full rationale). Every other write (raw event ingest) is plain,
    already-atomic ``FileCheckpointStore.save`` -- it doesn't need the
    extra layer because there's no "servable to everyone" visibility
    concern for an individual unit's own append.
"""
from __future__ import annotations

import os
from typing import Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from consolidation import consolidate_class
from lar_import import CheckpointIntegrityError
from models import (
    ConsolidateResult,
    EventAck,
    EventIn,
    HealthOut,
    LibraryEntryOut,
    LibraryOut,
    ResolveCommitRequest,
)
from storage import DMNStore, LibraryUnconfirmedCommitError

# --------------------------------------------------------------------------- config

DMN_STORAGE_DIR = os.environ.get("DMN_STORAGE_DIR", "dmn_checkpoints")
DMN_HMAC_SECRET = os.environ.get("DMN_HMAC_SECRET")  # None => signing disabled (documented Lar default)
MIN_EVENTS_FOR_CONSOLIDATION = int(os.environ.get("DMN_MIN_EVENTS_FOR_CONSOLIDATION", "3"))

if not DMN_HMAC_SECRET:
    raise RuntimeError(
        "DMN_HMAC_SECRET is not set. This service's entire tamper-detection "
        "guarantee (requirement 1: 'genuinely signed and tamper-detectable') "
        "depends on FileCheckpointStore(hmac_secret=...) being configured -- "
        "refusing to start unsigned rather than silently downgrading to an "
        "unsigned store."
    )

store = DMNStore(directory=DMN_STORAGE_DIR, hmac_secret=DMN_HMAC_SECRET)

app = FastAPI(
    title="DMN Service",
    description=(
        "Central Delta-space adapter-library consolidation service (LTL paper). "
        "Durable, signed store-and-forward -- NOT a verification/trust authority. "
        "See module docstring in app.py for the full boundary statement."
    ),
    version="0.1.0",
)


@app.exception_handler(CheckpointIntegrityError)
def _integrity_error_handler(request: Request, exc: CheckpointIntegrityError) -> JSONResponse:
    """
    Surfaces Lar's own tamper-detection exception as a clear HTTP error
    instead of a generic 500 -- this IS the requirement-1 guarantee made
    visible over HTTP: a corrupted/tampered signed file is refused, never
    silently served.
    """
    return JSONResponse(
        status_code=500,
        content={
            "error": "CheckpointIntegrityError",
            "detail": str(exc),
            "case_id": exc.case_id,
            "path": exc.path,
        },
    )


# ------------------------------------------------------------------------- helpers

def _entry_to_out(entry: Dict) -> LibraryEntryOut:
    return LibraryEntryOut(**entry)


def _maybe_consolidate(class_name: str, failure_class: str | None) -> ConsolidateResult:
    events = store.get_events(class_name)
    if len(events) < MIN_EVENTS_FOR_CONSOLIDATION:
        return ConsolidateResult(
            class_name=class_name,
            consolidated=False,
            reason=(
                f"only {len(events)}/{MIN_EVENTS_FOR_CONSOLIDATION} events "
                f"accumulated for this class -- consolidation trigger not met"
            ),
        )
    entry = consolidate_class(class_name, events, failure_class=failure_class)
    store.commit_library_entry(class_name, entry)
    return ConsolidateResult(class_name=class_name, consolidated=True, entry=_entry_to_out(entry))


# --------------------------------------------------------------------------- routes

@app.get("/health", response_model=HealthOut)
def health() -> HealthOut:
    unconfirmed = store.find_unconfirmed_library_commits()
    if unconfirmed:
        return HealthOut(
            status="degraded",
            unconfirmed_library_commits=len(unconfirmed),
            detail=(
                "One or more consolidated-library commits were left ATTEMPTING "
                "by a crash and are unresolved -- GET /library will refuse to "
                "serve until POST /admin/resolve_commit is called."
            ),
        )
    return HealthOut(status="ok", unconfirmed_library_commits=0)


@app.post("/events", response_model=EventAck)
def submit_event(event: EventIn) -> EventAck:
    """
    Durable event submission (requirement 1). Stored via
    ``FileCheckpointStore(hmac_secret=...)`` -- signed, tamper-detectable.
    Auto-triggers consolidation for this class once
    ``MIN_EVENTS_FOR_CONSOLIDATION`` raw events have accumulated
    (requirement 2's "reasonable trigger" -- see README note in
    ``consolidation.py``/this module for why: simplest trigger that still
    demonstrates real multi-unit consolidation without an arbitrary timer).
    """
    import datetime as _dt

    payload = event.model_dump()
    if not payload.get("timestamp"):
        payload["timestamp"] = _dt.datetime.now(_dt.timezone.utc).isoformat()

    n_events = store.append_event(payload)
    result = _maybe_consolidate(event.class_name, event.failure_class)

    return EventAck(
        class_name=event.class_name,
        stored_events_for_class=n_events,
        consolidation_triggered=result.consolidated,
        min_events_for_consolidation=MIN_EVENTS_FOR_CONSOLIDATION,
    )


@app.post("/consolidate/{class_name}", response_model=ConsolidateResult)
def consolidate_now(class_name: str, failure_class: str | None = None) -> ConsolidateResult:
    """Manual consolidation trigger (requirement 2's 'your call' alternative to the auto-trigger)."""
    events = store.get_events(class_name)
    if not events:
        raise HTTPException(status_code=404, detail=f"no events stored for class {class_name!r}")
    entry = consolidate_class(class_name, events, failure_class=failure_class)
    store.commit_library_entry(class_name, entry)
    return ConsolidateResult(class_name=class_name, consolidated=True, entry=_entry_to_out(entry))


@app.get("/library", response_model=LibraryOut)
def get_library() -> LibraryOut:
    """
    Requirement 3: library retrieval. Refuses (503) to serve if a prior
    consolidation commit was left unconfirmed by a crash -- see
    ``storage.LibraryUnconfirmedCommitError`` / requirement 4.
    """
    try:
        state = store.get_library()
    except LibraryUnconfirmedCommitError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    classes = {name: _entry_to_out(entry) for name, entry in state.get("classes", {}).items()}
    return LibraryOut(classes=classes, commit_counter=state.get("commit_counter", 0))


@app.get("/library/{class_name}", response_model=LibraryEntryOut)
def get_library_entry(class_name: str) -> LibraryEntryOut:
    try:
        entry = store.get_library_entry(class_name)
    except LibraryUnconfirmedCommitError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if entry is None:
        raise HTTPException(status_code=404, detail=f"class {class_name!r} not yet consolidated")
    return _entry_to_out(entry)


@app.post("/admin/resolve_commit")
def resolve_commit(req: ResolveCommitRequest):
    """
    The one sanctioned way past an unresolved ATTEMPTING library-commit
    marker (requirement 4) -- thin HTTP wrapper over
    ``lar.checkpoint.confirm_action`` via ``DMNStore.resolve_unconfirmed_commit``.
    ``happened`` must be determined out-of-band by the operator (e.g. by
    checking whether ``consolidated_library.json`` already contains the
    class entry the interrupted commit was writing) -- this endpoint does
    not guess, exactly matching Lar's own "no automatic path" rule.
    """
    marker = store.resolve_unconfirmed_commit(step=req.step, happened=req.happened)
    return JSONResponse({"resolved_step": marker.step, "status": marker.status})
