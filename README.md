# DMN Service

**Central Delta-Space Adapter Library consolidation service for the
Snath / Lár-JEPA fleet architecture.**

A durable, signed store-and-forward for `AbstractDivergenceRouter` events:
a fleet of independent units reports events, this service consolidates
them into a shared, trustable library the fleet reads back from.

Built on [Lár](https://github.com/snath-ai/lar)'s checkpoint layer and
[lar-dmn](https://github.com/snath-ai/DMN)'s memory-layer primitives · License: not yet set

---

## What it is not

Not a verification or trust authority. It does not check whether a
submitted or consolidated anchor is *good*, whether trusting it would
actually help, or whether it's confounded (see FLEET §5's class-capture
finding for exactly the failure mode this leaves open). Verification
against real local data is, by design, the receiving unit's job, not
this service's. What this service guarantees is durability, signing, and
honest refusal when it can't yet serve a trustworthy answer, none of
that is a content-quality guarantee.

## API

| Route | Purpose |
|---|---|
| `GET /health` | liveness check |
| `POST /events` | durably append one unit's signed concept-anchor event under a named class; auto-triggers consolidation once `MIN_EVENTS_FOR_CONSOLIDATION` (default 3) events have accumulated |
| `POST /consolidate/{class_name}` | trigger consolidation manually |
| `GET /library` | serve the full consolidated library |
| `GET /library/{class_name}` | serve one class's consolidated entry |
| `POST /admin/resolve_commit` | the one sanctioned recovery path after a crash mid-commit |

## Storage and crash recovery

Both the raw per-class event log and the servable library live in a single
`FileCheckpointStore`. Raw events append under
`events__<class_name>`, atomic at the filesystem level (temp file +
`os.replace`). The consolidated library lives under one case ID,
`consolidated_library`, wrapped in Lár's `ActionMarker` protocol: an
`ATTEMPTING` marker is written and flushed, the library file is rewritten
atomically, then a `COMPLETED` marker confirms it. If the process dies
between those steps, `find_unconfirmed_markers` surfaces the orphaned
marker on next open, and `GET /library` refuses to serve (HTTP 503, not a
guess) until an operator calls `POST /admin/resolve_commit` with an
out-of-band determination of whether the write actually landed. Verified
with a real `SIGKILL` sent mid-commit in `tests/test_crash_recovery.py`,
not a simulated exception.

Single-process, single-instance store, a process-wide `threading.Lock`
serializes writes since `FileCheckpointStore` documents itself as unsafe
for concurrent writers to the same case ID and uvicorn dispatches
concurrently. Not a distributed store, by the same scope
`FileCheckpointStore` itself documents.

## Consolidation math

For a class with submitted anchors $\{\Delta_i\}_{i=1}^n$: centroid is the
arithmetic mean $\bar\Delta_c = \frac{1}{n}\sum_i \Delta_i$, the LTL
paper's own definition, not a new scheme. A per-cluster similarity floor,
$\tau_{sim,c} = \text{clip}(\overline{\cos(\Delta_i, \bar\Delta_c)} -
2\cdot\text{std}(\cos(\Delta_i,\bar\Delta_c)), 0, 1)$, is an engineering
default a consuming unit is free to override, not something the LTL paper
specifies. Each entry also carries the temporal-decay ingredients LTL's
$W(\Delta t) = \exp(-\lambda \Delta t)$ formula needs, but this service
stores the ingredients and leaves evaluating $W(\Delta t)$ and enforcing
$W_{min}$ to the receiving unit; `trust_weight()` is exposed as a
convenience helper, not invoked server-side.

## Getting started

```bash
# Required — refuses to start unsigned rather than silently downgrading
export DMN_HMAC_SECRET=<your-secret>

# Optional (defaults shown)
export DMN_STORAGE_DIR=dmn_checkpoints
export DMN_MIN_EVENTS_FOR_CONSOLIDATION=3

uvicorn app:app

# Test suite: consolidation math, crash recovery (real SIGKILL), tamper detection
pytest tests/
```

## Research

The Delta-Space Adapter Library formalism this service implements, and the
fleet-transfer results it was validated against, are established in:

- Sajeev, A.V. (2026). *The Lár Training Loop: Routing Flags as Gradient
  Signals.* [doi.org/10.5281/zenodo.20581128](https://doi.org/10.5281/zenodo.20581128)
- Sajeev, A.V. (2026). *Fleet Transfer: Cross-Robot Learning from
  Nineteen-Number Events, with a Corrected Physical Validation.*
  [doi.org/10.5281/zenodo.22079216](https://doi.org/10.5281/zenodo.22079216)

## Status

Real, tested (`pytest tests/`, 10/10 passing), not yet load-tested or
deployed persistently. Public as of 2026-09. No `LICENSE` file yet, a
separate, deliberate decision still pending, don't assume Apache 2.0 or
any other terms until one is actually added.

---

*Snath AI Open Source Research Initiative*
