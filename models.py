"""Pydantic request/response schemas for the DMN HTTP service."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class EventIn(BaseModel):
    """
    A compact event submitted by one fleet unit. Mirrors the exact schema
    already used by the real artifacts this service is grounded in:
    ``fleet_event_signed.json`` (class_name, concept_anchor, n_source_events,
    schema) and ``ood_guard.py``'s D-hard events (failure_class, z_proprio /
    z_vision, divergence). Only the compact anchor crosses the wire -- never
    raw images/observations.
    """
    unit_id: str = Field(..., description="Identifier of the submitting fleet unit")
    class_name: str = Field(..., description="Named failure/concept class this event belongs to")
    concept_anchor: List[float] = Field(..., min_length=1, description="Small numeric anchor vector (Delta_i)")
    n_source_events: int = Field(1, ge=1, description="How many local hard-case events this unit's own anchor was derived from")
    failure_class: Optional[str] = Field(None, description="One of the paper's named failure classes, if applicable (for decay-lambda lookup)")
    schema_labels: Optional[List[str]] = Field(None, description="Optional per-dimension labels for the anchor vector")
    timestamp: Optional[str] = Field(None, description="ISO-8601 timestamp; server fills in if omitted")


class EventAck(BaseModel):
    class_name: str
    stored_events_for_class: int
    consolidation_triggered: bool
    min_events_for_consolidation: int


class LibraryEntryOut(BaseModel):
    class_name: str
    failure_class: str
    centroid: List[float]
    dim: int
    tau_sim: float
    n_events: int
    n_source_events_total: int
    decay_lambda: float
    w_min: float
    consolidated_at: str
    contributing_units: List[str]


class LibraryOut(BaseModel):
    classes: Dict[str, LibraryEntryOut]
    commit_counter: int


class ConsolidateResult(BaseModel):
    class_name: str
    consolidated: bool
    reason: Optional[str] = None
    entry: Optional[LibraryEntryOut] = None


class ResolveCommitRequest(BaseModel):
    step: int
    happened: bool


class HealthOut(BaseModel):
    status: str
    unconfirmed_library_commits: int
    detail: Optional[str] = None
