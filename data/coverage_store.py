# data/coverage_store.py
#
# Persistent store of named "vaccination coverage / immunity by age" profiles,
# backed by a JSON file so they survive between sessions. Seeded with an Oregon
# coverage estimate. Managed from the Vaccination Planner page and selectable
# from the main dashboard's Initial conditions (applies as age-stratified
# starting immunity for the pertussis model).
#
# A profile maps each model age band -> percent immune / up-to-date.

from __future__ import annotations
import json
from copy import deepcopy
from pathlib import Path

from constants import DEFAULT_AGE_GROUPS
from data.observed_datasets import aggregate_to_bands, LANE_COUNTY_RAW

_STORE_PATH = Path(__file__).with_name("saved_coverage.json")

DEFAULT_PROFILE_NAME = "Lane County (observed)"


def default_coverage() -> dict:
    """Seed profile derived from the Lane County observed case data (the same
    dataset used in the calibration area): the percent of cases in each age band
    that were up-to-date on vaccination (Yes / (Yes + No), excluding Unknown).

    Note: this is the vaccinated *share of cases*, which is not identical to
    population vaccination coverage — in a highly vaccinated population many
    cases are breakthroughs. Replace with official ALERT IIS coverage where a
    true denominator is available.
    """
    bands = aggregate_to_bands(LANE_COUNTY_RAW)
    prof = {}
    for ag in DEFAULT_AGE_GROUPS:
        known = bands[ag]["naive"] + bands[ag]["partial"]  # No + Yes
        prof[ag] = round(100.0 * bands[ag]["partial"] / known, 1) if known > 0 else 0.0
    # The source folds 65+ into "50+"; backfill 65+ from the 50-64 value.
    if prof.get("65+", 0.0) == 0.0 and bands["65+"]["total"] == 0:
        prof["65+"] = prof.get("50-64", 0.0)
    return prof


def _normalise(mapping: dict) -> dict:
    """Return a profile with every model age band present (missing -> 0)."""
    return {ag: float(mapping.get(ag, 0.0)) for ag in DEFAULT_AGE_GROUPS}


def _seed() -> dict:
    return {DEFAULT_PROFILE_NAME: default_coverage()}


def _write(store: dict) -> None:
    with open(_STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2)


def load_saved_coverage() -> dict:
    """Return {profile_name: {age_band: pct}}. Creates the file seeded with the
    Oregon estimate on first access (or if unreadable)."""
    if not _STORE_PATH.exists():
        seed = _seed()
        _write(seed)
        return seed
    try:
        with open(_STORE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("store is not a dict")
        return {name: _normalise(prof) for name, prof in data.items()}
    except Exception:
        seed = _seed()
        _write(seed)
        return seed


def save_coverage_profile(name: str, mapping: dict) -> dict:
    """Create or overwrite a named coverage profile and persist it."""
    store = load_saved_coverage()
    store[name] = _normalise(mapping)
    _write(store)
    return store


def delete_coverage_profile(name: str) -> dict:
    """Remove a named profile (re-seeds the default if the store empties)."""
    store = load_saved_coverage()
    store.pop(name, None)
    if not store:
        store = _seed()
    _write(store)
    return store


def get_coverage_profile(name: str) -> dict:
    """Return a deep copy of a named profile (seed default if missing)."""
    return deepcopy(load_saved_coverage().get(name, default_coverage()))
