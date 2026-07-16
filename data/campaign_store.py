# data/campaign_store.py
#
# Persistent store of named vaccination-campaign sets, backed by a JSON file so
# they survive between sessions. Seeded with the CDC DTaP/Tdap schedule as the
# default set. Managed from the Vaccination Planner page and loadable from the
# pertussis vaccination menu.

from __future__ import annotations
import json
from copy import deepcopy
from pathlib import Path

_STORE_PATH = Path(__file__).with_name("saved_campaigns.json")

DEFAULT_SET_NAME = "DTaP/Tdap (CDC)"


def default_dtap_campaigns(sim_length: int = 250) -> list[dict]:
    """
    The CDC DTaP / Tdap immunization schedule expressed as dashboard campaigns
    (mapped onto the model age bands). Because the model has no birth cohorts or
    aging, each schedule element is program-level coverage of an age band over
    the run window rather than a dated dose.
    """
    end = max(0, int(sim_length) - 1)
    years = max(sim_length / 365.0, 0.05)  # decennial adult boosters ~10%/yr

    def camp(name, ages, cov, ve):
        return {
            "name": name,
            "start_day": 0,
            "end_day": end,
            "target_age_groups": ages,
            "coverage": round(cov, 3),
            "ve_sus": ve,
            "rollout": {"shape": "flat", "ramp_up_days": 0},
        }

    return [
        camp("DTaP primary series (doses 1-5, ages 0-4)", ["0-4"], 0.92, 0.80),
        camp("Tdap booster (adolescent ~11y)", ["5-19"], 0.90, 0.80),
        camp("Td/Tdap decennial booster (adults)", ["20-49", "50-64", "65+"], min(1.0, 0.10 * years), 0.70),
    ]


def _seed() -> dict:
    return {DEFAULT_SET_NAME: default_dtap_campaigns()}


def _write(store: dict) -> None:
    with open(_STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2)


def load_saved_campaigns() -> dict:
    """Return {set_name: [campaign, ...]}. Creates the file seeded with the
    DTaP/Tdap default the first time it is accessed (or if it is unreadable)."""
    if not _STORE_PATH.exists():
        seed = _seed()
        _write(seed)
        return seed
    try:
        with open(_STORE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("store is not a dict")
        return data
    except Exception:
        seed = _seed()
        _write(seed)
        return seed


def save_campaign_set(name: str, campaigns: list[dict]) -> dict:
    """Create or overwrite a named campaign set and persist it."""
    store = load_saved_campaigns()
    store[name] = deepcopy(campaigns)
    _write(store)
    return store


def delete_campaign_set(name: str) -> dict:
    """Remove a named set (the default set is re-seeded if the store empties)."""
    store = load_saved_campaigns()
    store.pop(name, None)
    if not store:
        store = _seed()
    _write(store)
    return store


def get_campaign_set(name: str) -> list[dict]:
    """Return a deep copy of a named set (empty list if missing)."""
    return deepcopy(load_saved_campaigns().get(name, []))
