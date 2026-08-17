# data/observed_datasets.py
#
# Observed (real-world) case datasets that can be selected in the dashboard and
# compared against model output. Each dataset is a cross-tab of case counts by
# age and vaccination-up-to-date status.
#
# Mapping to the pertussis partial-immunity model:
#   - "No"  (not up to date) -> naive-track cases      ~ I  (E_to_I)
#   - "Yes" (up to date)     -> partial-track cases    ~ Ip (Ep_to_Ip)
#   - "Unknown"              -> unresolved; excluded from track shares
#
# Ages 1-21 are single years; older rows are explicit bands. "50+" is not split
# into 50-64 / 65+ in the source data.

from constants import DEFAULT_AGE_GROUPS  # ["0-4","5-19","20-49","50-64","65+"]

# --- Raw source table: age_label -> {"No":, "Unknown":, "Yes":} ------------------
LANE_COUNTY_RAW = {
    "1": {"No": 16, "Unknown": 0, "Yes": 13},
    "2": {"No": 12, "Unknown": 0, "Yes": 10},
    "3": {"No": 3, "Unknown": 0, "Yes": 9},
    "4": {"No": 9, "Unknown": 0, "Yes": 1},
    "5": {"No": 4, "Unknown": 1, "Yes": 4},
    "6": {"No": 1, "Unknown": 1, "Yes": 7},
    "7": {"No": 5, "Unknown": 0, "Yes": 7},
    "8": {"No": 6, "Unknown": 0, "Yes": 7},
    "9": {"No": 4, "Unknown": 0, "Yes": 10},
    "10": {"No": 8, "Unknown": 1, "Yes": 18},
    "11": {"No": 22, "Unknown": 1, "Yes": 8},
    "12": {"No": 8, "Unknown": 1, "Yes": 8},
    "13": {"No": 10, "Unknown": 0, "Yes": 9},
    "14": {"No": 4, "Unknown": 0, "Yes": 15},
    "15": {"No": 6, "Unknown": 1, "Yes": 14},
    "16": {"No": 2, "Unknown": 0, "Yes": 35},
    "17": {"No": 1, "Unknown": 1, "Yes": 38},
    "18": {"No": 3, "Unknown": 0, "Yes": 21},
    "19": {"No": 3, "Unknown": 0, "Yes": 19},
    "20": {"No": 3, "Unknown": 2, "Yes": 9},
    "21": {"No": 4, "Unknown": 1, "Yes": 9},
    "22-24": {"No": 3, "Unknown": 2, "Yes": 7},
    "25-29": {"No": 3, "Unknown": 2, "Yes": 4},
    "30-34": {"No": 3, "Unknown": 0, "Yes": 4},
    "35-39": {"No": 2, "Unknown": 0, "Yes": 9},
    "40-44": {"No": 4, "Unknown": 1, "Yes": 6},
    "45-49": {"No": 2, "Unknown": 2, "Yes": 2},
    "50+": {"No": 9, "Unknown": 1, "Yes": 8},
}


def _age_label_to_band(label: str) -> str:
    """Map a source age label to one of DEFAULT_AGE_GROUPS."""
    if label == "50+":
        return "50-64"  # source does not separate 65+; folded here
    if "-" in label:
        lo = int(label.split("-")[0])
    else:
        lo = int(label)
    if lo <= 4:
        return "0-4"
    if lo <= 19:
        return "5-19"
    if lo <= 49:
        return "20-49"
    if lo <= 64:
        return "50-64"
    return "65+"


def aggregate_to_bands(raw: dict = LANE_COUNTY_RAW) -> dict:
    """
    Aggregate the raw table into the model's 5 age bands.

    Returns a dict: band -> {"No":, "Unknown":, "Yes":, "total":,
                             "naive":, "partial":}
    where naive == No and partial == Yes (track proxies).
    """
    bands = {ag: {"No": 0, "Unknown": 0, "Yes": 0} for ag in DEFAULT_AGE_GROUPS}
    for label, counts in raw.items():
        band = _age_label_to_band(label)
        for k in ("No", "Unknown", "Yes"):
            bands[band][k] += counts.get(k, 0)
    for ag, c in bands.items():
        c["total"] = c["No"] + c["Unknown"] + c["Yes"]
        c["naive"] = c["No"]       # not-up-to-date -> naive track (I)
        c["partial"] = c["Yes"]    # up-to-date     -> partial track (Ip)
    return bands


def summary(raw: dict = LANE_COUNTY_RAW) -> dict:
    """Overall totals and case-based track shares (excluding Unknown)."""
    no = sum(c["No"] for c in raw.values())
    unk = sum(c["Unknown"] for c in raw.values())
    yes = sum(c["Yes"] for c in raw.values())
    known = no + yes
    return {
        "No": no, "Unknown": unk, "Yes": yes, "total": no + unk + yes,
        "naive_share": (no / known) if known else None,
        "partial_share": (yes / known) if known else None,
    }


# Registry of available observed datasets (name -> metadata + accessors)
OBSERVED_DATASETS = {
    "Lane County, Oregon — pertussis cases": {
        "geography_hint": "United_States__Oregon__Lane_County",
        "raw": LANE_COUNTY_RAW,
        "note": "Cumulative confirmed pertussis cases by age and vaccination-up-to-date status.",
    },
}
