import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from typing import List, Dict


def _get_age_group_labels(population) -> List[str]:
    return [str(x) for x in population.Nk_names]


def _compute_contact_rhos(population, interventions: Dict[str, Dict], num_days: int) -> Dict[str, List[float]]:
    layers = list(population.contact_matrices.keys())
    mats = {layer: np.array(population.contact_matrices[layer]) for layer in layers}
    base_rho_layer = {layer: float(np.linalg.eigvals(mats[layer]).real.max()) for layer in layers}
    rhos = {layer: [0.0] * num_days for layer in layers}
    rhos["overall"] = [0.0] * num_days

    iv = interventions or {}
    for day in range(num_days):
        scaled_sum = None
        for layer in layers:
            red = 0.0
            if layer in iv:
                start = int(iv[layer].get("start", 0))
                end = int(iv[layer].get("end", -1))
                reduction = float(iv[layer].get("reduction", 0.0))
                if start <= day <= end:
                    red = reduction
            factor = max(0.0, 1.0 - red)
            rhos[layer][day] = base_rho_layer[layer] * factor
            scaled = mats[layer] * factor
            scaled_sum = scaled if scaled_sum is None else (scaled_sum + scaled)
        rhos["overall"][day] = float(np.linalg.eigvals(scaled_sum).real.max()) if scaled_sum is not None else 0.0

    # Convert to % of baseline (day 0) for each series (baseline = 100%)
    pct: Dict[str, List[float]] = {}
    for k, series in rhos.items():
        base = series[0] if len(series) > 0 else 0.0
        if base == 0.0:
            pct[k] = [0.0 for _ in series]
        else:
            pct[k] = [(v / base) * 100.0 for v in series]
    return pct


def render_interventions_tab(population, simulation_dates):
    st.subheader("Interventions")

    if population is None or simulation_dates is None:
        st.info("Run a simulation to view vaccination schedules.")
        return

    Nk = np.asarray(population.Nk)
    num_days = len(simulation_dates)

    # Build vaccinations chart
    df_vaccination_schedule = st.session_state.get("simulation_output", {}).get("vaccination_schedule")

    vax_chart = None
    if not df_vaccination_schedule.empty:
        # group by name and sum the doses
        df_vaccination_schedule = df_vaccination_schedule.groupby(["t"], as_index=False).sum(numeric_only=True)
        age_labels = _get_age_group_labels(population)
        rows = []
        for day_idx in range(num_days):
            for label in age_labels:
                doses = df_vaccination_schedule.loc[df_vaccination_schedule["t"] == day_idx, label].values[0]
                rows.append({"Day": day_idx, "Age group": label, "Doses": int(doses)})
        df = pd.DataFrame.from_records(rows)
    
        vax_chart = alt.Chart(df).mark_bar(size=3).encode(
            x=alt.X("Day:Q", title="Time (days)", scale=alt.Scale(domain=[0, max(0, num_days - 1)])),
            y=alt.Y("sum(Doses):Q", title="Doses per day"),
            color=alt.Color(
                "Age group:N",
                legend=alt.Legend(title="Age group", orient="bottom", direction="horizontal")
            ),
            tooltip=["Day:Q", "Age group:N", "Doses:Q"]
        ).properties(height=320, width=700)


    # Build contact intensity chart
    interventions = st.session_state.get("interventions", {})
    rhos_pct = _compute_contact_rhos(population, interventions, num_days)
    rows = []
    for layer, series in rhos_pct.items():
        for day, val in enumerate(series):
            rows.append({"Day": day, "Layer": layer, "Pct": float(val)})
    df_rho = pd.DataFrame(rows)
    layers = list(rhos_pct.keys())
    if "overall" in layers:
        layers = ["overall"] + [l for l in layers if l != "overall"]
    palette = ["#50f0d8", "#344b47", "#97b1ab", "#84caff", "#4395e3"]
    color_scale = alt.Scale(domain=layers, range=palette[:len(layers)])
    rho_chart = alt.Chart(df_rho).mark_line().encode(
        x=alt.X("Day:Q", title="Time (days)", scale=alt.Scale(domain=[0, max(0, num_days - 1)])),
        y=alt.Y("Pct:Q", title="Contact intensity (% of baseline)"),
        color=alt.Color("Layer:N", scale=color_scale, legend=alt.Legend(title=None, orient="bottom", direction="horizontal")),
        size=alt.condition(alt.datum.Layer == "overall", alt.value(3.0), alt.value(1.6)),
        tooltip=["Layer:N", "Day:Q", alt.Tooltip("Pct:Q", format=".2f")]
    ).properties(height=320, width=350)

    # Concatenate horizontally to align baselines
    if vax_chart is None:
        st.info("No vaccination doses scheduled within the simulation window.")
        st.altair_chart(rho_chart, use_container_width=True)
    else:
        combined = (
                alt.hconcat(vax_chart, rho_chart, spacing=16)
                .resolve_scale(y='independent', color='independent', x="independent")
                .resolve_legend(color='independent')
            )
        st.altair_chart(combined, use_container_width=True)


