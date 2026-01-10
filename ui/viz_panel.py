# ui/viz_panel.py

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from .plots import plot_contact_matrix, plot_population
from helpers import contact_matrix_df
from schemas import MODEL_COMPS
from constants import DEFAULT_AGE_GROUPS

def compute_metrics_with_deltas(selected_ids, reference_id, scenarios, results):
    base = compute_summary_metrics(selected_ids, scenarios, results)
    if base.empty:
        return base

    ref_name = scenarios[reference_id].get("name", reference_id)

    ref = base[base["scenario"] == ref_name][
        ["age_group", "peak_day", "peak_amplitude", "attack_rate"]
    ].rename(
        columns={
            "peak_day": "peak_day_ref",
            "peak_amplitude": "peak_amplitude_ref",
            "attack_rate": "attack_rate_ref",
        }
    )

    out = base.merge(ref, on="age_group", how="left")

    # deltas
    for m in ["peak_day", "peak_amplitude", "attack_rate"]:
        out[f"{m}_delta"] = out[m] - out[f"{m}_ref"]

        denom = out[f"{m}_ref"].replace({0.0: pd.NA})
        out[f"{m}_pct_delta"] = (out[f"{m}_delta"] / denom) * 100.0

    out["reference"] = ref_name
    return out


def compute_summary_metrics(selected_ids, scenarios, results):
    rows = []

    for sid in selected_ids:
        df = results[sid]
        name = scenarios[sid].get("name", sid)
        cfg = scenarios[sid].get("config", {})
        model = cfg.get("model", "")
        geo = cfg.get("geography", "")

        t = df["t"].to_numpy()

        # Find I_* columns and derive age groups from them
        i_cols = [c for c in df.columns if c.startswith("I_")]
        age_groups = [c.split("_", 1)[1] for c in i_cols]

        for ag in age_groups:
            i_col = f"I_{ag}"
            r_col = f"R_{ag}"
            if i_col not in df.columns or r_col not in df.columns:
                continue

            I = df[i_col].to_numpy()
            R = df[r_col].to_numpy()

            peak_idx = int(I.argmax())
            peak_day = int(t[peak_idx])
            peak_amp = float(I[peak_idx])
            attack_rate = float(R[-1] - R[0])

            rows.append(
                {
                    "scenario": name,
                    "model": model,
                    "geography": geo,
                    "age_group": ag,
                    "peak_day": peak_day,
                    "peak_amplitude": peak_amp,
                    "attack_rate": attack_rate,
                }
            )

    out = pd.DataFrame(rows)

    # Nice ordering for display
    if not out.empty:
        out = out.sort_values(["scenario", "age_group"]).reset_index(drop=True)

    return out


def _get_run_scenarios():
    scenarios = st.session_state.get("scenarios", {})
    results = st.session_state.get("results", {})
    # only scenarios that have results
    run_ids = [sid for sid in scenarios.keys() if sid in results]
    return run_ids, scenarios, results


def _series_from_columns(cols):
    """
    Returns:
      compartments: set[str]
      age_groups: set[str] (may include "" for non-age-stratified)
      pairs: set[tuple(compartment, age_group)]
    """
    pairs = set()
    for c in cols:
        if c == "t":
            continue
        if "_" in c:
            comp, ag = c.split("_", 1)
            pairs.add((comp, ag))
        else:
            pairs.add((c, ""))  # no age stratification
    compartments = {p[0] for p in pairs}
    age_groups = {p[1] for p in pairs}
    return compartments, age_groups, pairs


def _common_pairs(selected_ids, results):
    common = None
    for sid in selected_ids:
        df = results[sid]
        _, _, pairs = _series_from_columns(df.columns)
        common = pairs if common is None else (common & pairs)
    return common or set()


def render_vaccination_timeseries(df: pd.DataFrame) -> None:
    """
    Expects in scenario cfg:
      df  # DataFrame with columns: t, age groups...
    """

    if not isinstance(df, pd.DataFrame) or "t" not in df.columns:
        st.warning("Vaccination schedule is not a valid dataframe with a 't' column.")
        return

    age_cols = [c for c in df.columns if c != "t"]
    if not age_cols:
        st.warning("Vaccination schedule has no age-group columns.")
        return

    st.caption("Explore daily administered doses or cumulative doses by age group.")

    mode = st.radio(
        "View",
        options=["Daily doses", "Cumulative doses"],
        horizontal=True,
        key=f"vax_plot_mode",
    )

    # Prepare long dataframe
    plot_df = df[["t"] + age_cols].copy()

    if mode == "Cumulative doses":
        plot_df[age_cols] = plot_df[age_cols].cumsum()

    long_df = plot_df.melt(id_vars="t", value_vars=age_cols, var_name="age_group", value_name="doses")

    chart = (
        alt.Chart(long_df)
        .mark_area(opacity=0.75)  # stacked area gives "cool" rollout feel
        .encode(
            x=alt.X("t:Q", title="Day"),
            y=alt.Y("sum(doses):Q", title="Doses" if mode == "Daily doses" else "Cumulative doses"),
            color=alt.Color("age_group:N", title="Age group"),
            tooltip=[
                "age_group:N",
                alt.Tooltip("t:Q", title="Day"),
                alt.Tooltip("sum(doses):Q", title="Doses"),
            ],
        )
        .interactive()
    )

    st.altair_chart(chart, use_container_width=True)


def render_compartment_timeseries(compartments, selected_ids, scenarios, results):

    col1, col2 = st.columns([0.5, 0.5])
    with col1:
        comp_idx = np.where(np.array(compartments) == "I")[0][0]
        comp = st.selectbox("Compartment", options=compartments, index=int(comp_idx))
    
    with col2:
        total_ages = ["total"] + DEFAULT_AGE_GROUPS
        age = st.selectbox("Age group", options=total_ages, index=0)

    # ---- Prepare long dataframe for plotting
    series_col = f"{comp}_{age}" if age != "" else comp

    rows = []
    for sid in selected_ids:
        df = results[sid]
        if series_col not in df.columns:
            continue  # should not happen due to intersection logic
        name = scenarios[sid].get("name", sid)

        cfg = scenarios[sid]["config"]
        label = f"{name}" 

        tmp = df[["t", series_col]].copy()
        tmp.rename(columns={series_col: "value"}, inplace=True)
        tmp["scenario"] = label
        rows.append(tmp)

    plot_df = pd.concat(rows, ignore_index=True)

    st.caption(f"Showing: {series_col}")

    chart = (
        alt.Chart(plot_df)
        .mark_line()
        .encode(
            x=alt.X("t:Q", title="Day"),
            y=alt.Y("value:Q", title=series_col),
            color=alt.Color("scenario:N", title="Scenario", scale=alt.Scale(scheme="set2")),
            tooltip=["scenario:N", "t:Q", "value:Q"],
        )
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)


def render_metrics_tab(primary_id, selected_ids, scenarios, results):
    """
    Render the summary metrics tab with comparison to reference scenario.
    
    Args:
        primary_id: ID of the primary/reference scenario
        selected_ids: List of scenario IDs to include (primary + comparisons)
        scenarios: Dictionary of all scenarios
        results: Dictionary of all results
    """
    if not selected_ids:
        st.info("Select at least one scenario to compute summary metrics.")
        return
    
    metrics = compute_metrics_with_deltas(selected_ids, primary_id, scenarios, results)
    if metrics.empty:
        st.warning("No metrics available for the selected scenarios.")
        return

    metric = st.selectbox(
        "Metric",
        options=["peak_day", "peak_amplitude", "attack_rate"],
        format_func=lambda x: {
            "peak_day": "Day of peak",
            "peak_amplitude": "Peak amplitude",
            "attack_rate": "Attack rate (final R)",
        }[x],
    )

    view = st.radio(
        "Display",
        options=["Absolute", "Δ vs reference", "%Δ vs reference"],
        horizontal=True,
    )

    if view == "Absolute":
        value_col = metric
        y_title = metric
    elif view == "Δ vs reference":
        value_col = f"{metric}_delta"
        y_title = f"{metric} (Δ)"
    else:
        value_col = f"{metric}_pct_delta"
        y_title = f"{metric} (%Δ)"

    ref_name = scenarios[primary_id].get("name", primary_id)
    st.caption(f"Reference: {ref_name}")

    # Ensure consistent scenario labels in chart
    metrics = metrics.copy()
    metrics["scenario_label"] = metrics["scenario"].astype(str)

    chart = (
        alt.Chart(metrics)
        .mark_bar()
        .encode(
            x=alt.X("age_group:N", title="Age group"),
            y=alt.Y(f"{value_col}:Q", title=y_title),
            color=alt.Color("scenario:N", title="Scenario", scale=alt.Scale(scheme="set2")),
            xOffset="scenario:N",  # grouped-bar trick
            tooltip=[
                "scenario:N",
                "age_group:N",
                alt.Tooltip(f"{value_col}:Q", title=y_title),
            ],
        )
    )

    st.altair_chart(chart, use_container_width=True)

    # Download button
    csv = metrics.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download metrics (CSV)",
        data=csv,
        file_name="scenario_metrics_with_deltas.csv",
        mime="text/csv",
    )


def render_demographic_and_contacts_tab(
    population, 
    contact_matrices, 
    country_name, 
    sep_width=0.02
    ):
    """Render the population visualization tab."""

    view = st.radio(
            "Display",
            options=["Population", "Contact Matrix"],
            horizontal=True,
        )

    if view == "Population":

        # Population Data Download
        population_data = pd.DataFrame({
            "Age Group": population.Nk_names,
            "Count": population.Nk,
            "Percentage": 100 * population.Nk / population.Nk.sum()
        })
        st.download_button(
            label="Download Population Data (CSV)",
            data=population_data.to_csv(index=False),
            file_name=f"{country_name}_population.csv",
            mime="text/csv",
            use_container_width=False,
            help="Download population by age group (counts and percentages)"
        )

        # Population Metrics
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Total Population", f"{population.Nk.sum():,}")
        with col2: st.metric("Most Populous Age Group", f"{population.Nk_names[population.Nk.argmax()]}")
        with col3: st.metric("Least Populous Age Group", f"{population.Nk_names[population.Nk.argmin()]}")

        # Population Donut Plot
        plot_population(
            population=population,
            facecolor="#0c1019",
        )

    # Contact Matrix Panel
    if view == "Contact Matrix":    
        layer = st.selectbox("Contact layer", 
                            options=["overall", "home", "school", "work", "community"], 
                            index=0, 
                            help="Choose which contact layer to visualize")

        # Contact Matrix Data Download
        contact_df = contact_matrix_df(layer, population.contact_matrices, population.Nk_names)
        st.download_button(
                label="Download Contact Matrix (CSV)",
                data=contact_df.round(3).to_csv(index=True),
                file_name=f"{country_name}_contacts_{layer}.csv",
                mime="text/csv",
                use_container_width=False,
                help="Download contact matrix data with age groups and contact rates"
            )

        # Contact Matrix Metrics
        if layer == "overall":
            matrix = sum(population.contact_matrices.values())
        else:
            matrix = population.contact_matrices[layer]
        matrix = np.array(matrix)

        col1, col2, col3, col4 = st.columns(4)    
        with col1: st.metric("Most Active Age Group", f"{population.Nk_names[matrix.sum(axis=1).argmax()]}")
        with col2: st.metric("Least Active Age Group", f"{population.Nk_names[matrix.sum(axis=1).argmin()]}")
        with col3: st.metric("Mean Contact Rate", f"{matrix.sum(axis=1).mean():.2f}")
        with col4: st.metric("Spectral Radius", f"{np.linalg.eigvals(matrix).max().real:.2f}")
        
        # Contact Matrix Plot
        plot_contact_matrix(
            layer=layer,
            matrices=contact_matrices,  
            groups=population.Nk_names,  
            facecolor="#0c1019",
            cmap="oranges",
        )


def render_viz_panel(model: str, geography: str) -> None:
    st.subheader("Visualisation")

    run_ids, scenarios, results = _get_run_scenarios()
    if not run_ids:
        st.info("No run results yet. Click Run to generate output.")
        return

    # ---- SCENARIO SELECTION (shared across all tabs) ----
    st.markdown("**Scenario Selection**")
    
    last_run = st.session_state.get("last_run_scenario_id")
    default_primary = last_run if last_run in run_ids else run_ids[0]

    col1, col2 = st.columns([0.5, 0.5])
    
    with col1:
        primary_id = st.selectbox(
            "Primary scenario",
            options=run_ids,
            index=run_ids.index(default_primary),
            format_func=lambda sid: scenarios[sid].get("name", sid),
            key="viz_primary_scenario"
        )

    compare_pool = [sid for sid in run_ids if sid != primary_id]
    
    with col2:
        compare_ids = st.multiselect(
            "Compare with",
            options=compare_pool,
            default=[],
            format_func=lambda sid: scenarios[sid].get("name", sid),
            key="viz_compare_scenarios"
        )
    
    selected_ids = [primary_id] + compare_ids

    tab_ts, tab_metrics, tab_contact_interventions, tab_vaccinations, tab_population = st.tabs(["Trajectories", "Summary metrics", "Contact Interventions", "Vaccinations", "Population"])

    with tab_ts:
        render_compartment_timeseries(MODEL_COMPS[model], selected_ids, scenarios, results)

    with tab_metrics:
        render_metrics_tab(primary_id, selected_ids, scenarios, results)

    with tab_vaccinations:
        render_vaccination_timeseries(scenarios[primary_id]["config"]["daily_doses_by_age"])

    with tab_population: 
        render_demographic_and_contacts_tab(
            population=scenarios[primary_id]["config"]["population"],
            contact_matrices=scenarios[primary_id]["config"]["population"].contact_matrices,
            country_name=scenarios[primary_id]["config"]["geography"],
        )