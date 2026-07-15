# ui/viz_panel.py

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from copy import deepcopy
from .plots import plot_contact_matrix, plot_population
from helpers import contact_matrix_df
from schemas import MODEL_COMPS
from constants import DEFAULT_AGE_GROUPS
from collections import OrderedDict

ages_to_idx = {ag: i for i, ag in enumerate(DEFAULT_AGE_GROUPS)}


def compute_metrics_with_deltas(selected_ids, reference_id, scenarios, results):
    base = compute_summary_metrics(selected_ids, scenarios, results)
    if base.empty:
        return base

    ref_name = scenarios[reference_id].get("name", reference_id)

    ref = base[base["scenario"] == ref_name][
        ["age_group", "peak_day", "peak_amplitude", "attack_rate", "total_infections", "hospitalizations", "hospitalization_rate"]
    ].rename(
        columns={
            "peak_day": "peak_day_ref",
            "peak_amplitude": "peak_amplitude_ref",
            "attack_rate": "attack_rate_ref",
            "total_infections": "total_infections_ref",
            "hospitalizations": "hospitalizations_ref",
            "hospitalization_rate": "hospitalization_rate_ref",
        }
    )

    out = base.merge(ref, on="age_group", how="left")

    # deltas
    for m in ["peak_day", "peak_amplitude", "attack_rate", "total_infections", "hospitalizations", "hospitalization_rate"]:
        out[f"{m}_delta"] = out[m] - out[f"{m}_ref"]

        denom = out[f"{m}_ref"].replace({0.0: pd.NA})
        out[f"{m}_pct_delta"] = (out[f"{m}_delta"] / denom) * 100.0

    out["reference"] = ref_name
    return out


def compute_summary_metrics(selected_ids, scenarios, results):
    rows = []

    for sid in selected_ids:
        df_comp, df_trans = results[sid]["compartments"], results[sid]["transitions"]
        name = scenarios[sid].get("name", sid)
        cfg = scenarios[sid].get("config", {})
        population = cfg.get("population", {})
        model = cfg.get("model", "")
        geo = cfg.get("geography", "")

        t = df_comp["t"].to_numpy()

        for ag in DEFAULT_AGE_GROUPS + ["total"]:
            i_col = f"I_{ag}"
            r_col = f"R_{ag}"
            h_col = f"H_{ag}"

            e_to_i_col = f"E_to_I_{ag}"

            I = df_comp[i_col].to_numpy()
            R = df_comp[r_col].to_numpy()
            E_to_I = df_trans[e_to_i_col].to_numpy()

            # Pertussis has a parallel partial-immunity track (Ip). Include it in
            # prevalence and new-infection metrics so the "silent" partial cases
            # are not undercounted.
            if model == "SEIRS (Pertussis)":
                ip_col = f"Ip_{ag}"
                ep_to_ip_col = f"Ep_to_Ip_{ag}"
                if ip_col in df_comp.columns:
                    I = I + df_comp[ip_col].to_numpy()
                if ep_to_ip_col in df_trans.columns:
                    E_to_I = E_to_I + df_trans[ep_to_ip_col].to_numpy()

            peak_idx = int(I.argmax())
            peak_day = int(t[peak_idx])
            peak_amp = float(I[peak_idx])
            total_infections = float(np.sum(E_to_I))

            if ag == "total":
                attack_rate = 100.0 * float(np.sum(E_to_I) / population.Nk.sum())
            else:
                attack_rate = 100.0 * float(np.sum(E_to_I) / population.Nk[ages_to_idx[ag]])

            if model == "SEIHR (COVID-19)":
                H = df_comp[h_col].to_numpy()
                hospitalizations = float(np.sum(H))
                if ag == "total":
                    hospitalization_rate = 100.0 * float(np.sum(H) / population.Nk.sum())
                else:
                    hospitalization_rate = 100.0 * float(np.sum(H) / population.Nk[ages_to_idx[ag]])
            else:
                hospitalizations = None
                hospitalization_rate = None
            
            rows.append(
                {
                    "scenario": name,
                    "model": model,
                    "geography": geo,
                    "age_group": ag,
                    "peak_day": peak_day,
                    "peak_amplitude": peak_amp,
                    "attack_rate": attack_rate,
                    "total_infections": total_infections,
                    "hospitalizations": hospitalizations,
                    "hospitalization_rate": hospitalization_rate,
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


def render_spectral_radius_timeseries(dfs_dict: dict) -> None:
    """
    Render spectral radius timeseries comparison across scenarios.
    
    Args:
        dfs_dict: Dictionary of {scenario_name: spectral_radius_df}
                 Each df has columns: t, rho, rho_perc, layer
    """
    if not dfs_dict:
        st.warning("No spectral radius data available.")
        return

    st.caption("Compare contact matrix spectral radius across scenarios.")

    # Prepare combined dataframe
    all_data = []
    layers_set = set()
    
    for scenario_name, df in dfs_dict.items():
        if not isinstance(df, pd.DataFrame) or "t" not in df.columns:
            continue
        
        required_cols = ["t", "rho", "rho_perc", "layer"]
        if not all(col in df.columns for col in required_cols):
            continue
        
        layers_set.update(df["layer"].unique())
        df_copy = df.copy()
        df_copy["scenario"] = scenario_name
        all_data.append(df_copy)
    
    if not all_data:
        st.warning("No valid spectral radius data to display.")
        return
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Get available layers in desired order
    layer_order = ["overall", "home", "school", "work", "community"]
    available_layers = [layer for layer in layer_order if layer in layers_set]
    
    # Controls
    col1, col2 = st.columns(2)
    
    with col1:
        mode = st.radio(
            "View",
            options=["Percentage change", "Absolute value"],
            horizontal=True,
            key="spectral_radius_mode",
        )
    
    with col2:
        selected_layer = st.selectbox(
            "Contact layer",
            options=available_layers,
            index=0,  # Default to first available (typically "overall")
            key="spectral_radius_layer"
        )
    
    # Filter data for selected layer
    plot_data = combined_df[combined_df["layer"] == selected_layer]
    
    # Determine which column to plot
    y_col = "rho_perc" if mode == "Percentage change" else "rho"
    y_title = "Spectral radius (% change)" if mode == "Percentage change" else "Spectral radius"
    
    # Create chart with all scenarios as different colored lines
    chart = (
        alt.Chart(plot_data)
        .mark_line(strokeWidth=2, point=True)
        .encode(
            x=alt.X("t:Q", title="Day"),
            y=alt.Y(
                f"{y_col}:Q", 
                title=y_title
            ),
            color=alt.Color(
                "scenario:N", 
                title="Scenario",
                scale=alt.Scale(scheme="set2")
            ),
            tooltip=[
                alt.Tooltip("scenario:N", title="Scenario"),
                alt.Tooltip("t:Q", title="Day"),
                alt.Tooltip(f"{y_col}:Q", title=y_title, format=".3f"),
            ],
        )
        .interactive()
    )
    
    st.altair_chart(chart, use_container_width=True)
    

def render_vaccination_timeseries(dfs_dict: dict) -> None:
    """
    Render vaccination timeseries comparison across scenarios.
    
    Args:
        dfs_dict: Dictionary of {scenario_name: vaccination_df}
    """
    if not dfs_dict:
        st.warning("No vaccination schedules available.")
        return

    st.caption("Compare vaccination rollout across scenarios.")

    # Controls
    col1, col2 = st.columns(2)
    
    with col1:
        mode = st.radio(
            "View",
            options=["Daily doses", "Cumulative doses"],
            horizontal=True,
            key="vax_plot_mode",
        )
    
    # Prepare combined long dataframe
    all_data = []
    age_groups_set = set()
    
    for scenario_name, df in dfs_dict.items():
        if not isinstance(df, pd.DataFrame) or "t" not in df.columns:
            continue
        
        age_cols = [c for c in df.columns if c != "t"]
        if not age_cols:
            continue
        
        age_groups_set.update(age_cols)
        plot_df_vax = df[["t"] + age_cols].copy()

        if mode == "Cumulative doses":
            plot_df_vax[age_cols] = plot_df_vax[age_cols].cumsum()

        long_df = plot_df_vax.melt(
            id_vars="t", 
            value_vars=age_cols, 
            var_name="age_group", 
            value_name="doses"
        )
        long_df["scenario"] = scenario_name
        all_data.append(long_df)
    
    if not all_data:
        st.warning("No valid vaccination schedules to display.")
        return
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Age group selector
    with col2:
        selected_age = st.selectbox(
            "Age group",
            options=DEFAULT_AGE_GROUPS + ["total"],
            key="vax_age_selector"
        )
    
    # Filter data for selected age group
    plot_data = combined_df[combined_df["age_group"] == selected_age]
    
    # Create chart with all scenarios as different colored lines
    chart = (
        alt.Chart(plot_data)
        .mark_line(strokeWidth=2., point=False)
        .encode(
            x=alt.X("t:Q", title="Day"),
            y=alt.Y(
                "doses:Q", 
                title="Doses" if mode == "Daily doses" else "Cumulative doses"
            ),
            color=alt.Color(
                "scenario:N", 
                title="Scenario", 
                scale=alt.Scale(scheme="set2")
            ),
            tooltip=[
                alt.Tooltip("scenario:N", title="Scenario"),
                alt.Tooltip("t:Q", title="Day"),
                alt.Tooltip("doses:Q", title="Doses", format=","),
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
        df = results[sid]["compartments"]
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

    series_col_label = series_col.replace("_", " (") + ")"
    st.caption(f"Showing: {series_col_label}")

    chart = (
        alt.Chart(plot_df)
        .mark_line()
        .encode(
            x=alt.X("t:Q", title="Day"),
            y=alt.Y("value:Q", title=series_col_label),
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
    
    model = scenarios[primary_id]["config"]["model"]
    if model == "SEIHR (COVID-19)":
        metrics_labels = {
            "attack_rate": "Final Attack Rate (%)",
            "total_infections": "Total Infections",
            "peak_day": "Peak Prevalence Day",
            "peak_amplitude": "Peak Prevalence Amplitude",
            "hospitalizations": "Total Hospitalizations",
            "hospitalization_rate": "Hospitalization Rate (%)",
        }
    else:
        metrics_labels = {
            "attack_rate": "Final Attack Rate (%)",
            "total_infections": "Total Infections",
            "peak_day": "Peak Prevalence Day",
            "peak_amplitude": "Peak Prevalence Amplitude",
        }

    metric = st.selectbox("Metric", options=metrics_labels.keys(), format_func=lambda x: metrics_labels[x])

    view = st.radio(
        "Display",
        options=["Absolute", "Δ vs reference", "Relative Δ vs reference"],
        horizontal=True,
    )

    if view == "Absolute":
        value_col = metric
        y_title = metrics_labels[metric]
    elif view == "Δ vs reference":
        value_col = f"{metric}_delta"
        y_title = f"{metrics_labels[metric]} (Δ)"
    else:
        value_col = f"{metric}_pct_delta"
        y_title = f"{metrics_labels[metric]} (Relative Δ)"

    ref_name = scenarios[primary_id].get("name", primary_id)
    st.caption(f"Reference: {ref_name}")

    # Ensure consistent scenario labels in chart
    metrics = metrics.copy()
    metrics["scenario_label"] = metrics["scenario"].astype(str)

    chart = (
        alt.Chart(metrics)
        .mark_bar()
        .encode(
            x=alt.X(
                "age_group:N", 
                title="Age group", 
                sort=DEFAULT_AGE_GROUPS + ["total"]
            ),
            y=alt.Y(f"{value_col}:Q", title=y_title),
            color=alt.Color("scenario:N", title="Scenario", scale=alt.Scale(scheme="set2")),
            xOffset="scenario:N",  
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


def render_observed_comparison(primary_id, scenarios, results):
    """Compare the primary run against the selected observed case dataset."""
    from data.observed_datasets import OBSERVED_DATASETS
    from engine.calibration import observed_targets, modeled_case_summary, fit_error

    cfg = scenarios[primary_id]["config"]
    ds_name = cfg.get("observed_dataset")
    if not ds_name or ds_name not in OBSERVED_DATASETS:
        st.info("No observed dataset selected for the primary scenario.")
        return

    raw = OBSERVED_DATASETS[ds_name]["raw"]
    obs = observed_targets(raw)
    mod = modeled_case_summary(results[primary_id]["transitions"])
    err = fit_error(obs, mod)

    st.caption(
        f"Primary scenario **{scenarios[primary_id].get('name', primary_id)}** vs observed "
        f"**{ds_name}**. Model incidence is cumulative new infections "
        "(E→I naive, Eₚ→Iₚ partial); observed is cumulative cases."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Observed vaccinated share", f"{obs['overall_partial_share']:.0%}")
    c2.metric("Modeled vaccinated share", f"{mod['overall_partial_share']:.0%}")
    c3.metric("Age-dist RMSE (fit)", f"{err['age_dist_rmse']:.3f}")

    # --- Age distribution: observed vs modeled (normalised shares) ---
    rows = []
    for i, ag in enumerate(DEFAULT_AGE_GROUPS):
        rows.append({"age_group": ag, "source": "Observed", "share": float(obs["age_dist"][i])})
        rows.append({"age_group": ag, "source": "Modeled", "share": float(mod["age_dist"][i])})
    dist_df = pd.DataFrame(rows)

    st.markdown("**Case age distribution (share of total)**")
    chart = (
        alt.Chart(dist_df)
        .mark_bar()
        .encode(
            x=alt.X("age_group:N", title="Age group", sort=DEFAULT_AGE_GROUPS),
            y=alt.Y("share:Q", title="Share of cases", axis=alt.Axis(format="%")),
            color=alt.Color("source:N", title="", scale=alt.Scale(scheme="set2")),
            xOffset="source:N",
            tooltip=["age_group:N", "source:N", alt.Tooltip("share:Q", format=".1%")],
        )
    )
    st.altair_chart(chart, use_container_width=True)

    # --- Vaccinated (partial) share by band ---
    srows = []
    for i, ag in enumerate(DEFAULT_AGE_GROUPS):
        o = obs["partial_share_by_band"][i]
        m = mod["partial_share_by_band"][i]
        if np.isfinite(o):
            srows.append({"age_group": ag, "source": "Observed", "share": float(o)})
        if np.isfinite(m):
            srows.append({"age_group": ag, "source": "Modeled", "share": float(m)})
    if srows:
        share_df = pd.DataFrame(srows)
        st.markdown("**Vaccinated (partial-track) share of cases, by age band**")
        chart2 = (
            alt.Chart(share_df)
            .mark_bar()
            .encode(
                x=alt.X("age_group:N", title="Age group", sort=DEFAULT_AGE_GROUPS),
                y=alt.Y("share:Q", title="Vaccinated share", axis=alt.Axis(format="%")),
                color=alt.Color("source:N", title="", scale=alt.Scale(scheme="set2")),
                xOffset="source:N",
                tooltip=["age_group:N", "source:N", alt.Tooltip("share:Q", format=".1%")],
            )
        )
        st.altair_chart(chart2, use_container_width=True)

    # --- Raw comparison table + download ---
    table = pd.DataFrame({
        "age_group": DEFAULT_AGE_GROUPS,
        "observed_cases": obs["age_counts"].astype(int),
        "observed_naive(No)": obs["naive_by_band"].astype(int),
        "observed_partial(Yes)": obs["partial_by_band"].astype(int),
        "modeled_new_infections": np.round(mod["age_counts"]).astype(int),
        "modeled_naive": np.round(mod["naive_by_band"]).astype(int),
        "modeled_partial": np.round(mod["partial_by_band"]).astype(int),
    })
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.download_button(
        "Download observed-vs-modeled (CSV)",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name=f"observed_vs_modeled_{ds_name}.csv",
        mime="text/csv",
    )
    st.caption(
        "Note: the modeled age distribution is driven by the geography's contact matrix and "
        "will not necessarily match reporting-skewed surveillance data (e.g. adolescent-heavy "
        "pertussis case counts). The vaccinated-share comparison is the more meaningful "
        "structural check for this model."
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

    has_observed = bool(scenarios[primary_id]["config"].get("observed_dataset"))
    tab_labels = ["Trajectories", "Summary metrics", "Contact Interventions", "Vaccinations", "Population"]
    if has_observed:
        tab_labels.append("Observed vs modeled")
    _tabs = st.tabs(tab_labels)
    tab_ts, tab_metrics, tab_contact_interventions, tab_vaccinations, tab_population = _tabs[:5]

    with tab_ts:
        render_compartment_timeseries(MODEL_COMPS[model], selected_ids, scenarios, results)
        # Download button of all timeseries
        complete_df = pd.DataFrame()
        for sid in selected_ids:
            df = results[sid]["compartments"]
            df["scenario"] = scenarios[sid].get("name", sid)
            complete_df = pd.concat([complete_df, df], ignore_index=True)

        # Download button
        country_name = scenarios[selected_ids[0]]["config"]["geography"]
        csv = complete_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download timeseries (CSV)",
            data=csv,
            file_name=f"{country_name}_{model}_all_timeseries.csv",
            mime="text/csv",
        )

    with tab_metrics:
        render_metrics_tab(primary_id, selected_ids, scenarios, results)

    with tab_contact_interventions:
        dfs_dict = {scenarios[sid].get("name", sid): scenarios[sid]["config"]["spectral_radius_df"] for sid in selected_ids}
        dfs_dict = OrderedDict(sorted(dfs_dict.items(), key=lambda item: item[0]))
        render_spectral_radius_timeseries(dfs_dict)

    with tab_vaccinations:
        dfs_dict = {scenarios[sid].get("name", sid): scenarios[sid]["config"]["daily_doses_by_age_daily"] for sid in selected_ids}
        dfs_dict = OrderedDict(sorted(dfs_dict.items(), key=lambda item: item[0]))
        render_vaccination_timeseries(dfs_dict)

        complete_vax_df = pd.DataFrame()
        for sid in selected_ids:
            df = scenarios[sid]["config"]["daily_doses_by_age_daily"]
            df["scenario"] = scenarios[sid].get("name", sid)
            complete_vax_df = pd.concat([complete_vax_df, df], ignore_index=True)

        country_name = scenarios[selected_ids[0]]["config"]["geography"]
        csv = complete_vax_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download vaccination timeseries (CSV)",
            data=csv,
            file_name=f"{country_name}_{model}_all_vaccinations.csv",
            mime="text/csv",
            use_container_width=False,
            help="Download vaccination timeseries data with age groups and daily doses"
        )

    with tab_population:
        render_demographic_and_contacts_tab(
            population=scenarios[primary_id]["config"]["population"],
            contact_matrices=scenarios[primary_id]["config"]["population"].contact_matrices,
            country_name=scenarios[primary_id]["config"]["geography"],
        )

    if has_observed:
        with _tabs[5]:
            render_observed_comparison(primary_id, scenarios, results)