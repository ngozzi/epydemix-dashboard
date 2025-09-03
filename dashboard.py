import streamlit as st
import matplotlib.pyplot as plt
from epydemix.model import load_predefined_model
from epydemix.population import load_epydemix_population
from epydemix.utils import compute_simulation_dates
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from visualization import plot_contact_matrix, plot_contact_intensity, plot_population, plot_compartments_traj
from utils import invalidate_results, load_locations
from compute_statistics import compute_attack_rate, compute_peak_size, compute_peak_time, compute_endemic_state

# ---------- LAYOUT ----------
st.sidebar.markdown(
    """
    <a href="https://epydemix.org" target="_blank">
        <img src="https://cdn.prod.website-files.com/67bde9057c9d78157874e100/67c1d1122127f0a9ce202197_epydemix-logo-p-500.png" 
             style="width:100%;">
    </a>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    .fixed-logo {
        position: fixed;
        bottom: 60px;   /* adjust vertical position */
        right: 20px;    /* adjust horizontal position */
        width: 100px;
        z-index: 100;
    }
    </style>
    <a href="https://www.isi.it" target="_blank">
        <img src="https://www.isi.it/wp-content/uploads/2023/11/isi-logo-white.svg" class="fixed-logo">
    </a>
    """,
    unsafe_allow_html=True
)

# ---------- CONSTANTS ----------
start_date = datetime(2024, 1, 1)
LAYER_NAMES = ["home", "school", "work", "community"]
facecolor="#0c1019"

# ---------- SIDEBAR ----------
st.sidebar.title("Configuration")
with st.sidebar.form("sim_cfg"):

    st.markdown("### ⚙️ Simulation & Model Settings")

    with st.expander("Simulation Parameters", expanded=True):
        model_type = st.selectbox("Model Type", ["SIR", "SEIR", "SIS"], index=1)
        n_v = st.number_input("Number of simulations", 1, 100, 10)
        simulation_days_v = st.number_input("Simulation Days", 1, 365, 250)
        country_name = st.selectbox(
            "Select Country",
            options=load_locations(),
            index=0
        )

    # Model Parameters
    with st.expander("Model Parameters", expanded=False):
        R0_v = st.slider("Reproduction number ($R_0$)", 0.0, 20.0, 2.5, 0.1)
        infectious_period_v = st.slider("Infectious Period (days)", 1.0, 20.0, 4.0, 0.5)
        if model_type == "SEIR":
            incubation_period_v = st.slider("Incubation Period (days)", 1.0, 20.0, 3.0, 0.5)

    # Initial Conditions
    with st.expander("Initial Conditions", expanded=False):
        initial_infected_v = st.number_input("Initial Infected", 1, 100000, 10)

        if model_type != "SIS":
            initial_immunity_percentage_v = st.number_input("Initial Immunity Percentage", 0.0, 100.0, 0.0, 1.0)

    # Interventions
    # Interventions
    st.markdown("### 🤝 Contact Interventions")

    # Initialize once
    for layer in LAYER_NAMES:
        st.session_state.setdefault(f"{layer}_en", False)
        st.session_state.setdefault(f"{layer}_start", 0)
        st.session_state.setdefault(f"{layer}_end", int(simulation_days_v))
        st.session_state.setdefault(f"{layer}_red", 0)  # percent (int)

    for sel in LAYER_NAMES:
        with st.expander(f"Configure: {sel}", expanded=False):

            # Let the widget manage state via key (no manual assignment)
            st.checkbox(
                f"Enable intervention on {sel}",
                key=f"{sel}_en",
                value=bool(st.session_state[f"{sel}_en"]),
            )

            # Read current state to compute safe defaults
            cur_start = int(st.session_state[f"{sel}_start"])
            cur_end   = int(st.session_state[f"{sel}_end"])

            c1, c2 = st.columns(2)
            with c1:
                st.number_input(
                    "Start day",
                    min_value=0,
                    max_value=int(simulation_days_v),
                    value=cur_start,
                    step=1,
                    key=f"{sel}_start",
                )

            # Re-read start after potential change this render
            new_start = int(st.session_state[f"{sel}_start"])
            safe_end_default = max(new_start, cur_end)

            with c2:
                st.number_input(
                    "End day",
                    min_value=new_start,
                    max_value=int(simulation_days_v),
                    value=safe_end_default,
                    step=1,
                    key=f"{sel}_end",
                )

            # IMPORTANT: keep everything int for this slider
            st.slider(
                "Reduction of contacts (%)",
                min_value=0, max_value=100, step=1,
                value=int(st.session_state[f"{sel}_red"]),
                key=f"{sel}_red",
            )

    # Build dict
    interventions = {}
    for layer in LAYER_NAMES:
        if st.session_state[f"{layer}_en"]:
            interventions[layer] = {
                "start": int(st.session_state[f"{layer}_start"]),
                "end":   int(st.session_state[f"{layer}_end"]),
                "reduction": int(st.session_state[f"{layer}_red"]) / 100.0,  # fraction
            }

    with st.expander("Intervention summary", expanded=True):
        if interventions:
            for k, v in interventions.items():
                st.write(f"**{k}**: days {v['start']}–{v['end']}, reduction {int(v['reduction']*100)}%")
        else:
            st.write("No interventions enabled.")


    # -------- Parameter Overrides --------
    st.markdown("### 🦠 Parameter Overrides")

    override_specs = [
        {"name": "R0", "disp": "$R_0$", "min": 0.0, "max": 20.0, "step": 0.1, "default": float(R0_v)},
        {"name": "infectious_period", "disp": "Infectious period (days)","min": 1.0, "max": 30.0, "step": 0.5, "default": float(infectious_period_v)},
    ]

    # Initialize defaults once
    for spec in override_specs:
        p = spec["name"]
        st.session_state.setdefault(f"{p}_ovr_en", False)
        st.session_state.setdefault(f"{p}_ovr_start", 0)
        st.session_state.setdefault(f"{p}_ovr_end", simulation_days_v)
        st.session_state.setdefault(f"{p}_ovr_value", spec["default"])

    for spec in override_specs:
        p = spec["name"]
        disp = spec["disp"]
        minv, maxv, step = spec["min"], spec["max"], spec["step"]

        with st.expander(f"Override {disp}", expanded=False):
            # Each widget gets a unique key; we pass current state as defaults
            st.checkbox(
                "Enable override",
                key=f"{p}_ovr_en",
                value=st.session_state[f"{p}_ovr_en"],
            )
            c1, c2 = st.columns(2)
            with c1:
                st.number_input(
                    "Start day",
                    min_value=0, max_value=simulation_days_v,
                    value=st.session_state[f"{p}_ovr_start"],
                    step=1,
                    key=f"{p}_ovr_start",
                )
            with c2:
                st.number_input(
                    "End day",
                    min_value=int(st.session_state[f"{p}_ovr_start"]),
                    max_value=simulation_days_v,
                    value=st.session_state[f"{p}_ovr_end"],
                    step=1,
                    key=f"{p}_ovr_end",
                )
            st.slider(
                "Override value",
                min_value=minv, max_value=maxv,
                value=float(st.session_state[f"{p}_ovr_value"]),
                step=step,
                key=f"{p}_ovr_value",
            )

    # Build dict from session_state
    parameter_overrides = {}
    for spec in override_specs:
        p = spec["name"]
        if st.session_state[f"{p}_ovr_en"]:
            parameter_overrides[p] = {
                "start_day": int(st.session_state[f"{p}_ovr_start"]),
                "end_day":   int(st.session_state[f"{p}_ovr_end"]),
                "param":     float(st.session_state[f"{p}_ovr_value"]),
            }

    with st.expander("Overrides summary", expanded=True):
        if parameter_overrides:
            for k, v in parameter_overrides.items():
                label = next(s["disp"] for s in override_specs if s["name"] == k)
                st.write(f"**{label}**: days {v['start_day']}–{v['end_day']}, value {v['param']}")
        else:
            st.write("No overrides enabled.")

    st.markdown("### 📖 About")
    with st.expander("Readme", expanded=False):
        st.markdown("""
                    ### How to use this dashboard

                    This app lets you **configure and run epidemic simulations** with the [Epydemix](https://github.com/epistorm/epydemix) library.  
                    Use the **sidebar** to set inputs, then **Run Simulation** to generate results.

                    ---

                    #### 1) Configure settings (sidebar)
                    - **Simulation Parameters**  
                    Choose the **model** (SIR/SEIR/SIS), **# simulations**, **simulation days**, and **country** (for population + contacts).
                    - **Model Parameters**  
                    Set **$R_0$**, **infectious period** (days), and **incubation period** (SEIR only).
                    - **Initial Conditions**  
                    Set **initial infected** and (for SIR/SEIR) **initial immunity** (%).

                    ---

                    #### 2) Contact Interventions (by layer)
                    For **home / school / work / community**:
                    - **Enable intervention**, pick **start day** and **end day**, and set **reduction of contacts (%)**.
                    - See a compact **summary** of all enabled interventions.

                    ---

                    #### 3) Parameter Overrides (time-windowed)
                    Temporarily override model parameters between two days:
                    - **$R_0$ override** → adjusts the **transmission rate β** only in that window.  
                    - **Infectious period override** → adjusts the **recovery rate μ = 1 / period** only in that window.  
                    For each override: **enable**, set **start/end day**, and choose the **override value**.  
                    A summary lists all active overrides.

                    > Note: Overrides and contact interventions can be combined. Overrides apply only within their day range.

                    ---

                    #### 4) Apply & Run
                    - Click **Apply settings** (bottom of the sidebar) to save sidebar changes without running.
                    - Click **Run Simulation** (top of main panel) to execute the model using the current settings.
                    - If you change settings after a run, you’ll see a prompt to **Run Simulation** again.

                    ---

                    #### 5) Explore results (top navigation)
                    - **Compartments**  
                    Select **compartment** and **age group**. Shows all stochastic trajectories (thin lines) and the **median** (toggle).  
                    Below the plot you’ll find tables:
                    - **Attack rate (%)** (SIR/SEIR): median and 95% CI by age group and total.
                    - **Peak size (absolute)** (with Infected): median and 95% CI.
                    - **Peak time (day)** (with Infected): distribution summary.
                    - **Endemic state** (SIS): median and 95% CI of long-run infected counts.
                    - **Population**  
                    Age distribution as **counts** or **percentages**.
                    - **Contacts**  
                    Contact matrix by **layer** (or overall), with annotated cell values.
                    - **Interventions**  
                    **Contact intensity** (%) over time by layer (overall highlighted).

                    ---

                    #### Tips
                    - For smoother UI, batch changes: adjust settings → **Apply settings** → **Run Simulation**.
                    - Plot controls (e.g., compartment, age group, “Show median”) update **instantly** and don’t re-run the model.
                    - Interventions reduce contacts; **$R_0$ overrides** change **β**, **infectious-period overrides** change **μ**—only inside their chosen window.
                    """)


    apply_cfg = st.form_submit_button("Apply settings")
    if apply_cfg:
        invalidate_results()


# ---------- MAIN ----------
st.title("Epydemix Simulation Dashboard")
run_button = st.button("Run Simulation")

if run_button:
    with st.spinner("Running simulations..."):

        # Load population and contact matrices
        population = load_epydemix_population(country_name)
        C = np.array([population.contact_matrices[layer] for layer in population.contact_matrices])

        # Calculate transmission rate
        spectral_radius = np.linalg.eigvals(C.sum(axis=0)).max()
        mu_v = 1 / infectious_period_v
        beta_v = R0_v * mu_v / spectral_radius

        # Calculate incubation rate if SEIR
        if model_type == "SEIR":
            gamma_v = 1 / incubation_period_v
        else: 
            gamma_v = 0

        # Set end date
        end_date = start_date + timedelta(days=simulation_days_v)

        # Set initial conditions
        if model_type == "SEIR":
            initial_R = (initial_immunity_percentage_v / 100. * population.Nk).astype(int)
            initial_I = np.random.multinomial(int(initial_infected_v / 2), population.Nk / population.Nk.sum())
            initial_E = np.random.multinomial(int(initial_infected_v / 2), population.Nk / population.Nk.sum())
            initial_conditions = {
                "Susceptible": population.Nk - initial_I - initial_E - initial_R,
                "Exposed": initial_E,
                "Infected": initial_I,
                "Recovered": initial_R
            }

        elif model_type == "SIS":
            initial_I = np.random.multinomial(int(initial_infected_v), population.Nk / population.Nk.sum())
            initial_conditions = {
                "Susceptible": population.Nk - initial_I,
                "Infected": initial_I
            }

        elif model_type == "SIR":
            initial_R = (initial_immunity_percentage_v / 100. * population.Nk).astype(int)
            initial_I = np.random.multinomial(int(initial_infected_v), population.Nk / population.Nk.sum())
            initial_conditions = {
                "Susceptible": population.Nk - initial_I - initial_R,
                "Infected": initial_I,
                "Recovered": initial_R
            }

        # Create model
        m = load_predefined_model(model_type, transmission_rate=beta_v, recovery_rate=mu_v, incubation_rate=gamma_v)
        m.set_population(population)

        # Add interventions
        for layer, iv in interventions.items():
            m.add_intervention(
                layer_name=layer,
                start_date=start_date + timedelta(days=iv["start"]),
                end_date=start_date + timedelta(days=iv["end"]),
                reduction_factor=1.0 - iv["reduction"]
            )

        # Add parameter overrides
        for param, ovr in parameter_overrides.items():
            # Compute beta from R0 
            if param == "R0":
                value_ovr = ovr["param"] * mu_v / spectral_radius
                param_name = "transmission_rate"
            elif param == "infectious_period": 
                value_ovr = ovr["param"]
                param_name = "recovery_rate" 
            else: 
                raise ValueError(f"Invalid parameter: {param}")
            
            m.override_parameter(
                parameter_name=param_name,
                start_date=start_date + timedelta(days=ovr["start_day"]),
                end_date=start_date + timedelta(days=ovr["end_day"]),
                value=value_ovr
            )

        # Run simulations and save results
        r = m.run_simulations(Nsim=n_v, start_date=start_date, end_date=end_date, initial_conditions_dict=initial_conditions)
        st.session_state["trajectories"] = r.get_stacked_compartments()
        st.session_state["population"] = population

        simulation_dates = compute_simulation_dates(start_date, end_date)
        m.compute_contact_reductions(simulation_dates)

        # Compute contact intensity in different layers
        rhos = {}
        for layer in LAYER_NAMES:
            rho_0 = np.linalg.eigvals(population.contact_matrices[layer]).max().real
            rhos[layer] = 100 * np.array([np.linalg.eigvals(m.Cs[date][layer]).max().real for date in simulation_dates]) / rho_0
        rhos["overall"] = 100 * np.array([np.linalg.eigvals(m.Cs[date]["overall"]).max().real for date in simulation_dates]) / spectral_radius
        st.session_state["rhos"] = rhos
        st.session_state["ever_ran"] = True
        st.session_state["model"] = m

# ---- VISUALIZATION (ALWAYS RUNS) ----
trj = st.session_state.get("trajectories")
ever_ran = st.session_state.get("ever_ran")

if trj is None:
    if st.session_state.get("ever_ran"):
        st.warning("⚠️ Parameters changed. Click **Run Simulation** to update results.")
    else:
        st.info("👋 Configure settings in the sidebar and click **Run Simulation**.")

else:
    population = st.session_state.get("population")
    rhos = st.session_state.get("rhos")
    m = st.session_state.get("model")

    # Visualization tabs
    tab_labels = ["Summary", "Compartments", "Population", "Contacts", "Interventions"]

    # Initialize default tab in session state
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = tab_labels[0]

    # Radio bound directly to session state
    selected = st.radio(
        "Navigation",
        tab_labels,
        index=tab_labels.index(st.session_state.active_tab),
        horizontal=True,
        key="active_tab"  
    )

    if st.session_state.active_tab == "Summary":
        st.subheader("Run configuration")

        # -- Basic settings table
        base_rows = [
            ("Model", model_type),
            ("Country", country_name),
            ("Simulation days", simulation_days_v),
            ("Simulations (N)", n_v),
        ]
        # model parameters (base / global)
        base_rows.append(("R₀ (base)", f"{R0_v:.2f}"))
        base_rows.append(("Infectious period (base, days)", f"{infectious_period_v:.2f}"))
        if model_type == "SEIR":
            base_rows.append(("Incubation period (base, days)", f"{incubation_period_v:.2f}"))

        #st.table(pd.DataFrame(base_rows, columns=["Setting", "Value"]))
        st.dataframe(pd.DataFrame(base_rows, columns=["Setting", "Value"]), use_container_width=True)

        # -- Interventions table
        st.subheader("Contact interventions")
        if interventions:
            df_iv = pd.DataFrame(
                [
                    {
                        "Layer": layer,
                        "Start day": v["start"],
                        "End day": v["end"],
                        "Reduction (%)": int(round(100 * v["reduction"]))
                    }
                    for layer, v in interventions.items()
                ]
            ).sort_values(["Start day", "Layer"])
            st.dataframe(df_iv, use_container_width=True)
        else:
            st.info("No contact interventions enabled.")

        # -- Parameter overrides table
        st.subheader("Parameter overrides")
        if parameter_overrides:
            # map internal names to display
            disp_map = {
                "R0": "R₀",
                "infectious_period": "Infectious period (days)"
            }
            rows = []
            for pname, spec in parameter_overrides.items():
                rows.append({
                    "Parameter": disp_map[pname],
                    "Start day": spec["start_day"],
                    "End day": spec["end_day"],
                    "Override value": spec["param"]
                })
            df_ovr = pd.DataFrame(rows).sort_values(["Start day", "Parameter"])
            # Nice formatting
            fmt = {"Override value": "{:.2f}"}
            st.dataframe(df_ovr.style.format(fmt), use_container_width=True)
        else:
            st.info("No parameter overrides enabled.")

        # Small footer note
        st.caption(
            "Notes: Interventions scale contacts by layer in their active window. "
            "R₀ overrides adjust the transmission rate, while infectious-period overrides adjust the recovery rate; both apply only within their day range."
        )


    if st.session_state.active_tab == "Compartments":
        #compartment = st.selectbox("Compartment", m.compartments, index=int(np.where(np.array(m.compartments) == "Infected")[0][0]), key="p1_comp")
        #age_group   = st.selectbox("Age group", ["total", "0-4", "5-19", "20-49", "50-64", "65+"], index=0, key="p1_age")
        col1, col2 = st.columns(2)

        with col1:
            compartment = st.selectbox(
                "Compartment",
                m.compartments,
                index=int(np.where(np.array(m.compartments) == "Infected")[0][0]),
                key="p1_comp"
            )

        with col2:
            age_group = st.selectbox(
                "Age group",
                ["total", "0-4", "5-19", "20-49", "50-64", "65+"],
                index=0,
                key="p1_age"
            )

        show_median = st.checkbox("Show median", value=True, key="p1_med")
        plot_compartments_traj(trj, compartment, age_group, show_median)

        if model_type != "SIS":
            # -------- 1) Attack rate (%) --------
            if any(k.startswith("Recovered_") for k in trj.keys()):
                df_attack = compute_attack_rate(trj, population)
                st.subheader("Attack rate (%)")
                st.dataframe(
                    df_attack.style.format({
                        "Median (%)": "{:.1f}",
                        "95% CI low (%)": "{:.1f}",
                        "95% CI high (%)": "{:.1f}",
                    }),
                    use_container_width=True
                )
            else:
                st.info("Attack rate unavailable for this model (no **Recovered** compartment).")

            # -------- 2) Peak size (absolute) --------
            if any(k.startswith("Infected_") for k in trj.keys()):
                df_peak = compute_peak_size(trj, population)
                st.subheader("Peak size (Infected, absolute)")
                st.dataframe(
                    df_peak.style.format({
                        "Median peak": "{:,.0f}",
                        "95% CI low": "{:,.0f}",
                        "95% CI high": "{:,.0f}",
                    }),
                    use_container_width=True
                )
            else:
                st.info("Peak size unavailable (no **Infected** series found).")

            # -------- 3) Peak time (day) --------
            if any(k.startswith("Infected_") for k in trj.keys()):
                df_peaktime = compute_peak_time(trj, population)
                st.subheader("Peak time")
                st.dataframe(df_peaktime, use_container_width=True)
            else: 
                st.info("Peak time unavailable (no **Infected** series found).")
        
        else: 
            # -------- 4) Endemic state (absolute) --------
            if any(k.startswith("Infected_") for k in trj.keys()):
                df_endemic = compute_endemic_state(trj, population)
                st.subheader("Endemic state (Infected, absolute)")

                st.dataframe(df_endemic.style.format({
                        "Median endemic": "{:,.0f}",
                        "95% CI low": "{:,.0f}",
                        "95% CI high": "{:,.0f}",
                    }), use_container_width=True)
            else:
                st.info("Endemic state unavailable (no **Infected** series found).")

    if st.session_state.active_tab == "Population":
        show_percent = st.checkbox("Show percentage", value=False, key="p1_per")
        plot_population(population, show_percent)

    if st.session_state.active_tab == "Contacts":
        contact = st.selectbox("Contact Layer",  ["overall"] + LAYER_NAMES, index=0, key="p2_layer")
        fig, ax = plt.subplots(dpi=600)
        fig.set_facecolor(facecolor)
        plot_contact_matrix(ax, contact, population.contact_matrices, population.Nk_names, "Contacts per day in " + contact)
        st.pyplot(fig)

    if st.session_state.active_tab == "Interventions":
        plot_contact_intensity(rhos)


