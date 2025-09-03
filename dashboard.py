import streamlit as st
import matplotlib.pyplot as plt
from epydemix.model import load_predefined_model
from epydemix.population import load_epydemix_population
from epydemix.utils import compute_simulation_dates
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from visualization import plot_compartments_traj, plot_contact_matrix, plot_contact_intensity, plot_population, plot_compartments_traj_altair
from utils import invalidate_results, load_locations
from compute_statistics import compute_attack_rate, compute_peak_size, compute_peak_time, compute_endemic_state

# ---------- LAYOUT ----------
st.sidebar.image("https://cdn.prod.website-files.com/67bde9057c9d78157874e100/67c1d1122127f0a9ce202197_epydemix-logo-p-500.png", use_container_width=True)
st.markdown(
    """
    <style>
    .fixed-logo {
        position: fixed;
        bottom: 60px;   /* was 20px — push it up */
        right: 20px;
        width: 100px;
        z-index: 100;
    }
    </style>
    <img src="https://www.isi.it/wp-content/uploads/2023/11/isi-logo-white.svg" class="fixed-logo">
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
    
    # Parameter overrides
    with st.expander("Parameter Overrides", expanded=False):
        # TODO: add parameter overrides
        pass

    # Interventions
    st.markdown("### Interventions")
    
    # Hold all layer states in session_state so switching layer keeps values
    for layer in LAYER_NAMES:
        st.session_state.setdefault(f"{layer}_en", False)
        st.session_state.setdefault(f"{layer}_start", 0)
        st.session_state.setdefault(f"{layer}_end", simulation_days_v)
        st.session_state.setdefault(f"{layer}_red", 0)

    sel = st.selectbox("Select a contact layer to edit", LAYER_NAMES, index=0)

    with st.expander(f"Configure: {sel}", expanded=True):
        st.session_state[f"{sel}_en"]  = st.checkbox(f"Enable intervention on {sel}", value=st.session_state[f"{sel}_en"])
        st.session_state[f"{sel}_start"] = st.number_input("Start day", 0, simulation_days_v, st.session_state[f"{sel}_start"], 1, key=f"in_{sel}_start")
        st.session_state[f"{sel}_end"]   = st.number_input("End day", int(st.session_state[f"{sel}_start"]), simulation_days_v, st.session_state[f"{sel}_end"], 1, key=f"in_{sel}_end")
        st.session_state[f"{sel}_red"]   = st.slider("Reduction of contacts (%)", 0, 100, st.session_state[f"{sel}_red"], 1, key=f"in_{sel}_red")

    # build dict from session_state
    interventions = {}
    for layer in LAYER_NAMES:
        if st.session_state[f"{layer}_en"]:
            interventions[layer] = {
                "start": int(st.session_state[f"{layer}_start"]),
                "end": int(st.session_state[f"{layer}_end"]),
                "reduction": st.session_state[f"{layer}_red"] / 100.0,
            }

    with st.expander("Intervention summary", expanded=True):
        if interventions:
            for k, v in interventions.items():
                st.write(f"**{k}**: days {v['start']}–{v['end']}, reduction {int(v['reduction']*100)}%")
        else:
            st.write("No interventions enabled.")

    with st.expander("Instructions", expanded=False):
        st.markdown("""
        ### How to use this dashboard

        This dashboard lets you **configure and run epidemic simulations** with the [Epydemix](https://github.com/epistorm/epydemix) library.

        **1. Configure settings in the sidebar**
        - Open the expanders to set **Simulation Parameters**, **Model Parameters**, and **Initial Conditions**.
        - Under **Interventions**, you can add measures that reduce contacts in specific layers (home, school, work, community).
        - For each intervention, choose:
        - **Start day**: when the intervention begins.
        - **End day**: when it ends.
        - **Reduction (%)**: the percentage reduction of contacts in that layer.

        **2. Apply settings**
        - After adjusting parameters, click **Apply settings** at the bottom of the sidebar.
        - This saves your configuration but does **not** yet run the simulation.

        **3. Run the simulation**
        - Click **Run Simulation** (below the title in the main panel) to start the model with your chosen settings.
        - Progress will be shown with a spinner, and results are stored for visualization.

        **4. Explore results**
        - Use the tabs in the main panel:
        - **Compartments**: epidemic curves by compartment and age group, with optional median.
        - **Population**: age distribution of the selected population, absolute or percentage.
        - **Contacts**: visualizations of contact matrices by layer.
        - **Interventions**: intensity of contacts over time, showing the effect of interventions.

        ---
        **Tips:**
        - Change parameters in the sidebar → click **Apply settings** → rerun with **Run Simulation**.
        - Plot-specific controls (compartment, age group, show median, etc.) update instantly without re-running the model.
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
    tab_labels = ["Compartments", "Population", "Contacts", "Interventions"]

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

    if st.session_state.active_tab == "Compartments":
        compartment = st.selectbox("Compartment", m.compartments, index=int(np.where(np.array(m.compartments) == "Infected")[0][0]), key="p1_comp")
        age_group   = st.selectbox("Age group", ["total", "0-4", "5-19", "20-49", "50-64", "65+"], index=0, key="p1_age")
        show_median = st.checkbox("Show median", value=True, key="p1_med")
        #fig, ax = plt.subplots(dpi=600)
        #fig.set_facecolor(facecolor)
        #plot_compartments_traj(ax, trj, compartment, age_group, show_median)
        #st.pyplot(fig)
        plot_compartments_traj_altair(trj, compartment, age_group, show_median)

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
        #fig, ax = plt.subplots(dpi=300)
        #fig.set_facecolor(facecolor)
        show_percent = st.checkbox("Show percentage", value=False, key="p1_per")
        #plot_population(ax, population, show_percent)
        #st.pyplot(fig)
        plot_population(population, show_percent)

    if st.session_state.active_tab == "Contacts":
        contact = st.selectbox("Contact Layer",  ["overall"] + LAYER_NAMES, index=0, key="p2_layer")
        fig, ax = plt.subplots(dpi=600)
        fig.set_facecolor(facecolor)
        plot_contact_matrix(ax, contact, population.contact_matrices, population.Nk_names, "Contacts per day in " + contact)
        st.pyplot(fig)

    if st.session_state.active_tab == "Interventions":
        plot_contact_intensity(rhos)


