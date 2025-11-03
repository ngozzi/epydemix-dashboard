import streamlit as st
import numpy as np
from datetime import timedelta
from epydemix.population import load_epydemix_population
from epydemix.utils import compute_simulation_dates
from .config_engine import (
    eval_derived, build_epimodel_from_config, 
    compute_override_value
)

def build_run_config() -> dict:
    get = st.session_state.get

    # Prefer UI keys if present, else fall back
    country_name = get("ui_country", get("country_name"))
    model_config = get("ui_model_config", get("model_config"))
    param_values = get("ui_param_values", get("param_values"))
    interventions = get("ui_interventions", get("interventions"))
    parameter_overrides = get("ui_parameter_overrides", get("parameter_overrides"))
    initial_conditions = get("ui_initial_conditions", get("initial_conditions"))
    n_sims = get("ui_n_sims", get("n_sims"))
    sim_days = get("ui_sim_days", get("sim_days"))

    return {
        "country_name": country_name,
        "model_config": model_config,
        "param_values": param_values,
        "interventions": interventions,
        "parameter_overrides": parameter_overrides,
        "initial_conditions": initial_conditions,
        "n_sims": int(n_sims) if n_sims is not None else None,
        "sim_days": int(sim_days) if sim_days is not None else None,
    }

def convert_initial_conditions_to_arrays(initial_conditions_pct, population, compartments):
    """
    Convert percentage-based initial conditions to array-based format for EpiModel.
    
    Args:
        initial_conditions_pct: Dict with compartment names as keys and percentages as values
        population: Population object with Nk (age group sizes)
        compartments: List of compartment names
    
    Returns:
        Dict with compartment names as keys and numpy arrays as values
    """
    initial_conditions_dict = {}
    
    # Get total population size
    total_population = population.Nk.sum()
    
    for compartment in compartments:
        if compartment in initial_conditions_pct:
            # Convert percentage to absolute number
            pct = initial_conditions_pct[compartment] / 100.0
            total_compartment_pop = int(total_population * pct)
            
            # Distribute across age groups proportionally to age group sizes
            age_group_proportions = population.Nk / population.Nk.sum()
            compartment_by_age = (total_compartment_pop * age_group_proportions).astype(int)
            
            # Ensure we don't exceed population in any age group
            compartment_by_age = np.minimum(compartment_by_age, population.Nk)
            
            initial_conditions_dict[compartment] = compartment_by_age
        else:
            # Default to zero if compartment not specified
            initial_conditions_dict[compartment] = np.zeros(len(population.Nk), dtype=int)
    
    return initial_conditions_dict

def run_simulation(run_cfg, start_date):
    """Run the epidemic simulation with current configuration."""
    with st.spinner("Running simulations..."):
        # 1. Load population and contact matrices
        population = load_epydemix_population(run_cfg["country_name"])
        
        # 2. Compute spectral radius for transmission rate calculation
        C = np.array([population.contact_matrices[layer] for layer in population.contact_matrices])
        spectral_radius = np.linalg.eigvals(C.sum(axis=0)).real.max()
        
        # 3. Build derived parameters from user inputs
        context = {"spectral_radius": spectral_radius}
        derived = eval_derived(
            run_cfg["model_config"].derived_parameters,
            run_cfg["param_values"],
            context
        )
        
        # 4. Build EpiModel from configuration
        model = build_epimodel_from_config(
            run_cfg["model_config"],
            derived,
            population
        )
        
        # 5. Apply interventions
        for layer, intervention in run_cfg["interventions"].items():
            model.add_intervention(
                layer_name=layer,
                start_date=start_date + timedelta(days=intervention["start"]),
                end_date=start_date + timedelta(days=intervention["end"]),
                reduction_factor=1.0 - intervention["reduction"]
            )
        
        # 6. Apply parameter overrides
        for param_name, override in run_cfg["parameter_overrides"].items():
            engine_param, override_value = compute_override_value(
                param_name, override["param"], 
                run_cfg["model_config"], 
                run_cfg["param_values"], 
                context
            )
            model.override_parameter(
                parameter_name=engine_param,
                start_date=start_date + timedelta(days=override["start_day"]),
                end_date=start_date + timedelta(days=override["end_day"]),
                value=override_value
            )
        
        # 7. Convert initial conditions to proper format
        initial_conditions_dict = convert_initial_conditions_to_arrays(
            run_cfg["initial_conditions"],
            population,
            run_cfg["model_config"].compartments
        )
        
        # 8. Run simulations
        Nsim = int(run_cfg["n_sims"])
        sim_days = int(run_cfg["sim_days"])
        end_date = start_date + timedelta(days=sim_days)
        results = model.run_simulations(
            Nsim=Nsim,
            start_date=start_date,
            end_date=end_date,
            initial_conditions_dict=initial_conditions_dict
        )
        
        # 9. Store results in session state
        simulation_output = {
            "simulation_results": results,
            "population": population,
            "model": model,
            "simulation_dates": compute_simulation_dates(start_date, end_date)
        }
        
        return simulation_output
        # 10. Compute additional metrics
        #compute_contact_intensities(model, population, st.session_state["simulation_dates"])

def validate_simulation_config():
    """Validate that all required configuration is present."""
    # Implementation here
    pass

def compute_contact_intensities(model, population, simulation_dates):
    """Compute contact intensities for intervention visualization."""
    # Implementation here
    pass