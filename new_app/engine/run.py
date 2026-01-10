# engine/run.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Any
import numpy as np
import pandas as pd
from epydemix.model import EpiModel
from constants import START_DATE, N_SIM, DEFAULT_AGE_GROUPS
from datetime import timedelta


def create_vaccination_rate_function(eligible_compartments):
    """
    Generator function that creates a vaccination rate computation function.
    
    Args:
        eligible_compartments: list of compartment names (e.g., ["S", "R"]) that are 
                              eligible for vaccination and contribute to the denominator
    
    Returns:
        A function that computes vaccination rates based on the specified eligible compartments
    """
    
    def compute_vaccination_rate(params, data):
        """ 
        Compute the vaccination rate.

        Args:
            params: list of parameters for this transition, first element is the total number of doses for a given day
            data: dictionary containing the population, the compartments, and other information about the system

        Returns:
            np.array of vaccination rates for each age group
        """
        # Get total doses for today for each age group
        total_doses = params[0][data["t"]]

        # Compute the total eligible population (sum of all specified compartments)
        eligible_pop = sum(data["pop"][data["comp_indices"][comp]] 
                          for comp in eligible_compartments)
        
        # Compute the fraction of susceptible population w.r.t eligible population
        fraction_S = data["pop"][data["comp_indices"]["S"]] / eligible_pop
        effective_doses = total_doses * fraction_S

        # Compute the rate of vaccination for each age group
        # (edge case: more doses than S individuals -> rate_vax ~ 0.999)
        rate_vax = []
        for i in range(len(effective_doses)):
            if effective_doses[i] < data["pop"][data["comp_indices"]["S"]][i]: 
                rate_vax.append(effective_doses[i] / data["pop"][data["comp_indices"]["S"]][i])
            else: 
                rate_vax.append(0.999)

        return np.array(rate_vax)
    
    return compute_vaccination_rate


def create_initial_conditions(model, Nk, infected_pct, immune_pct): 
    if model == "SEIR (Measles)":
        # initialize
        ic = {
            "S": np.zeros_like(Nk), 
            "E": np.zeros_like(Nk), 
            "I": np.zeros_like(Nk), 
            "R": np.zeros_like(Nk)
            }
        
        # infected 
        total_infected = Nk * (infected_pct / 100.)
        ic["I"] = (total_infected / 2).astype(int)
        ic["E"] = (total_infected / 2).astype(int)

        # background immunity
        ic["R"] = (Nk * (immune_pct / 100.)).astype(int)

        # remaining susceptible
        ic["S"] = Nk - ic["I"] - ic["E"] - ic["R"]

        return ic

    elif model == "SEIRS (Influenza)":
        return None
    else:
        raise ValueError(f"Model {model} not supported")


def compute_beta(model, R0, C, params): 
    if model == "SEIR (Measles)":
        return R0 * (1 / params["infectious_period"]) / np.linalg.eigvals(C.sum(axis=0)).real.max()
    elif model == "SEIRS (Influenza)":
        return R0 * (1 / params["infectious_period"]) / np.linalg.eigvals(C.sum(axis=0)).real.max()
    else:
        raise ValueError(f"Model {model} not supported")
    

def run_seir_stub(scenario: dict) -> pd.DataFrame:
    sim_length = int(scenario.get("sim_length", 250))
    age_groups = DEFAULT_AGE_GROUPS

    # Build model
    model = EpiModel(compartments=["S", "E", "I", "R", "V"])
    model.add_transition("S", "E", params=("beta", "I"), kind="mediated")
    model.add_transition("E", "I", params=("gamma"), kind="spontaneous")
    model.add_transition("I", "R", params=("mu"), kind="spontaneous")

    # Add population 
    model.set_population(scenario["population"])

    vax_rate_function = create_vaccination_rate_function(scenario["vaccination_settings"]["target_compartments"])
    model.register_transition_kind("vaccination", vax_rate_function)
    model.add_transition("S", "V", params=(scenario["daily_doses_by_age"][age_groups].values,), kind="vaccination")

    # Set parameters
    C = np.array([scenario["population"].contact_matrices[layer] for layer in scenario["population"].contact_matrices])
    model.add_parameter(
        parameters_dict={
            "beta": compute_beta(scenario["model"], scenario["model_params"]["R0"], C, scenario["model_params"]), 
            "gamma": 1. / scenario["model_params"]["incubation_period"],
            "mu": 1. / scenario["model_params"]["infectious_period"],
        }
    )

    # Initial conditions (switch function)
    ic = create_initial_conditions(
        scenario["model"], 
        model.population.Nk, 
        scenario["initial_conditions"]["infected_pct"], 
        scenario["initial_conditions"]["immune_pct"],
        )

    # Apply Contact interventions
    for intervention in scenario["contact_interventions"]:
        model.add_intervention(
                layer_name=intervention["layer"],
                start_date=START_DATE + timedelta(days=intervention["start_day"]),
                end_date=START_DATE + timedelta(days=intervention["end_day"]),
                reduction_factor=1.0 - (intervention["reduction_pct"] / 100.)
            )

    # Run Simulations 
    results = model.run_simulations(
            Nsim=N_SIM,
            start_date=START_DATE,
            end_date=START_DATE + timedelta(days=sim_length),
            initial_conditions_dict=ic
        )

    # Format Output
    df_median = results.get_quantiles_compartments(quantiles=[0.5])
    df_median["t"] = np.arange(sim_length+1, dtype=int)
    df_median.drop(columns=["quantile", "date"], inplace=True)

    return df_median


MODEL_RUNNERS: dict[str, Callable[..., pd.DataFrame]] = {
    "SEIR (Measles)": run_seir_stub,
}


def run_scenario(scenario: dict) -> pd.DataFrame:
    model = scenario.get("model")
    if model not in MODEL_RUNNERS:
        raise ValueError(f"No runner registered for model={model!r}")

    return MODEL_RUNNERS[model](scenario)
