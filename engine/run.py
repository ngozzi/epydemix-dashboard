# engine/run.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Any
import numpy as np
import pandas as pd
from epydemix.model import EpiModel
from epydemix.utils import convert_to_2Darray, compute_simulation_dates
from constants import START_DATE, N_SIM, DEFAULT_AGE_GROUPS, LAYER_NAMES
from datetime import timedelta, datetime

SEASONALITY_OPTIONS = {
    "Strong": 0.5,
    "Moderate": 0.65,
    "Medium": 0.75,
    "Weak": 0.85,
    "Low": 0.9,
    "None": 1.0,
}


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


def create_initial_conditions(model, Nk, infected_pct, immune_pct,
                              partial_infection_pct=33.0, partial_immune_pct=71.0):
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
        # initialize
        ic = {
            "S": np.zeros_like(Nk), 
            "E": np.zeros_like(Nk), 
            "I": np.zeros_like(Nk), 
            "R": np.zeros_like(Nk), 
            "R1": np.zeros_like(Nk)
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

    elif model == "SEIRS (Pertussis)":
        # 8-compartment partial-immunity model. Seed infections across both the
        # naive (E/I) and partial (Ep/Ip) tracks, and distribute the background
        # immunity pool across Sp/Rp/R using endemic-like proportions
        # (most vaccinated individuals sit in the partially-immune Sp pool).
        ic = {c: np.zeros_like(Nk) for c in ["S", "E", "I", "R", "Sp", "Ep", "Ip", "Rp"]}

        # infected: split between naive and partial tracks (user-controlled),
        # each track split evenly between exposed and infectious
        total_infected = Nk * (infected_pct / 100.)
        partial_frac = partial_infection_pct / 100.
        naive_inf = total_infected * (1. - partial_frac)
        partial_inf = total_infected * partial_frac
        ic["E"] = (naive_inf / 2).astype(int)
        ic["I"] = (naive_inf / 2).astype(int)
        ic["Ep"] = (partial_inf / 2).astype(int)
        ic["Ip"] = (partial_inf / 2).astype(int)

        # background immunity pool -> Sp (partial susceptible), Rp, R.
        # Sp share is user-controlled; the remainder keeps the Rp:R ratio (0.24:0.05).
        immune = Nk * (immune_pct / 100.)
        sp_share = partial_immune_pct / 100.
        remainder = max(0., 1. - sp_share)
        rp_share = remainder * (0.24 / 0.29)
        r_share = remainder * (0.05 / 0.29)
        ic["Sp"] = (immune * sp_share).astype(int)
        ic["Rp"] = (immune * rp_share).astype(int)
        ic["R"] = (immune * r_share).astype(int)

        # remaining fully-susceptible naive individuals
        ic["S"] = (Nk - ic["E"] - ic["I"] - ic["Ep"] - ic["Ip"]
                   - ic["Sp"] - ic["Rp"] - ic["R"])

        return ic

    elif model == "SEIHR (COVID-19)":
        # initialize
        ic = {
            "S": np.zeros_like(Nk), 
            "E": np.zeros_like(Nk), 
            "I": np.zeros_like(Nk), 
            "R": np.zeros_like(Nk), 
            "H": np.zeros_like(Nk)
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

    else:
        raise ValueError(f"Model {model} not supported")


def compute_beta(model, R0, C, params): 
    if model in ["SEIR (Measles)", "SEIRS (Influenza)", "SEIRS (Pertussis)", "SEIHR (COVID-19)"]:
        return R0 * (1 / params["infectious_period"]) / np.linalg.eigvals(C.sum(axis=0)).real.max()
    else:
        raise ValueError(f"Model {model} not supported")


def compute_seasonality_factor(start_date, end_date, days_to_max, seasonality_min, seasonality_max=1, dt=1.):
    """Compute seasonal modulation factors over a date range."""
    seasonality_factors = []
    start_date, end_date = pd.to_datetime(start_date), pd.to_datetime(end_date)
    day_max = start_date + timedelta(days=days_to_max)
    simulation_dates = compute_simulation_dates(start_date, end_date, dt=dt)

    for day in simulation_dates:
        day = pd.Timestamp(day)
        day_max_yearly = datetime(day.year, day_max.month, day_max.day)  # Peak seasonality date
        s_r = seasonality_min / seasonality_max
        factor = 0.5 * ((1 - s_r) * np.sin(2 * np.pi / 365 * (day - day_max_yearly).days + 0.5 * np.pi) + 1 + s_r)
        seasonality_factors.append(factor)

    return np.array(seasonality_factors)


def run_seihr_stub(scenario: dict) -> pd.DataFrame:
    sim_length = int(scenario.get("sim_length", 250))
    dt = float(scenario.get("time_step", 0.2))
    age_groups = DEFAULT_AGE_GROUPS

    # Build model
    model = EpiModel(compartments=["S", "E", "I", "R", "H", "V"])
    model.add_transition("S", "E", params=("beta", "I"), kind="mediated")
    model.add_transition("E", "I", params=("gamma"), kind="spontaneous")
    model.add_transition("I", "R", params=("mu * (1 - ph)"), kind="spontaneous")
    model.add_transition("I", "H", params=("mu * ph"), kind="spontaneous")
    model.add_transition("H", "R", params=("mu_H"), kind="spontaneous")

    # Add population 
    model.set_population(scenario["population"])

    # Add vaccination rate function
    vax_rate_function = create_vaccination_rate_function(scenario["vaccination_settings"]["target_compartments"])
    model.register_transition_kind("vaccination", vax_rate_function)
    model.add_transition("S", "V", params=(scenario["daily_doses_by_age"][age_groups].values,), kind="vaccination")

    # Set parameters
    C = np.array([scenario["population"].contact_matrices[layer] for layer in scenario["population"].contact_matrices])
    beta = compute_beta(scenario["model"], scenario["model_params"]["R0"], C, scenario["model_params"])

    # pH 
    pH = []
    for i in range(len(age_groups)):
        pH.append(scenario["model_params"][f"ph_{i}"] / 100.)
    pH = convert_to_2Darray(pH)

    model.add_parameter(
        parameters_dict={
            "beta": beta, 
            "gamma": 1. / scenario["model_params"]["incubation_period"],
            "mu": 1. / scenario["model_params"]["infectious_period"],
            "mu_H": 1. / scenario["model_params"]["hospital_stay"],
            "ph": pH,
        }
    )

    # Initial conditions
    ic = create_initial_conditions(
        scenario["model"], 
        model.population.Nk, 
        scenario["initial_conditions"]["infected_pct"], 
        scenario["initial_conditions"]["immune_pct"],
    )

    # Apply Contact interventions
    for intervention in scenario["contact_interventions"]:
        if intervention["layer"] == "all":
            for layer in LAYER_NAMES:
                model.add_intervention(
                    layer_name=layer,
                    start_date=START_DATE + timedelta(days=intervention["start_day"]),
                    end_date=START_DATE + timedelta(days=intervention["end_day"]),
                    reduction_factor=1.0 - (intervention["reduction_pct"] / 100.)
                )
        else:
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
            initial_conditions_dict=ic,
            dt=dt,
        )

    # Format Output (compartments and transitions)
    df_median_comp = results.get_quantiles_compartments(quantiles=[0.5])
    df_median_comp["t"] = np.arange(len(df_median_comp), dtype=int) + 1
    df_median_comp.drop(columns=["quantile", "date"], inplace=True)

    df_median_trans = results.get_quantiles_transitions(quantiles=[0.5])
    df_median_trans["t"] = np.arange(len(df_median_trans), dtype=int) + 1
    df_median_trans.drop(columns=["quantile", "date"], inplace=True)

    return df_median_comp, df_median_trans


def run_seirs_stub(scenario: dict) -> pd.DataFrame:
    sim_length = int(scenario.get("sim_length", 250))
    dt = float(scenario.get("time_step", 0.2))
    age_groups = DEFAULT_AGE_GROUPS

    # Build model
    model = EpiModel(compartments=["S", "E", "I", "R", "R1", "V"])
    model.add_transition("S", "E", params=("beta", "I"), kind="mediated")
    model.add_transition("E", "I", params=("gamma"), kind="spontaneous")
    model.add_transition("I", "R", params=("mu"), kind="spontaneous")
    model.add_transition("R", "R1", params=("mu_waning*2"), kind="spontaneous")
    model.add_transition("R1", "S", params=("mu_waning*2"), kind="spontaneous")

    # Add population 
    model.set_population(scenario["population"])

    vax_rate_function = create_vaccination_rate_function(scenario["vaccination_settings"]["target_compartments"])
    model.register_transition_kind("vaccination", vax_rate_function)
    model.add_transition("S", "V", params=(scenario["daily_doses_by_age"][age_groups].values,), kind="vaccination")

    # Set parameters
    C = np.array([scenario["population"].contact_matrices[layer] for layer in scenario["population"].contact_matrices])
    beta = compute_beta(scenario["model"], scenario["model_params"]["R0"], C, scenario["model_params"])
    seasonality_factor = compute_seasonality_factor(
        START_DATE, 
        START_DATE + timedelta(days=sim_length), 
        scenario["model_params"]["seasonality_peak_day"], 
        SEASONALITY_OPTIONS[scenario["model_params"]["seasonality_amplitude"]], 
        dt=dt
    )
    model.add_parameter(
        parameters_dict={
            "beta": beta * seasonality_factor, 
            "gamma": 1. / scenario["model_params"]["incubation_period"],
            "mu": 1. / scenario["model_params"]["infectious_period"],
            "mu_waning": 1. / scenario["model_params"]["waning_immunity_period"],
        }
    )

    # Initial conditions
    ic = create_initial_conditions(
        scenario["model"], 
        model.population.Nk, 
        scenario["initial_conditions"]["infected_pct"], 
        scenario["initial_conditions"]["immune_pct"],
    )

    # Apply Contact interventions
    for intervention in scenario["contact_interventions"]:
        if intervention["layer"] == "all":
            for layer in LAYER_NAMES:
                model.add_intervention(
                    layer_name=layer,
                    start_date=START_DATE + timedelta(days=intervention["start_day"]),
                    end_date=START_DATE + timedelta(days=intervention["end_day"]),
                    reduction_factor=1.0 - (intervention["reduction_pct"] / 100.)
                )
        else:
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
            initial_conditions_dict=ic, 
            dt=dt,
        )

    # Format Output (compartments and transitions)
    df_median_comp = results.get_quantiles_compartments(quantiles=[0.5])
    df_median_comp["t"] = np.arange(len(df_median_comp), dtype=int) + 1
    df_median_comp.drop(columns=["quantile", "date"], inplace=True)

    df_median_trans = results.get_quantiles_transitions(quantiles=[0.5])
    df_median_trans["t"] = np.arange(len(df_median_trans), dtype=int) + 1
    df_median_trans.drop(columns=["quantile", "date"], inplace=True)

    return df_median_comp, df_median_trans


def run_seir_stub(scenario: dict) -> pd.DataFrame:
    sim_length = int(scenario.get("sim_length", 250))
    dt = float(scenario.get("time_step", 0.2))
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
        if intervention["layer"] == "all":
            for layer in LAYER_NAMES:
                model.add_intervention(
                    layer_name=layer,
                    start_date=START_DATE + timedelta(days=intervention["start_day"]),
                    end_date=START_DATE + timedelta(days=intervention["end_day"]),
                    reduction_factor=1.0 - (intervention["reduction_pct"] / 100.)
                )
        else:
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
            initial_conditions_dict=ic, 
            dt=dt,
        )

    # Format Output (compartments and transitions)
    df_median_comp = results.get_quantiles_compartments(quantiles=[0.5])
    df_median_comp["t"] = np.arange(len(df_median_comp), dtype=int) + 1
    df_median_comp.drop(columns=["quantile", "date"], inplace=True)

    df_median_trans = results.get_quantiles_transitions(quantiles=[0.5])
    df_median_trans["t"] = np.arange(len(df_median_trans), dtype=int) + 1
    df_median_trans.drop(columns=["quantile", "date"], inplace=True)

    return df_median_comp, df_median_trans


def run_pertussis_stub(scenario: dict) -> pd.DataFrame:
    """
    Age-structured pertussis model: 8-compartment SEIRS with a parallel
    partial-immunity track (Wearing & Rohani 2009).

        Naive track    : S  -> E  -> I  -> R
        Partial track  : Sp -> Ep -> Ip -> Rp
        Cross-links    : S -> Sp (vaccination), R -> Sp (omega1),
                         Rp -> S (omega2)

    Force of infection is driven by both I and Ip: beta * (I + sigma * Ip).
    Because epydemix mediated transitions take a single mediating compartment,
    the combined force of infection is split into one transition per source
    compartment. Partial susceptibles (Sp) are infected at reduced rate delta.
    R0 is defined on the naive track (beta / gamma_naive).
    """
    sim_length = int(scenario.get("sim_length", 250))
    dt = float(scenario.get("time_step", 0.2))
    age_groups = DEFAULT_AGE_GROUPS
    mp = scenario["model_params"]

    # Build model (no V compartment; vaccination routes S -> Sp)
    model = EpiModel(compartments=["S", "E", "I", "R", "Sp", "Ep", "Ip", "Rp"])

    # Force of infection on naive susceptibles: beta * (I + sigma * Ip)
    model.add_transition("S", "E", params=("beta", "I"), kind="mediated")
    model.add_transition("S", "E", params=("beta_sigma", "Ip"), kind="mediated")

    # Force of infection on partial susceptibles: delta * beta * (I + sigma * Ip)
    model.add_transition("Sp", "Ep", params=("beta_delta", "I"), kind="mediated")
    model.add_transition("Sp", "Ep", params=("beta_delta_sigma", "Ip"), kind="mediated")

    # Incubation (latent -> infectious)
    model.add_transition("E", "I", params=("gamma"), kind="spontaneous")
    model.add_transition("Ep", "Ip", params=("gamma"), kind="spontaneous")

    # Recovery (partial track recovers faster: milder, shorter illness)
    model.add_transition("I", "R", params=("mu"), kind="spontaneous")
    model.add_transition("Ip", "Rp", params=("mu_p"), kind="spontaneous")

    # Waning immunity
    model.add_transition("R", "Sp", params=("omega1"), kind="spontaneous")
    model.add_transition("Rp", "S", params=("omega2"), kind="spontaneous")

    # Add population
    model.set_population(scenario["population"])

    # Vaccination moves susceptibles into the partially-immune pool: S -> Sp
    vax_rate_function = create_vaccination_rate_function(scenario["vaccination_settings"]["target_compartments"])
    model.register_transition_kind("vaccination", vax_rate_function)
    model.add_transition("S", "Sp", params=(scenario["daily_doses_by_age"][age_groups].values,), kind="vaccination")

    # Set parameters
    C = np.array([scenario["population"].contact_matrices[layer] for layer in scenario["population"].contact_matrices])
    beta = compute_beta(scenario["model"], mp["R0"], C, mp)
    seasonality_factor = compute_seasonality_factor(
        START_DATE,
        START_DATE + timedelta(days=sim_length),
        mp["seasonality_peak_day"],
        SEASONALITY_OPTIONS[mp["seasonality_amplitude"]],
        dt=dt,
    )
    beta_t = beta * seasonality_factor
    sigma = mp["rel_infectiousness_partial"]
    delta = mp["rel_susceptibility_partial"]

    model.add_parameter(
        parameters_dict={
            "beta": beta_t,
            "beta_sigma": beta_t * sigma,
            "beta_delta": beta_t * delta,
            "beta_delta_sigma": beta_t * delta * sigma,
            "gamma": 1. / mp["incubation_period"],
            "mu": 1. / mp["infectious_period"],
            "mu_p": 1. / mp["infectious_period_partial"],
            "omega1": 1. / (mp["waning_full_to_partial_years"] * 365.),
            "omega2": 1. / (mp["waning_partial_to_susceptible_years"] * 365.),
        }
    )

    # Initial conditions
    ic = create_initial_conditions(
        scenario["model"],
        model.population.Nk,
        scenario["initial_conditions"]["infected_pct"],
        scenario["initial_conditions"]["immune_pct"],
        partial_infection_pct=scenario["initial_conditions"].get("partial_infection_pct", 33.0),
        partial_immune_pct=scenario["initial_conditions"].get("partial_immune_pct", 71.0),
    )

    # Apply Contact interventions
    for intervention in scenario["contact_interventions"]:
        if intervention["layer"] == "all":
            for layer in LAYER_NAMES:
                model.add_intervention(
                    layer_name=layer,
                    start_date=START_DATE + timedelta(days=intervention["start_day"]),
                    end_date=START_DATE + timedelta(days=intervention["end_day"]),
                    reduction_factor=1.0 - (intervention["reduction_pct"] / 100.)
                )
        else:
            model.add_intervention(
                    layer_name=intervention["layer"],
                    start_date=START_DATE + timedelta(days=intervention["start_day"]),
                    end_date=START_DATE + timedelta(days=intervention["end_day"]),
                    reduction_factor=1.0 - (intervention["reduction_pct"] / 100.)
                )

    # Run Simulations
    results = model.run_simulations(
            Nsim=N_SIM,
            start_date=START_DATE,
            end_date=START_DATE + timedelta(days=sim_length),
            initial_conditions_dict=ic,
            dt=dt,
        )

    # Format Output (compartments and transitions)
    df_median_comp = results.get_quantiles_compartments(quantiles=[0.5])
    df_median_comp["t"] = np.arange(len(df_median_comp), dtype=int) + 1
    df_median_comp.drop(columns=["quantile", "date"], inplace=True)

    df_median_trans = results.get_quantiles_transitions(quantiles=[0.5])
    df_median_trans["t"] = np.arange(len(df_median_trans), dtype=int) + 1
    df_median_trans.drop(columns=["quantile", "date"], inplace=True)

    return df_median_comp, df_median_trans


MODEL_RUNNERS: dict[str, Callable[..., pd.DataFrame]] = {
    "SEIR (Measles)": run_seir_stub,
    "SEIRS (Influenza)": run_seirs_stub,
    "SEIRS (Pertussis)": run_pertussis_stub,
    "SEIHR (COVID-19)": run_seihr_stub,
}


def run_scenario(scenario: dict) -> pd.DataFrame:
    model = scenario.get("model")
    if model not in MODEL_RUNNERS:
        raise ValueError(f"No runner registered for model={model!r}")

    return MODEL_RUNNERS[model](scenario)
