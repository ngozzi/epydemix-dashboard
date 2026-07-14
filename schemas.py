# schemas.py

MODEL_COMPS = {"SEIR (Measles)": ["S", "E", "I", "R", "V"],
               "SEIRS (Influenza)": ["S", "E", "I", "R", "R1", "V"],
               # Pertussis: 8-compartment SEIRS with a parallel partial-immunity
               # track (naive S->E->I->R and partial Sp->Ep->Ip->Rp). Vaccination
               # routes S->Sp, so no separate V compartment is used.
               "SEIRS (Pertussis)": ["S", "E", "I", "R", "Sp", "Ep", "Ip", "Rp"],
               "SEIHR (COVID-19)": ["S", "E", "I", "H", "R", "V"]}

MODEL_PARAM_SCHEMAS = {
    
    "SEIR (Measles)": [
        {
            "key": "R0",
            "label": "$R_0$",
            "type": "float",
            "min": 0.1,
            "max": 20.0,
            "step": 0.1,
            "default": 12.0,
        },
        {
            "key": "incubation_period",
            "label": "Incubation period (days)",
            "type": "float",
            "min": 0.5,
            "max": 30.0,
            "step": 0.5,
            "default": 11.0,
        },
        {
            "key": "infectious_period",
            "label": "Infectious period (days)",
            "type": "float",
            "min": 0.5,
            "max": 30.0,
            "step": 0.5,
            "default": 9.0,
        },
    ],
    "SEIRS (Influenza)": [
        {
            "key": "R0",
            "label": "$R_0$",
            "type": "float",
            "min": 0.1,
            "max": 20.0,
            "step": 0.1,
            "default": 1.5,
        },
        {
            "key": "incubation_period",
            "label": "Incubation period (days)",
            "type": "float",
            "min": 0.5,
            "max": 20.0,
            "step": 0.5,
            "default": 1.5,
        },
        {
            "key": "infectious_period",
            "label": "Infectious period (days)",
            "type": "float",
            "min": 0.5,
            "max": 20.0,
            "step": 0.5,
            "default": 1.5,
        }, 
        {
            "key": "waning_immunity_period",
            "label": "Waning immunity period (days)",
            "type": "float",
            "min": 5.0,
            "max": 1000.0,
            "step": 5.0,
            "default": 365.0,
        },
        {
            "key": "seasonality_peak_day",
            "label": "Seasonality peak day (day of the year)",
            "type": "float",
            "min": 1,
            "max": 365,
            "step": 1,
            "default": 125,
        },
        {
            "key": "seasonality_amplitude",
            "label": "Seasonality",
            "type": "discrete",
            "options": ["Strong", "Moderate", "Medium", "Low", "None"],
            "default": "Medium",
        }
    ],
    # 8-compartment SEIRS with partial-immunity track (Wearing & Rohani 2009).
    # R0 is defined on the naive track (beta / gamma_naive). The partial track
    # is governed by sigma (relative infectiousness of Ip) and delta (relative
    # susceptibility of Sp).
    "SEIRS (Pertussis)": [
        {
            "key": "R0",
            "label": "$R_0$",
            "type": "float",
            "min": 0.1,
            "max": 20.0,
            "step": 0.1,
            "default": 12.0,
        },
        {
            "key": "incubation_period",
            "label": "Incubation period (days)",
            "type": "float",
            "min": 0.5,
            "max": 30.0,
            "step": 0.5,
            "default": 9.0,
        },
        {
            "key": "infectious_period",
            "label": "Infectious period — naive $I$ (days)",
            "type": "float",
            "min": 0.5,
            "max": 40.0,
            "step": 0.5,
            "default": 21.0,
        },
        {
            "key": "infectious_period_partial",
            "label": "Infectious period — partial $I_p$ (days)",
            "type": "float",
            "min": 0.5,
            "max": 40.0,
            "step": 0.5,
            "default": 10.0,
        },
        {
            "key": "rel_infectiousness_partial",
            "label": r"Relative infectiousness of $I_p$ ($\sigma$)",
            "type": "float",
            "min": 0.0,
            "max": 1.0,
            "step": 0.05,
            "default": 0.2,
        },
        {
            "key": "rel_susceptibility_partial",
            "label": r"Relative susceptibility of $S_p$ ($\delta$)",
            "type": "float",
            "min": 0.0,
            "max": 1.0,
            "step": 0.05,
            "default": 0.3,
        },
        {
            "key": "waning_full_to_partial_years",
            "label": r"Waning full$\rightarrow$partial ($\omega_1$, years)",
            "type": "float",
            "min": 0.5,
            "max": 30.0,
            "step": 0.5,
            "default": 4.0,
        },
        {
            "key": "waning_partial_to_susceptible_years",
            "label": r"Waning partial$\rightarrow$susceptible ($\omega_2$, years)",
            "type": "float",
            "min": 1.0,
            "max": 50.0,
            "step": 1.0,
            "default": 15.0,
        },
        {
            "key": "seasonality_peak_day",
            "label": "Seasonality peak day (day of the year)",
            "type": "float",
            "min": 1,
            "max": 365,
            "step": 1,
            "default": 240,
        },
        {
            "key": "seasonality_amplitude",
            "label": "Seasonality",
            "type": "discrete",
            "options": ["Strong", "Moderate", "Medium", "Low", "None"],
            "default": "Low",
        }
    ],
    "SEIHR (COVID-19)": [
        {
            "key": "R0",
            "label": "$R_0$",
            "type": "float",
            "min": 0.1,
            "max": 20.0,
            "step": 0.1,
            "default": 2.5,
        },
        {
            "key": "incubation_period",
            "label": "Incubation period (days)",
            "type": "float",
            "min": 0.5,
            "max": 20.0,
            "step": 0.5,
            "default": 3.0,
        },
        {
            "key": "infectious_period",
            "label": "Infectious period (days)",
            "type": "float",
            "min": 0.5,
            "max": 20.0,
            "step": 0.5,
            "default": 2.5,
        },
        {
           "key": "hospital_stay",
           "label": "Hospital stay (days)",
           "type": "float",
           "min": 0.,
           "max": 25.0,
           "step": 1.0,
           "default": 5.0, 
        },  
        {
            "key": "ph",
            "label": "Probability of hospitalization (%)",
            "type": "by_age_float",
            "min": 0.,
            "max": 100.0,
            "step": 0.1,
            "default": [0.2, 0.5, 1.5, 5., 18.],
        }
    ]
}


INITIAL_CONDITION_DEFAULTS = {
    "SEIR (Measles)": {"infected_pct": 0.1, "immune_pct": 85.0},
    "SEIRS (Influenza)": {"infected_pct": 0.1, "immune_pct": 25.0},
    # "Background immunity" seeds the partial-immunity pool (Sp/Rp/R). At an
    # endemic start most of a vaccinated population carries partial immunity.
    # partial_infection_pct: share of seeded infections in the partial track (Ep/Ip).
    # partial_immune_pct:    Sp share of the immunity pool (remainder split Rp/R).
    "SEIRS (Pertussis)": {
        "infected_pct": 0.1,
        "immune_pct": 85.0,
        "partial_infection_pct": 33.0,
        "partial_immune_pct": 71.0,
    },
    "SEIHR (COVID-19)": {"infected_pct": 0.1, "immune_pct": 25.0},
}
