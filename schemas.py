# schemas.py

MODEL_COMPS = {"SEIR (Measles)": ["S", "E", "I", "R", "V"], 
               "SEIRS (Influenza)": ["S", "E", "I", "R", "R1", "V"], 
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
    "SEIHR (COVID-19)": {"infected_pct": 0.1, "immune_pct": 25.0},
}
