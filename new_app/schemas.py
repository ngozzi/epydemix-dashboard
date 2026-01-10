# schemas.py

MODEL_COMPS = {"SEIR (Measles)": ["S", "E", "I", "R", "V"], 
               "SEIR (Influenza)": ["S", "E", "I", "R", "V"]}

MODEL_PARAM_SCHEMAS = {
    
    "SEIR (Measles)": [
        {
            "key": "R0",
            "label": "R0",
            "type": "float",
            "min": 0.1,
            "max": 18.0,
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
            "label": "R0",
            "type": "float",
            "min": 0.1,
            "max": 18.0,
            "step": 0.1,
            "default": 2.0,
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
        {
            "key": "waning_immunity_period",
            "label": "Waning immunity period (days)",
            "type": "float",
            "min": 0.5,
            "max": 1000.0,
            "step": 0.5,
            "default": 365.0,
        }
    ],
}


INITIAL_CONDITION_DEFAULTS = {
    "SEIR (Measles)": {"infected_pct": 0.1, "immune_pct": 85.0},
    "SEIRS (Influenza)": {"infected_pct": 0.01, "immune_pct": 10.0},
}
