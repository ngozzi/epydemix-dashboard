# pages/background.py

import streamlit as st
from layout.header import show_dashboard_header
from layout.sidebar import render_sidebar
from layout.logos import show_logos
from pathlib import Path
from version import __version__


st.set_page_config(
  page_title="Background", 
  page_icon="assets/epydemix-icon.svg", 
  layout="wide", 
  initial_sidebar_state="collapsed"
)

st.markdown("""
        <style>
            .block-container {
                padding-top: 2rem;
                padding-bottom: 5rem;
            }
        </style>
        """, unsafe_allow_html=True)

show_dashboard_header()
render_sidebar()

st.markdown(f"## Background & Documentation <span style='color: gray; font-weight: normal; font-size: 0.6em;'>v{__version__}</span>", unsafe_allow_html=True)

st.markdown("""
This dashboard provides an interactive platform for exploring epidemic scenarios through 
age-structured compartmental models. It is built on the [**epydemix**](https://github.com/epistorm/epydemix) framework, 
a flexible Python package to build, run and analyse stochastic epidemic simulations on real-world demographic and contact pattern data.
""")

# --- Introduction ---
st.header("Overview")

st.markdown("""
### Dashboard Functionality

The EpyScenario Dashboard provides an accessible interface for:

1. **Model configuration**: Select epidemic models, geographies, and initial conditions
2. **Intervention design**: Configure contact reduction measures and vaccination campaigns
3. **Scenario comparison**: Run multiple scenarios and compare outcomes side-by-side
4. **Visualization**: Explore epidemic trajectories, summary metrics, and demographic patterns

The underlying approach uses age-structured stochastic compartmental models to represent
disease transmission, with contact patterns between age groups derived from synthetic social
contact data for over 400 real-world geographies.

### Simulation Engine

Simulations are **stochastic**, implemented via **chain binomial processes**—the paradigm currently
supported by epydemix. This means that each simulation run produces different trajectories due to
random sampling of transmission events. To simplify the user interface, the dashboard displays the
**median across multiple runs**. Future versions will include confidence intervals to better
visualize the uncertainty in epidemic projections.

Users can configure the **time step (Δt)**, which controls the integration step of the simulation:

- **Smaller dt** (e.g., 0.1–0.3): More accurate results, particularly for fast-spreading diseases,
  but slower computation
- **Larger dt** (e.g., 0.5–1.0): Faster simulations, but risk of numerical instability or
  overshooting when transmission rates are high

For most scenarios, the default value provides a good balance between accuracy and performance.
""")

st.divider()

# --- Epidemic Models ---
st.header("Supported Epidemic Models")

st.markdown("""
### Age-Structured Compartmental Models

The underlying approach uses age-structured stochastic compartmental models to represent disease transmission.
Individuals are grouped by age and are represented by a set of compartments, each representing a different disease state.
Additionally, individuals are also divided into demographic groups, each representing a different age category.
The following five age groups are supported:
- **0-4 years**: Children aged 0-4
- **5-19 years**: Children and adolescents aged 5-19
- **20-49 years**: Adults aged 20-49
- **50-64 years**: Adults aged 50-64
- **65+ years**: Elderly aged 65+

Contact patterns between age groups are represented by contact matrices, which are derived from synthetic social contact data for over 400 real-world geographies.
These matrices capture realistic age-mixing patterns. For example, school-age children have high 
contact rates with each other, while elderly individuals have fewer overall contacts.

To know more about the underlying approach, please refer to the [**epydemix paper**](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1013735).
""")

st.markdown("""
### SEIR (Measles)

The **SEIR model** divides the population into four compartments:

- **S (Susceptible)**: Healthy individuals who can contract the disease
- **E (Exposed)**: Infected (not yet infectious) individuals in the latent period
- **I (Infectious)**: Individuals who are infectious and can transmit the disease
- **R (Recovered)**: Individuals who have recovered and gained permanent immunity

""")

svg_path = Path("assets/seir.svg")
st.image(str(svg_path), caption="SEIR Model Compartment Flow", use_container_width=False)


st.markdown("""
The model has three possibile transitions between compartments:

1. **Infection (S → E)**: Susceptible individuals become exposed through contact with infectious individuals. 
   The force of infection depends on the disease transmission rate (i.e., $\\beta$) and the contact patterns between age groups.

2. **Disease progression (E → I)**: Exposed individuals become infectious after the latent period (1/$\\epsilon$).

3. **Recovery (I → R)**: Infectious individuals recover and gain lifelong immunity after the infectious period (1/$\\mu$).

#### Measles Parameterization

The default parameters of the SEIR model are calibrated for measles, a highly contagious viral disease. 
These are the default parameters used in the model:
- **$R_0$**: 12
- **Incubation period**: 11 days
- **Infectious period**: 9 days
- **Background immunity**: 85%

We note that the specified $R_0$ is in absence of background immunity. Users can nonetheless modify the parameters to align the model to other diseases.
""")

st.markdown("""
### SEIRS (Influenza)

The **SEIRS model** extends the SEIR framework by incorporating **waning immunity**, allowing recovered individuals to become susceptible again over time. This better captures diseases where immunity is temporary, such as seasonal influenza.

To realistically model immunity waning, we introduce an intermediate compartment **R₁** between R and S. This two-stage process follows an **Erlang distribution with shape parameter 2**, providing a more realistic delay distribution than simple exponential decay.
""") 

svg_path = Path("assets/seirs.svg")
st.image(str(svg_path), caption="SEIRS Model with Seasonality Compartment Flow", use_container_width=False)

st.markdown("""
The additional transitions beyond the standard SEIR model are:

1. **Waning immunity 1/2 (R → R₁)**: Recovered individuals transition to the intermediate compartment R₁ after an average time of (1/μ_waning)/2.
2. **Waning immunity 2/2 (R₁ → S)**: Individuals in R₁ return to the susceptible compartment S after an additional (1/μ_waning)/2 period.

The total time from R → S averages 1/μ_waning, but the two-stage process creates more realistic, less variable immunity duration.

#### Seasonality

The model incorporates **seasonal forcing** to capture the characteristic winter peaks of respiratory infections. Seasonality is implemented by modulating the transmission rate (β) using a sinusoidal pattern:

- **Seasonality peak day**: Day of the year (1-365) when transmission is highest
- **Seasonality amplitude**: Strength of seasonal variation, specified as a discrete level:
  - **Strong**: Large seasonal variation (e.g., strong winter peaks)
  - **Moderate**: Noticeable seasonal pattern
  - **Medium**: Moderate seasonal effect (default for influenza)
  - **Low**: Mild seasonal variation
  - **None**: No seasonality (constant transmission year-round)

**Important:** The specified R₀ represents the **baseline reproductive number at the seasonal peak**, assuming no background immunity or interventions. The effective R₀ varies throughout the year based on the seasonal forcing.

#### Influenza Parameterization

The default parameters are calibrated for seasonal influenza:

- **R₀**: 2.0 (at seasonal peak without background immunity)
- **Incubation period**: 1.5 days
- **Infectious period**: 1.5 days  
- **Waning immunity**: 365 days (~1 year)
- **Seasonality peak**: Day 125 (with respect to the start date of the simulation)
- **Seasonality amplitude**: Medium
- **Background immunity**: 15%

These parameters can be adjusted to model other respiratory pathogens with temporary immunity, such as RSV or endemic coronaviruses.
""")


st.markdown("""
### SEIHR (COVID-19)
The **SEIHR model** extends the SEIRS framework by incorporating **hospitalization**, allowing infected individuals to be hospitalized and recover from the disease.

The additional transitions beyond the standard SEIRS model are:

1. **Hospitalization (I → H)**: Infectious individuals transition to the hospitalized compartment H depending on age-specific probability of hospitalization ($p_H$).
2. **Recovery from hospitalization (H → R)**: Individuals in H return to the recovered compartment R after the duration of hospital stay ($1/\mu_H$).


""")

svg_path = Path("assets/seihr.svg")
st.image(str(svg_path), caption="SEIHR Model with Hospitalization Compartment Flow", use_container_width=False)

st.markdown("""
#### COVID-19 Parameterization

The default parameters are calibrated for COVID-19:

- **R₀**: 2.5
- **Incubation period**: 3 days
- **Infectious period**: 2.5 days
- **Length of hospital stay**: 5 days
- **Probability of hospitalization given infection by age group**: 0.2%, 0.5%, 1.5%, 5%, 18% 

We stress that the probability of hospitalization is age-dependent and can be specified by the user.
Users can also specify the length of hospital stay, or change the value of other parameters to align the model to other diseases.
""")

st.divider()

# --- Contact Interventions ---
st.header("Contact Interventions")

st.markdown("""
### How Contact Interventions Work

Contact interventions model **non-pharmaceutical interventions (NPIs)** that reduce person-to-person 
contact rates, such as social distancing, lockdowns, or facility closures.

#### Contact Matrices

Models use contact matrices that describe the average number of contacts 
between different age groups in four different settings:

- **Home**: Household contacts
- **School**: Educational institution contacts  
- **Work**: Workplace contacts
- **Community**: Other social contacts (shopping, recreation, etc.)

Assuming $K$ age groups, each contact matrix is a $K \\times K$ matrix. The element
$C^s_{ij}$ represents the average number of contacts between an individual in age group $i$ with 
individuals in age group $j$ in setting $s$. 
Overall contacts are the sum of the contacts in all four settings, i.e., $C = \sum_{s=1}^4 C^s$.

#### Defining Interventions

Users can configure interventions by specifying:

1. **Start date**: When the intervention begins (day of simulation)
2. **End date**: When the intervention ends
3. **Contact layer**: Which setting is affected (home, school, work, or community)
4. **Reduction percentage**: How much contacts are reduced (0-100%)

During the intervention period, the contact matrix for the specified layer is modified:
```
C_modified^s = C^s_original × (1 - reduction/100)
```

In other words, the contact matrix is scaled down by the reduction percentage.

Users can define **multiple interventions** that may overlap in time. This allows modeling complex 
intervention scenarios, such as school closures, work-from-home policies, social distancing, and lockdowns.
""")

st.divider()

# --- Vaccinations ---
st.header("Vaccination Campaigns")

st.markdown("""
### How Vaccination Works

Vaccination is implemented with an **all-or-nothing** approach applied to susceptible individuals, 
meaning each vaccine dose either provides full protection or no protection.
Vaccinated individuals are moved to the vaccinated compartment (V) and are completely removed from the disease dynamics.

#### Campaign Parameters

Users define vaccination campaigns by specifying:

1. **Start date**: When the campaign begins
2. **End date**: When the campaign ends  
3. **Target age groups**: Which age groups are targeted by the campaign
4. **Target coverage**: Percentage of population in target age groups to vaccinate (%)
5. **Vaccine efficacy (VE)**: Probability that a dose provides protection (0-100%)
6. **Rollout shape**: How doses are distributed over time

We stress that doses can be allocated differently across age groups, enabling
age-targeted vaccination campaigns, risk-based strategies, and realistic rollout policies.

#### Dose Calculation

The model converts coverage targets into daily dose allocations:

**Step 1: Total doses needed**
```
Total doses = (Target coverage / 100) × Target population
```

Note: Coverage is calculated on the **target population**, not accounting for existing immunity. 

**Step 2: Effective doses**

Since vaccine efficacy < 100%, not all administered doses provide protection:
```
Effective doses = Total doses × (VE / 100)
```

This implements the probabilistic nature of vaccine protection in our framework.

**Step 3: Daily allocation**

Doses are distributed across the campaign period according to the rollout shape.

#### Rollout Shapes

**Flat rollout** (default):
- Constant number of doses administered each day
- Formula: `Daily doses = Effective doses / Campaign duration`

**Ramp-up rollout**:
- Linear increase in daily doses during initial ramp-up period
- Models realistic capacity constraints (e.g., training vaccinators, supply chain setup)
- After ramp-up, maintains maximum capacity
- Formula during ramp-up: `Daily doses = (Day number / Ramp-up period) × Maximum daily capacity`

#### Advanced Features

By default, vaccination doses are administered only to **susceptible individuals** (S compartment). 
However, users can optionally specify additional compartments eligible for vaccination, such as 
exposed (E) or recovered (R) individuals. This flexibility enables modeling of real-world mass 
vaccination campaigns where doses are distributed broadly without screening for infection status 
or prior immunity through serological testing.

**Important consideration:** Restricting vaccination to susceptible individuals represents an 
optimistic scenario, as it assumes perfect targeting with no wasted doses on already-infected 
or immune individuals. In contrast, allowing vaccination across all compartments more realistically 
reflects campaigns where some doses are inevitably administered to individuals who cannot benefit 
from them, resulting in lower effective coverage for the same number of doses administered.
""")


# --- Footer ---
st.markdown("""
---
### Learn More

- **epydemix GitHub repository**: https://github.com/epistorm/epydemix
- **Source code of the dashboard**: https://github.com/ngozzi/epydemix-dashboard

### Questions or Feedback?

If you have questions about the models or suggestions for improvements, please contact us at epydemix@isi.it.
""")

st.markdown("""
---
### Citation

If you use this dashboard in your research, please cite:

```
@article{gozzi2025epydemix,
  title={Epydemix: An open-source Python package for epidemic modeling with integrated approximate Bayesian calibration},
  author={Gozzi, Nicol{\'o} and Chinazzi, Matteo and Davis, Jessica T and Gioannini, Corrado and Rossi, Luca and Ajelli, Marco and Perra, Nicola and Vespignani, Alessandro},
  journal={PLOS Computational Biology},
  volume={21},
  number={11},
  pages={e1013735},
  year={2025},
  publisher={Public Library of Science San Francisco, CA USA}
}
```
""")

st.divider()
show_logos()