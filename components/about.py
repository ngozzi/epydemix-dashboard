def get_about_message():
    """Return the About message content for the dashboard."""
    return """
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
    - If you change settings after a run, you'll see a prompt to **Run Simulation** again.

    ---

    #### 5) Explore results (top navigation)
    - **Compartments**  
    Select **compartment** and **age group**. Shows all stochastic trajectories (thin lines) and the **median** (toggle).  
    Below the plot you'll find tables:
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
    - Plot controls (e.g., compartment, age group, "Show median") update **instantly** and don't re-run the model.
    - Interventions reduce contacts; **$R_0$ overrides** change **β**, **infectious-period overrides** change **μ**—only inside their chosen window.
    """
