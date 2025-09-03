import numpy as np
import pandas as pd
import seaborn as sns
import altair as alt
import streamlit as st

import pandas as pd
import numpy as np
import altair as alt
import streamlit as st

def plot_compartments_traj_altair(
    trj, comp, age, show_median=True, facecolor="#0c1019", linecolor="#50f0d8"
):
    key = f"{comp}_{age}"
    if key not in trj:
        st.warning(f"Missing: {key}")
        return

    series = np.asarray(trj[key])  # (Nsim, T)
    if series.ndim != 2 or series.size == 0:
        st.warning(f"Invalid series for {key}")
        return

    Nsim, T = series.shape
    df = pd.DataFrame({
        "Day": np.tile(np.arange(T), Nsim),
        "Simulation": np.repeat(np.arange(Nsim), T).astype(str),
        "Value": series.reshape(-1).astype(float),
    })

    # Base encodings (turn OFF vertical grid on X here)
    base = alt.Chart(df).encode(
        x=alt.X(
            "Day:Q",
            axis=alt.Axis(title="Time", labelColor="white", titleColor="white", grid=False)  # no vertical grid
        ),
        y=alt.Y(
            "Value:Q",
            axis=alt.Axis(title=f"{comp} ({age})", labelColor="white", titleColor="white")  # horizontal grid handled globally
        ),
    ).properties(height=450)

    # All trajectories: thin, semi-transparent white
    traj = base.mark_line(strokeWidth=0.7, opacity=0.10, color="white").encode(
        detail="Simulation:N"
    )

    layers = [traj]

    # Median computed inside Altair so layering is safe
    if show_median:
        median_line = (
            base.transform_aggregate(Median="median(Value)", groupby=["Day"])
                .mark_line(strokeWidth=2, color=linecolor)
                .encode(y="Median:Q")
        )
        layers.append(median_line)

    chart = (
        alt.layer(*layers)
        # Match your other plots: dotted horizontal grid, no spines/border
        .configure_axis(
            grid=True,
            gridColor="white",
            gridOpacity=0.4,
            gridDash=[2, 4],   # dotted horizontal grid
            domain=False,      # remove axis spines
            tickColor="white",
            tickOpacity=0.0    # hide tick marks
        )
        .configure_view(strokeWidth=0)       # remove outer border
        .configure(background=facecolor)     # dark background
    )

    st.altair_chart(chart, use_container_width=True)



def plot_compartments_traj(ax, trj, comp, age, show_median=True, facecolor="#0c1019", linecolor="#50f0d8"):
    """Plot the trajectory of a compartment over time"""
    # guard: key might not exist
    key = f"{comp}_{age}"
    if key not in trj:
        ax.text(0.5, 0.5, f"Missing: {key}", ha="center", va="center", color="white")
        return
    series = trj[key]
    T = range(len(series[0]))
    ax.set_facecolor(facecolor)
    for i in range(len(series)):
        ax.plot(T, series[i], color="white", alpha=0.1, linewidth=0.7)
    if show_median:
        import numpy as np
        ax.plot(T, np.median(series, axis=0), label="Median", color=linecolor)
        ax.legend(facecolor=facecolor, labelcolor="white", frameon=False)
    ax.set_xlabel("Time", color="white")
    ax.set_ylabel(comp + " (" + age + ")", color="white")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(width=0, colors="white")
    ax.grid(axis="y", linestyle="dotted", alpha=0.5, linewidth=0.5)


def plot_contact_matrix(ax, layer, matrices, groups, title, facecolor="#0c1019", cmap="mako"):
    """Plot the contact matrix"""

    if layer == "overall": 
        matrix = np.array([matrices[layer] for layer in matrices]).sum(axis=0)
    else:
        matrix = matrices[layer]

    ax.set_facecolor(facecolor)
    ax.imshow(matrix, origin="lower", aspect="equal", cmap=sns.color_palette(cmap, as_cmap=True))
    # annotate the values
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="white", fontsize=6)

    ax.set_xlabel("Age Group (contacted)", color="white", fontsize=6)
    ax.set_ylabel("Age Group (contacting)", color="white", fontsize=6)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(width=0, colors="white")

    # Add grid to better separate cells
    ax.set_xticks(np.arange(len(groups)))
    ax.set_yticks(np.arange(len(groups)))
    ax.set_xticklabels(groups, ha='center', rotation=45, fontsize=6)
    ax.set_yticklabels(groups, fontsize=6)
    ax.set_xticks(np.arange(-.5, len(groups), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(groups), 1), minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=0.5)
    ax.set_title(title, color="white", fontsize=10)


def plot_population(population, show_percent=False, facecolor="#0c1019", linecolor="#50f0d8"):
    """Plot population distribution with Altair."""
    df = pd.DataFrame({
        "Age Group": population.Nk_names,
        "Count": population.Nk
    })
    if show_percent:
        df["Value"] = 100 * df["Count"] / df["Count"].sum()
        ylabel = "Individuals (%)"
    else:
        df["Value"] = df["Count"]
        ylabel = "Individuals (total)"

    chart = (
        alt.Chart(df)
        .mark_bar(color=linecolor)
        .encode(
            x=alt.X("Age Group:N", axis=alt.Axis(title="Age Group", labelColor="white", titleColor="white")),
            y=alt.Y("Value:Q", axis=alt.Axis(title=ylabel, labelColor="white", titleColor="white")),
            tooltip=["Age Group", alt.Tooltip("Value", format=".1f")]
        )
        .properties(height=450, background=facecolor)
    )

    st.altair_chart(chart, use_container_width=True)


def plot_contact_intensity(rhos: dict, facecolor="#0c1019"):
    """
    rhos: dict like {"overall": [...], "home":[...], "school":[...], "work":[...], "community":[...]}
    """

    # ---- tidy data
    rows = []
    for layer, rho in rhos.items():
        for day, val in enumerate(rho):
            rows.append({"Day": day, "Layer": layer, "Value": float(val)})
    df = pd.DataFrame(rows)

    # ---- colors (overall = cyan, others = nice palette)
    layers = list(rhos.keys())
    if "overall" in layers:
        layers = ["overall"] + [l for l in layers if l != "overall"]

    nice_palette = [
        "#50f0d8",  # overall (cyan)
        "#344b47",  # home
        "#97b1ab",  # school
        "#84caff",  # work
        "#4395e3"  # community
    ]
    color_scale = alt.Scale(domain=layers, range=nice_palette[:len(layers)])

    legend_cols = min(len(layers), 5)

    base = alt.Chart(df).encode(
        x=alt.X("Day:Q", axis=alt.Axis(title="Days", labelColor="white", titleColor="white", grid=False)),
        y=alt.Y("Value:Q", axis=alt.Axis(title="Contact Intensity (%)", labelColor="white", titleColor="white")),
        color=alt.Color(
            "Layer:N",
            scale=color_scale,
            legend=alt.Legend(
                title=None,
                orient="bottom",
                direction="horizontal",
                columns=legend_cols,
                labelColor="white",
            ),
        ),
        tooltip=[alt.Tooltip("Layer:N"), alt.Tooltip("Day:Q"), alt.Tooltip("Value:Q", format=".2f")],
    ).properties(
        height=450,
        background=facecolor,
    )

    # ---- thicker cyan line for overall
    chart = base.mark_line().encode(
        size=alt.condition(
            alt.datum.Layer == "overall", alt.value(3.0), alt.value(1.6)
        )
    )

    st.altair_chart(chart, use_container_width=True)
