import numpy as np
import pandas as pd
import seaborn as sns
import altair as alt
import streamlit as st

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

def plot_population(ax, population, show_percent=False, facecolor="#0c1019", linecolor="#50f0d8"):
    """Plot the population distribution"""
    ax.set_facecolor(facecolor)
    if show_percent:
        ax.bar(population.Nk_names, 100 * population.Nk / population.Nk.sum(), color=linecolor, zorder=1)
        ax.set_ylabel("Individuals (%)", color="white")
    else:
        ax.bar(population.Nk_names, population.Nk, color=linecolor, zorder=1)
        ax.set_ylabel("Individuals (total)", color="white")
    ax.set_xlabel("Age Group", color="white")
    
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(width=0, colors="white")
    ax.grid(axis="y", linestyle="dotted", alpha=0.5, linewidth=0.5, zorder=0)


def plot_contact_intensity(ax, rhos, facecolor="#0c1019", linecolor="#50f0d8"):
    """Plot the contact intensity"""
    ax.set_facecolor(facecolor)

    for layer, rho in rhos.items():
        ax.plot(range(len(rho)), 
                rho, 
                color=linecolor if layer == "overall" else "grey", 
                label=layer if layer == "overall" else None,
                linewidth=2, 
                alpha = 1.0 if layer == "overall" else 0.5)
        
    # annotate the name of the layers avoiding overlaps of the text
    for layer, rho in rhos.items():
        ax.text(len(rho) - 1, rho[-1], layer, ha="right", va="bottom", color="white", fontsize=6)

    #ax.legend(facecolor=facecolor, labelcolor="white", frameon=False)
    ax.set_xlabel("Days", color="white")
    ax.set_ylabel("Contact Intensity (%)", color="white")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(width=0, colors="white")
    ax.grid(axis="y", linestyle="dotted", alpha=0.5, linewidth=0.5)


def plot_contact_intensity_native(rhos: dict, facecolor="#0c1019"):
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
        "#ff7f0e",  # home
        "#1f77b4",  # school
        "#2ca02c",  # work
        "#d62728"  # community
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
        height=350,
        background=facecolor,
    )

    # ---- thicker cyan line for overall
    chart = base.mark_line().encode(
        size=alt.condition(
            alt.datum.Layer == "overall", alt.value(3.0), alt.value(1.6)
        )
    )

    # ---- style tweaks: dotted horizontal grid, no vertical grid, no spines
    chart = chart.configure_axis(
        grid=True,
        gridColor="white",
        gridOpacity=0.2,
        gridWidth=0.5,
        gridDash=[2, 4],       # dotted
        domain=False,          # remove axis lines (spines)
        tickColor="white",
        tickOpacity=0.0,       # hide tick marks
    ).configure_view(
        strokeWidth=0          # remove outer border
    )

    st.altair_chart(chart, use_container_width=True)
