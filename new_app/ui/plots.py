import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px

def plot_contact_matrix(
    layer, matrices, groups, facecolor="#0c1019", cmap="copper"
):
    """Altair heatmap of a contact matrix with per-cell annotations."""

    # --- choose matrix (layer or overall) ---
    if layer == "overall":
        matrix = np.sum(np.stack(list(matrices.values()), axis=0), axis=0)
    else:
        matrix = np.asarray(matrices[layer])

    # --- tidy dataframe ---
    n = len(groups)
    assert matrix.shape == (n, n), "Matrix shape must match groups length."
    df = pd.DataFrame(
        [{"Contacting": groups[i], "Contacted": groups[j], "Value": float(matrix[i, j])}
         for i in range(n) for j in range(n)]
    )

    color_scale = alt.Scale(scheme=cmap) if isinstance(cmap, str) else alt.Scale(range=cmap)

    # Domain from data range
    vmin, vmax = float(df["Value"].min()), float(df["Value"].max())
    color_scale.domain = [vmin, vmax]

    # --- base heatmap with cell strokes to emulate grid ---
    base = alt.Chart(df).properties(
        width="container", height=550
    ).encode(
        x=alt.X("Contacted:N",
                sort=groups,
                axis=alt.Axis(
                    title="Age Group (contacted)",
                    labelColor="white", titleColor="white"
                )),
        y=alt.Y("Contacting:N",
                sort=groups[::-1],
                axis=alt.Axis(
                    title="Age Group (contacting)",
                    labelColor="white", titleColor="white"
                )),
    )

    heat = base.mark_rect(
        stroke="white", strokeWidth=0.5  # grid lines between cells
    ).encode(
        color=alt.Color("Value:Q", scale=color_scale, legend=None),
        tooltip=[
            alt.Tooltip("Contacting:N"),
            alt.Tooltip("Contacted:N"),
            alt.Tooltip("Value:Q", format=".2f"),
        ],
    )

    # --- annotation layer (values in cells) ---
    text = base.mark_text(
        color="white", fontSize=15
    ).encode(
        text=alt.Text("Value:Q", format=".2f")
    )

    chart = (heat + text).configure_axis(
        grid=False,           # no default grid (we draw cell strokes instead)
        domain=False,         # remove axis spines
        tickOpacity=0.0,      # hide ticks
    ).configure_view(
        strokeWidth=0         # no outer border
    ).configure(
        background=facecolor  # dark background
    )

    st.altair_chart(chart, use_container_width=True)

def plot_population(
    population, facecolor="#0c1019", palette=None
):
    """Plot population distribution with Plotly."""
    df = pd.DataFrame({
        "Age Group": population.Nk_names,
        "Count": population.Nk,
    })

    if palette is None:
        palette = ["#FFAE42", "#E07A2D", "#C65A1E", "#8C4A2F", "#3A2A1F"]


    fig = px.pie(
        df,
        names="Age Group",
        values="Count",
        hole=0.6,
        color="Age Group",
        color_discrete_sequence=palette,
    )

    fig.update_traces(
        textinfo="percent+label",
        textfont_color="#e5e7eb",
         textfont_size=16,
        hovertemplate="<b>%{label}</b><br>Percent: %{percent:.1%}<br>Count: %{value:,}<extra></extra>",
    )

    fig.update_layout(
        showlegend=False,
        paper_bgcolor=facecolor,
        plot_bgcolor=facecolor,
        uniformtext_minsize=12, 
        uniformtext_mode="hide",
        margin=dict(l=0, r=0, t=0, b=0),
    )

    st.plotly_chart(fig, use_container_width=True)
