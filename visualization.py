import numpy as np
import seaborn as sns


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


def plot_contact_intensity(ax, rho, facecolor="#0c1019", linecolor="#50f0d8"):
    """Plot the contact intensity"""
    ax.set_facecolor(facecolor)
    ax.plot(range(len(rho)), rho, color=linecolor, linewidth=2)
    ax.set_xlabel("Days", color="white")
    ax.set_ylabel("Contact Intensity (%)", color="white")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(width=0, colors="white")
    ax.grid(axis="y", linestyle="dotted", alpha=0.5, linewidth=0.5)
    

