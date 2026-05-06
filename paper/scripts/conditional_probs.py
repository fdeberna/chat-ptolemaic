import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------
# Models
# -----------------------------------

models = ["Base", "QLoRA-500", "QLoRA-1000"]

# -----------------------------------
# Colorblind-friendly muted palette
# -----------------------------------

frame_colors = {
    "Premodern only": "#4C78A8",   # muted blue
    "Modern only": "#F58518",      # muted orange
    "Both": "#B279A2",             # muted purple
    "Neither": "#BAB0AC",          # soft gray
}

stance_colors = {
    "Geocentric": "#4C78A8",
    "Heliocentric": "#F58518",
    "Ambiguous / other": "#BAB0AC",
}

# -----------------------------------
# Frame distribution
# -----------------------------------

frame_data = {
    "Base": {
        "Premodern only": 0.308929 - 0.042857,
        "Modern only": 0.296429 - 0.042857,
        "Both": 0.042857,
        "Neither": 0.437500,
    },
    "QLoRA-500": {
        "Premodern only": 0.644643 - 0.026786,
        "Modern only": 0.055357 - 0.026786,
        "Both": 0.026786,
        "Neither": 0.326786,
    },
    "QLoRA-1000": {
        "Premodern only": 0.653571 - 0.012500,
        "Modern only": 0.026786 - 0.012500,
        "Both": 0.012500,
        "Neither": 0.332143,
    },
}

# -----------------------------------
# Conditional stance within Premodern
# -----------------------------------

stance_given_pre = {
    "Base": {
        "Geocentric": 0.242775,
        "Heliocentric": 0.127168,
        "Ambiguous / other": 0.630058,
    },
    "QLoRA-500": {
        "Geocentric": 0.227147,
        "Heliocentric": 0.038781,
        "Ambiguous / other": 0.734072,
    },
    "QLoRA-1000": {
        "Geocentric": 0.234973,
        "Heliocentric": 0.021858,
        "Ambiguous / other": 0.743169,
    },
}

# -----------------------------------
# Plot helper
# -----------------------------------

def plot_stacked(ax, data, categories, colors, title, ylabel):

    x = np.arange(len(models))
    bottom = np.zeros(len(models))

    for cat in categories:

        vals = np.array([data[m][cat] for m in models]) * 100

        bars = ax.bar(
            x,
            vals,
            bottom=bottom,
            label=cat,
            color=colors[cat],
            edgecolor="white",
            linewidth=0.6,
        )

        # Add labels only for sufficiently large regions
        for i, v in enumerate(vals):
            if v >= 8:
                ax.text(
                    i,
                    bottom[i] + v / 2,
                    f"{v:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                )

        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylim(0, 100)

    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)

    # Cleaner aesthetics
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Put legend outside plot
    ax.legend(
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

# -----------------------------------
# Create figure
# -----------------------------------

fig, axes = plt.subplots(
    1,
    2,
    figsize=(11, 4.5),
    constrained_layout=True
)

# Left panel
plot_stacked(
    axes[0],
    frame_data,
    ["Premodern only", "Modern only", "Both", "Neither"],
    frame_colors,
    "Explanatory frame distribution",
    "Share of generations (%)",
)

# Right panel
plot_stacked(
    axes[1],
    stance_given_pre,
    ["Geocentric", "Heliocentric", "Ambiguous / other"],
    stance_colors,
    "Stance distribution within Premodern frame",
    "Conditional share (%)",
)

# fig.suptitle(
#     "QLoRA shifts frame distribution more than within-frame stance",
#     fontsize=13,
# )

# -----------------------------------
# Save
# -----------------------------------

plt.savefig(
    "frame_stance_stacked_bars.pdf",
    bbox_inches="tight"
)

plt.savefig(
    "frame_stance_stacked_bars.png",
    dpi=300,
    bbox_inches="tight"
)

# plt.show()