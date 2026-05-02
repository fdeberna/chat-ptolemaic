import pandas as pd
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------
# Style: must come BEFORE creating fig/ax
# ------------------------------------------------------------

plt.style.use("default")
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "black",
    "font.size": 12,
    "axes.titlesize": 24,
    "axes.labelsize": 18,
    "legend.fontsize": 14,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "grid.alpha": 0.5,
    "grid.linestyle": "--",
})

# ------------------------------------------------------------
# Data
# ------------------------------------------------------------

data = [
    ["A", "it seems", 0.0167],
    ["A", "according to", 0.0673],
    ["A", "it would seem", 0.0167],
    ["A", "any hedge phrase", 0.1161],
    ["B", "it seems", 0.0595],
    ["B", "according to", 0.1101],
    ["B", "it would seem", 0.0321],
    ["B", "any hedge phrase", 0.2024],
]

df = pd.DataFrame(data, columns=["model", "phrase", "rate"])

pivot_df = df.pivot(index="phrase", columns="model", values="rate")

phrase_order = [
    "it seems",
    "according to",
    "it would seem",
    "any hedge phrase",
]
pivot_df = pivot_df.loc[phrase_order]

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

x = np.arange(len(pivot_df))
width = 0.38

fig, ax = plt.subplots(figsize=(12, 8))

ax.set_axisbelow(True)  # grid behind bars

ax.bar(
    x - width / 2,
    pivot_df["A"],
    width,
    label="Model A (general language only)",
)

ax.bar(
    x + width / 2,
    pivot_df["B"],
    width,
    label="Model B (astronomy fine-tuned)",
)

# ------------------------------------------------------------
# Labels and formatting
# ------------------------------------------------------------

ax.set_ylabel("Frequency")
ax.set_xlabel("Hedging phrase")
ax.set_title("Frequency of Scholastic Hedging Expressions")

ax.set_xticks(x)
ax.set_xticklabels(pivot_df.index, rotation=35, ha="right")

ax.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda y, _: f"{100*y:.0f}%")
)

ax.grid(axis="y", linestyle="--", alpha=0.8)
ax.grid(axis="x", visible=False)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(frameon=False)

plt.tight_layout()

plt.savefig("hedging_phrases_histogram.pdf", bbox_inches="tight")
# plt.savefig(
#     "hedging_phrases_histogram.png",
#     dpi=300,
#     bbox_inches="tight"
# )