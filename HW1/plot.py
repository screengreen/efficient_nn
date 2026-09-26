import json
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

from equations import bytes_moved, energy, flops, latency

with open("results/theta.json") as file:
    theta = json.load(file)

df = pd.read_csv("results/measurements.csv")
df["is_validation"] = df["is_validation"].astype(str) == "True"
df = df.dropna(subset=["energy", "latency"]).sort_values(["S", "B"]).reset_index(drop=True)
df["flops"] = flops(df["S"].values, df["B"].values)
df["bytes"] = bytes_moved(df["S"].values, df["B"].values)
df["energy_mj"] = df["energy"] * 1e3
df["latency_ms"] = df["latency"] * 1e3
df["predicted_energy_mj"] = energy(
    df["S"].values, df["B"].values, theta["energy"]
) * 1e3
df["predicted_latency_ms"] = latency(
    df["S"].values, df["B"].values, theta["latency"]
) * 1e3

norm = Normalize(vmin=np.log2(df["S"]).min(), vmax=np.log2(df["S"]).max())
cmap = plt.get_cmap("viridis")

os.makedirs("results/figures", exist_ok=True)

plots = [
    ("flops", "energy_mj", "predicted_energy_mj", "FLOPs", "Energy, mJ", "Energy vs FLOPs", "energy_vs_flops.png"),
    ("bytes", "energy_mj", "predicted_energy_mj", "Bytes moved", "Energy, mJ", "Energy vs bytes moved", "energy_vs_bytes.png"),
    ("flops", "latency_ms", "predicted_latency_ms", "FLOPs", "Latency, ms", "Latency vs FLOPs", "latency_vs_flops.png"),
    ("bytes", "latency_ms", "predicted_latency_ms", "Bytes moved", "Latency, ms", "Latency vs bytes moved", "latency_vs_bytes.png"),
]

for x, measured, predicted, xlabel, ylabel, title, filename in plots:
    fig, ax = plt.subplots(figsize=(7, 5.5))

    # One predicted curve per image size; batch size moves along each curve.
    for S in sorted(df["S"].unique()):
        d = df[df["S"] == S].sort_values(x)
        ax.plot(
            d[x],
            d[predicted],
            color=cmap(norm(np.log2(S))),
            linewidth=1.5,
            alpha=0.85,
        )

    for is_validation, marker, label in [(False, "o", "train"), (True, "^", "validation")]:
        d = df[df["is_validation"] == is_validation]
        ax.scatter(
            d[x],
            d[measured],
            c=np.log2(d["S"]),
            cmap=cmap,
            norm=norm,
            marker=marker,
            label=label,
            s=25,
        )

    ax.set(xscale="log", yscale="log", xlabel=xlabel, ylabel=ylabel, title=title)
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="log2(S)")
    ax.legend(
        handles=[
            Line2D([], [], color="black", marker="o", linestyle="none", label="measured train"),
            Line2D([], [], color="black", marker="^", linestyle="none", label="measured validation"),
            Line2D([], [], color="black", linewidth=1.5, label="predicted"),
        ]
    )
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    path = os.path.join("results", "figures", filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(path)
