import json

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from equations import energy, latency

df = pd.read_csv("results/measurements.csv")

train = df[df["is_validation"] == False]
val = df[df["is_validation"] == True]


def fit(predict, frame, target, x0):
    usable = frame[frame[target].notna()]
    S = usable["S"].to_numpy()
    B = usable["B"].to_numpy()
    y = usable[target].to_numpy()
    result = least_squares(
        lambda theta: np.log(predict(S, B, theta)) - np.log(y),
        x0=x0,
        bounds=(1e-12, np.inf),
    )

    return result.x


def metrics(y, y_pred):
    ape = np.abs((y - y_pred) / y) * 100

    return {
        "mae": float(np.mean(np.abs(y - y_pred))),
        "mape": float(np.mean(ape)),
        "median_ape": float(np.median(ape)),
        "p90_ape": float(np.percentile(ape, 90)),
    }


def report(name, frame, target, predict, theta, unit):
    usable = frame[frame[target].notna()]
    y = usable[target].to_numpy()
    y_pred = predict(usable["S"].to_numpy(), usable["B"].to_numpy(), theta)
    m = metrics(y, y_pred)
    scale = 1e3 if unit == "ms" else 1

    print(
        f"{name:5s} MAE={m['mae'] * scale:8.3f} {unit}  "
        f"MAPE={m['mape']:6.2f} %  "
        f"median APE={m['median_ape']:6.2f} %  "
        f"p90 APE={m['p90_ape']:6.2f} %"
    )

    return m


theta_latency = fit(latency, train, "latency", [1e-3, 1e-3, 1e-2])

print("latency")
print(f"launch floor : {theta_latency[0] * 1e3:8.3f} ms")
print(f"per GFLOP    : {theta_latency[1] * 1e3:8.3f} ms")
print(f"per GB       : {(1 - theta_latency[2]) * 1e3:8.3f} ms")

train_latency_metrics = report("train", train, "latency", latency, theta_latency, "ms")
val_latency_metrics = report("val", val, "latency", latency, theta_latency, "ms")

theta_energy = fit(energy, train, "energy", [0.05, 0.03, 0.02])

print()
print("energy")
print(f"launch floor : {theta_energy[0] * 1e3:8.2f} mJ")
print(f"per GFLOP    : {theta_energy[1] * 1e3:8.3f} mJ")
print(f"slowdown     : {theta_energy[2]:8.4f} per GB moved")

train_energy_metrics = report("train", train, "energy", energy, theta_energy, "J")
val_energy_metrics = report("val", val, "energy", energy, theta_energy, "J")

with open("results/theta.json", "w") as file:
    json.dump(
        {
            "latency": theta_latency.tolist(),
            "launch_floor_seconds": float(theta_latency[0]),
            "seconds_per_gflop": float(theta_latency[1]),
            "seconds_per_gb": float(1 - theta_latency[2]),
            "train": train_latency_metrics,
            "validation": val_latency_metrics,
            "validation_mae": val_latency_metrics["mae"],
            "validation_mape": val_latency_metrics["mape"],
            "energy": theta_energy.tolist(),
            "energy_floor_joules": float(theta_energy[0]),
            "joules_per_gflop": float(theta_energy[1]),
            "energy_slowdown_per_gb": float(theta_energy[2]),
            "energy_train": train_energy_metrics,
            "energy_validation": val_energy_metrics,
            "energy_validation_mae": val_energy_metrics["mae"],
            "energy_validation_mape": val_energy_metrics["mape"],
        },
        file,
        indent=4,
    )
