"""Compare exported white-, gray-, and black-box predictions consistently."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pid_tuning_common import PROJECT_DIR, TIME_COLUMN, load_recipe


PREDICTION_DIR = PROJECT_DIR / "predictions"
PREDICTION_FILES = {
    "White Box": (PREDICTION_DIR / "white_box_preds.csv", "WhiteBox_Temp"),
    "Gray Box": (PREDICTION_DIR / "gray_box_preds.csv", "GrayBox_Temp"),
    "Black Box": (PREDICTION_DIR / "black_box_preds.csv", "BlackBox_Temp"),
}


def load_aligned_predictions() -> pd.DataFrame:
    """Require complete timestamp alignment with freshly filtered test data."""
    recipe = load_recipe()
    aligned = pd.DataFrame(
        {
            TIME_COLUMN: recipe.time,
            "Measured_Temp": recipe.measured_temperature,
        }
    )
    for name, (path, prediction_column) in PREDICTION_FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; run its model script first")
        frame = pd.read_csv(path, usecols=[TIME_COLUMN, prediction_column])
        if len(frame) != len(aligned) or not np.allclose(
            frame[TIME_COLUMN].to_numpy(dtype=float),
            aligned[TIME_COLUMN].to_numpy(dtype=float),
            rtol=0.0,
            atol=1.0e-6,
        ):
            raise ValueError(
                f"{path.name} does not match every test_set.csv timestamp; "
                "rerun that model script to refresh predictions"
            )
        values = frame[prediction_column].to_numpy(dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{path.name} contains non-finite predictions")
        aligned[name] = values
    return aligned


def main() -> None:
    data = load_aligned_predictions()
    measured = data["Measured_Temp"].to_numpy()
    metric_rows: list[dict[str, float | str]] = []

    print(f"Aligned comparison samples: {len(data):,}")
    print("\nPerformance against filtered test temperature")
    for name in PREDICTION_FILES:
        predicted = data[name].to_numpy()
        residual = predicted - measured
        row = {
            "Model": name,
            "MAE": float(np.mean(np.abs(residual))),
            "RMSE": float(np.sqrt(np.mean(residual**2))),
            "Bias": float(np.mean(residual)),
            "MaxAbsError": float(np.max(np.abs(residual))),
        }
        metric_rows.append(row)
        print(
            f"  {name:10s} MAE={row['MAE']:.4f} °C, "
            f"RMSE={row['RMSE']:.4f} °C, bias={row['Bias']:+.4f} °C"
        )

    print("\nPairwise prediction disagreement (RMSE)")
    for first, second in combinations(PREDICTION_FILES, 2):
        difference = data[first].to_numpy() - data[second].to_numpy()
        disagreement = float(np.sqrt(np.mean(difference**2)))
        print(f"  {first} vs {second}: {disagreement:.4f} °C")

    metrics_path = PREDICTION_DIR / "model_comparison_metrics.csv"
    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)
    print(f"\nSaved metric table to: {metrics_path}")

    elapsed = data[TIME_COLUMN].to_numpy() - data[TIME_COLUMN].iloc[0]
    figure, (temperature_axis, residual_axis) = plt.subplots(
        2,
        1,
        figsize=(13, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
        constrained_layout=True,
    )
    temperature_axis.plot(
        elapsed,
        measured,
        color="black",
        linewidth=1.5,
        label="Filtered measured temperature",
    )
    colors = {"White Box": "tab:orange", "Gray Box": "tab:green", "Black Box": "tab:red"}
    for name in PREDICTION_FILES:
        temperature_axis.plot(
            elapsed, data[name], color=colors[name], linewidth=1.2, label=name
        )
        residual_axis.plot(
            elapsed,
            data[name] - measured,
            color=colors[name],
            linewidth=1.0,
            label=name,
        )
    temperature_axis.set_title("FrED Thermal Model Comparison")
    temperature_axis.set_ylabel("Temperature (°C)")
    temperature_axis.grid(True, alpha=0.3)
    temperature_axis.legend()
    residual_axis.axhline(0.0, color="black", linewidth=0.8)
    residual_axis.set_xlabel("Elapsed time (s)")
    residual_axis.set_ylabel("Prediction error (°C)")
    residual_axis.grid(True, alpha=0.3)
    residual_axis.legend()
    plt.show()


if __name__ == "__main__":
    main()
