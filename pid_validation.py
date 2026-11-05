"""Validate saved FrED plant models against a new PID experiment.

Place the new recording at ``data/pid_validation.csv`` and run this script.
It prompts for the Kp, Ki, and Kd gains actually used in that experiment.
The same setpoint recipe and time stamps from the recording drive three
independent, closed-loop simulations.  Only the first filtered measured
temperature is used to initialize the simulations; later measurements are
reserved for validation.

The simulated controller uses the PID and anti-windup implementation in
``pid_tuning_common.py``.  A different controller implementation, initial
integrator state, or actuator behavior can also cause model/experiment error.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pid_tuning_common import (
    DATA_DIR,
    PIDGains,
    PID_OUTPUT_COLUMN,
    PROJECT_DIR,
    Recipe,
    SETPOINT_COLUMN,
    TEMPERATURE_COLUMN,
    TIME_COLUMN,
    filter_temperature,
    load_models,
    read_temperature_section,
    simulate_pid,
)


VALIDATION_PATH = DATA_DIR / "pid_validation.csv"
OUTPUT_PATH = PROJECT_DIR / "predictions" / "pid_validation_preds.csv"
ACTUAL_DUTY_COLUMNS = (PID_OUTPUT_COLUMN, "Temperature Duty Cycle %")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare saved thermal models with a new PID validation run"
    )
    parser.add_argument(
        "--data", type=Path, default=VALIDATION_PATH,
        help="Validation CSV (default: data/pid_validation.csv)",
    )
    parser.add_argument("--kp", type=float, help="Proportional gain used in the run")
    parser.add_argument("--ki", type=float, help="Integral gain used in the run")
    parser.add_argument("--kd", type=float, help="Derivative gain used in the run")
    parser.add_argument("--no-plot", action="store_true", help="Skip the plot")
    args = parser.parse_args()
    supplied = [args.kp is not None, args.ki is not None, args.kd is not None]
    if any(supplied) and not all(supplied):
        parser.error("provide all three of --kp, --ki, and --kd, or omit all to be prompted")
    return args


def read_gains(args: argparse.Namespace) -> PIDGains:
    """Prompt interactively unless all three gains were supplied on the CLI."""
    if args.kp is None:
        print("Enter the PID gains used to record the validation CSV:")
        try:
            values = [float(input(f"  {label}: ")) for label in ("Kp", "Ki", "Kd")]
        except (ValueError, EOFError) as exc:
            raise ValueError("Enter three numeric PID gains, or use --kp/--ki/--kd") from exc
    else:
        values = [args.kp, args.ki, args.kd]
    if not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise ValueError("PID gains must be finite and non-negative")
    return PIDGains(*values)


def load_validation_recipe(path: Path, gains: PIDGains) -> tuple[Recipe, bool]:
    """Load either a FrED sectioned export or a regular three-column CSV."""
    with path.open("r", encoding="utf-8-sig") as source:
        first_line = source.readline().strip()
    if first_line.upper() == "TEMPERATURE DATA":
        frame = read_temperature_section(path)
    else:
        frame = pd.read_csv(path, encoding="utf-8-sig")
        frame.columns = frame.columns.str.strip()

    required = (TIME_COLUMN, TEMPERATURE_COLUMN, SETPOINT_COLUMN)
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")
    duty_column = next((column for column in ACTUAL_DUTY_COLUMNS if column in frame), None)
    numeric_columns = [*required] + ([duty_column] if duty_column else [])
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[numeric_columns].isna().any().any():
        raise ValueError(f"{path.name} contains missing or nonnumeric validation samples")

    time = frame[TIME_COLUMN].to_numpy(dtype=float)
    if len(time) < 2 or not np.all(np.isfinite(time)) or np.any(np.diff(time) <= 0):
        raise ValueError("Validation timestamps must be finite and strictly increasing")
    measured = filter_temperature(time, frame[TEMPERATURE_COLUMN].to_numpy(dtype=float))
    setpoint = frame[SETPOINT_COLUMN].to_numpy(dtype=float)
    if not np.all(np.isfinite(measured)) or not np.all(np.isfinite(setpoint)):
        raise ValueError("Validation temperatures and setpoints must be finite")
    recorded_duty = (
        frame[duty_column].to_numpy(dtype=float)
        if duty_column else np.zeros(len(frame), dtype=float)
    )
    recipe = Recipe(
        time=time,
        elapsed=time - time[0],
        dt=np.diff(time),
        setpoint=setpoint,
        measured_temperature=measured,
        recorded_duty=recorded_duty,
        old_gains=gains,
    )
    return recipe, duty_column is not None


def error_metrics(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    error = predicted - actual
    return float(np.mean(np.abs(error))), float(np.sqrt(np.mean(error**2)))


def main() -> None:
    args = parse_arguments()
    path = args.data.expanduser()
    if not path.is_file():
        raise SystemExit(
            f"Validation data not found: {path}. Upload the recording as "
            "data/pid_validation.csv, then rerun this script."
        )

    gains = read_gains(args)
    recipe, has_recorded_duty = load_validation_recipe(path, gains)
    models = load_models()

    print(f"Validation file: {path} ({len(recipe.time):,} samples)")
    print(f"Entered gains: Kp={gains.kp:g}, Ki={gains.ki:g}, Kd={gains.kd:g}")
    actual_tracking = error_metrics(recipe.setpoint, recipe.measured_temperature)
    print(f"Actual recipe tracking: MAE={actual_tracking[0]:.4f} C, "
          f"RMSE={actual_tracking[1]:.4f} C")

    output = pd.DataFrame({
        TIME_COLUMN: recipe.time,
        "Setpoint_Temp": recipe.setpoint,
        "Measured_Temp": recipe.measured_temperature,
    })
    results: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    columns = {"White Box": "WhiteBox_Temp", "Gray Box": "GrayBox_Temp",
               "Black Box": "BlackBox_Temp"}
    for name, artifact in models.items():
        predicted, duty = simulate_pid(name, artifact, gains, recipe)
        if not np.all(np.isfinite(predicted)):
            raise ValueError(f"{name} produced non-finite validation predictions")
        mae, rmse = error_metrics(recipe.measured_temperature, predicted)
        tracking_mae, tracking_rmse = error_metrics(recipe.setpoint, predicted)
        print(f"{name}: measured-response MAE={mae:.4f} C, RMSE={rmse:.4f} C; "
              f"predicted recipe tracking MAE={tracking_mae:.4f} C, "
              f"RMSE={tracking_rmse:.4f} C")
        if has_recorded_duty:
            duty_mae, _ = error_metrics(recipe.recorded_duty[:-1], duty[:-1])
            print(f"  Simulated vs recorded heater command MAE={duty_mae:.4f} percentage points")
        output[columns[name]] = predicted
        results[name] = (predicted, duty)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved validation comparison to: {OUTPUT_PATH}")

    if not args.no_plot:
        figure, axis = plt.subplots(figsize=(13, 7), constrained_layout=True)
        axis.plot(recipe.elapsed, recipe.setpoint, "k:", linewidth=1.4,
                  label="Temperature recipe")
        axis.plot(recipe.elapsed, recipe.measured_temperature, color="0.25",
                  linewidth=1.6, label="Measured temperature (filtered)")
        colors = {"White Box": "tab:orange", "Gray Box": "tab:green",
                  "Black Box": "tab:red"}
        for name, (predicted, _) in results.items():
            axis.plot(recipe.elapsed, predicted, color=colors[name], linewidth=1.3,
                      label=f"{name} prediction")
        axis.set_title("PID Validation: Recorded Response vs Saved Models")
        axis.set_xlabel("Elapsed time (s)")
        axis.set_ylabel("Temperature (C)")
        axis.grid(True, alpha=0.3)
        axis.legend()
        plt.show()


if __name__ == "__main__":
    main()
