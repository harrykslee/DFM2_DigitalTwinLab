"""White-box lumped-capacitance model for the FrED heat block.

The script evaluates the same physical model against two experiments:

* ``step_PWM10.csv`` -- a constant-PWM step-response experiment.
* ``test_set.csv`` -- a dynamic, multi-setpoint PID experiment.

Only measured temperature and the recorded heater command are used.  No model
parameters are fitted to either dataset, so this remains a pure white-box
comparison.
"""

import json
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Physical model parameters (SI units)
# ---------------------------------------------------------------------------
C_TH = 50.0  # Thermal capacitance, J/K
R_TH = 20.0  # Thermal resistance to ambient, K/W
Q_MAX = 40.0  # Heater power at 100 percent PWM, W

# Resolve paths relative to this script so it works from any working directory.
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
PREDICTION_DIR = PROJECT_DIR / "predictions"
MODEL_DIR = PROJECT_DIR / "models"
STEP_DATA_PATH = DATA_DIR / "step_PWM10.csv"
TEST_DATA_PATH = DATA_DIR / "test_set.csv"
PREDICTION_PATH = PREDICTION_DIR / "white_box_preds.csv"
MODEL_PATH = MODEL_DIR / "white_box_model.json"

TIME_COLUMN = "Timestamp (s)"
TEMPERATURE_COLUMN = "Temperature (C)"
DUTY_COLUMN = "Temperature Duty Cycle %"
PID_OUTPUT_COLUMN = "Temperature PID output"
DT_COLUMN = "dt (s)"

# Filter all measured temperatures before they are used by the model.  The
# centered windows smooth noise without introducing a timing shift.
MEDIAN_FILTER_SECONDS = 3.0
MOVING_AVERAGE_SECONDS = 8.0


def odd_window_size(duration: float, sample_interval: float) -> int:
    """Convert a filter duration in seconds to an odd sample count."""
    size = max(3, int(round(duration / sample_interval)))
    return size if size % 2 == 1 else size + 1


def filter_temperature(time: np.ndarray, temperature: np.ndarray) -> np.ndarray:
    """Apply centered median and moving-average filters to temperature data."""
    median_dt = float(np.median(np.diff(time)))
    median_window = odd_window_size(MEDIAN_FILTER_SECONDS, median_dt)
    average_window = odd_window_size(MOVING_AVERAGE_SECONDS, median_dt)

    filtered = pd.Series(temperature).rolling(
        window=median_window, center=True, min_periods=1
    ).median()
    filtered = filtered.rolling(
        window=average_window, center=True, min_periods=1
    ).mean()
    return filtered.to_numpy(dtype=float)


def read_temperature_section(csv_path: Path) -> pd.DataFrame:
    """Read only the TEMPERATURE DATA section of a multi-section CSV file.

    The FrED logger writes a label line before the CSV header and may append
    diameter, motor, or spooler tables later in the same file.  Those tables
    are separated from the temperature table by a blank line.  Extracting the
    first section before calling ``read_csv`` avoids interpreting their rows as
    temperature data.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    lines = csv_path.read_text(encoding="utf-8-sig").splitlines()

    try:
        section_label_index = next(
            index
            for index, line in enumerate(lines)
            if line.strip().upper() == "TEMPERATURE DATA"
        )
    except StopIteration as exc:
        raise ValueError(f"No TEMPERATURE DATA section found in {csv_path}") from exc

    # Find the first nonblank line after the label; this is the CSV header.
    header_index = section_label_index + 1
    while header_index < len(lines) and not lines[header_index].strip():
        header_index += 1

    section_lines = []
    for line in lines[header_index:]:
        if not line.strip():
            break
        section_lines.append(line)

    if len(section_lines) < 2:
        raise ValueError(f"Temperature section in {csv_path} contains no data rows")

    frame = pd.read_csv(StringIO("\n".join(section_lines)))
    frame.columns = frame.columns.str.strip()
    return frame


def prepare_experiment(csv_path: Path) -> dict[str, object]:
    """Load and validate one experiment and construct its integration steps."""
    frame = read_temperature_section(csv_path)

    # The step test calls the heater command a duty cycle.  The older PID log
    # records the same percentage command under "Temperature PID output".
    if DUTY_COLUMN in frame.columns:
        input_column = DUTY_COLUMN
    elif PID_OUTPUT_COLUMN in frame.columns:
        input_column = PID_OUTPUT_COLUMN
    else:
        raise ValueError(
            f"{csv_path.name} must contain either {DUTY_COLUMN!r} or "
            f"{PID_OUTPUT_COLUMN!r}."
        )

    required_columns = [TIME_COLUMN, TEMPERATURE_COLUMN, input_column]
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_path.name}: {missing}")

    numeric_columns = required_columns.copy()
    if DT_COLUMN in frame.columns:
        numeric_columns.append(DT_COLUMN)

    # Coercion gives a clear validation path if a logger emits a malformed row.
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    invalid_rows = frame[required_columns].isna().any(axis=1)
    if invalid_rows.any():
        # Some logs end their temperature section with one partially written
        # sensor row (for example, a temperature with no timestamp or command).
        # Such a row cannot be integrated, so report and remove it explicitly.
        bad_row_numbers = (np.flatnonzero(invalid_rows.to_numpy()) + 2).tolist()
        print(
            f"Warning: dropping incomplete/non-numeric rows from {csv_path.name}: "
            f"{bad_row_numbers[:10]}"
        )
        frame = frame.loc[~invalid_rows].reset_index(drop=True)

    time = frame[TIME_COLUMN].to_numpy(dtype=float)
    raw_temperature = frame[TEMPERATURE_COLUMN].to_numpy(dtype=float)
    pwm = frame[input_column].to_numpy(dtype=float)

    if len(frame) < 2:
        raise ValueError(f"{csv_path.name} needs at least two temperature samples")
    if not np.all(np.isfinite(time)) or not np.all(np.isfinite(raw_temperature)):
        raise ValueError(f"{csv_path.name} contains non-finite time or temperature data")
    if not np.all(np.isfinite(pwm)):
        raise ValueError(f"{csv_path.name} contains non-finite heater commands")
    if np.any((pwm < 0.0) | (pwm > 100.0)):
        raise ValueError(f"Heater commands in {csv_path.name} must be from 0 to 100 percent")

    timestamp_steps = np.diff(time)
    if np.any(timestamp_steps <= 0.0):
        raise ValueError(f"Timestamps in {csv_path.name} must be strictly increasing")

    # From this point onward, all initialization, metrics, plots, and exports
    # use the filtered signal rather than the raw thermocouple measurements.
    measured = filter_temperature(time, raw_temperature)

    # Each logger dt value describes the interval ending at its row.  Ignore
    # the first dt because the first measured temperature is the initial state.
    if DT_COLUMN in frame.columns:
        dt = frame[DT_COLUMN].to_numpy(dtype=float)
        if not np.all(np.isfinite(dt[1:])) or np.any(dt[1:] <= 0.0):
            raise ValueError(f"Time steps in {csv_path.name} must be finite and positive")
        integration_steps = dt[1:]
        step_source = DT_COLUMN
    else:
        integration_steps = timestamp_steps
        step_source = f"differences of {TIME_COLUMN}"

    return {
        "name": csv_path.name,
        "frame": frame,
        "time": time,
        "measured": measured,
        "pwm": pwm,
        "integration_steps": integration_steps,
        "input_column": input_column,
        "step_source": step_source,
    }


def simulate_lumped_capacitance(
    measured_initial_temperature: float,
    pwm_percent: np.ndarray,
    integration_steps: np.ndarray,
    ambient_temperature: float,
) -> np.ndarray:
    """Integrate the first-order heat balance with explicit Euler steps.

    The recorded command at sample ``i - 1`` is treated as constant over the
    interval ending at sample ``i`` (a zero-order hold).
    """
    prediction = np.empty(pwm_percent.size, dtype=float)
    prediction[0] = measured_initial_temperature

    for i in range(1, pwm_percent.size):
        heater_power = (pwm_percent[i - 1] / 100.0) * Q_MAX
        heat_loss = (prediction[i - 1] - ambient_temperature) / R_TH
        d_temperature_dt = (heater_power - heat_loss) / C_TH
        prediction[i] = prediction[i - 1] + integration_steps[i - 1] * d_temperature_dt

    return prediction


def error_metrics(measured: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    """Return mean absolute error and root mean squared error in degrees C."""
    residual = predicted - measured
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual**2)))
    return mae, rmse


def evaluate_experiment(experiment: dict[str, object]) -> dict[str, object]:
    """Simulate one experiment and attach its predictions and error metrics."""
    measured = experiment["measured"]
    ambient_temperature = float(measured[0])
    predicted = simulate_lumped_capacitance(
        measured_initial_temperature=ambient_temperature,
        pwm_percent=experiment["pwm"],
        integration_steps=experiment["integration_steps"],
        ambient_temperature=ambient_temperature,
    )
    mae, rmse = error_metrics(measured, predicted)

    return {
        **experiment,
        "ambient_temperature": ambient_temperature,
        "predicted": predicted,
        "mae": mae,
        "rmse": rmse,
    }


def plot_comparisons(results: list[dict[str, object]]) -> None:
    """Plot measured and predicted temperatures for both experiments."""
    fig, axes = plt.subplots(len(results), 1, figsize=(12, 9), constrained_layout=True)
    axes = np.atleast_1d(axes)

    titles = {
        "step_PWM10.csv": "10% PWM Step-Response Test",
        "test_set.csv": "Dynamic Multi-Setpoint PID Test",
    }

    for axis, result in zip(axes, results):
        axis.plot(
            result["time"],
            result["measured"],
            color="tab:blue",
            linewidth=1.5,
            label="Filtered Measured Temperature",
        )
        axis.plot(
            result["time"],
            result["predicted"],
            color="tab:orange",
            linewidth=2.0,
            label="White-Box Model Prediction",
        )
        axis.set_title(titles.get(result["name"], result["name"]))
        axis.set_xlabel("Timestamp (s)")
        axis.set_ylabel("Temperature (°C)")
        axis.grid(True, alpha=0.3)
        axis.legend()

    fig.suptitle("FrED Heat-Block White-Box Model Validation", fontsize=14)
    plt.show()


def main() -> None:
    """Run both validations, report errors, export test predictions, and plot."""
    experiments = [
        prepare_experiment(STEP_DATA_PATH),
        prepare_experiment(TEST_DATA_PATH),
    ]
    results = [evaluate_experiment(experiment) for experiment in experiments]

    print("FrED lumped-capacitance white-box model")
    print(f"Parameters: C_th={C_TH:.1f} J/K, R_th={R_TH:.1f} K/W, Q_max={Q_MAX:.1f} W")
    for result in results:
        print(f"\n{result['name']}")
        print(f"  Samples: {len(result['time'])}")
        print(f"  PWM input: {result['input_column']}")
        print(f"  Time steps: {result['step_source']}")
        print(f"  Ambient temperature: {result['ambient_temperature']:.3f} °C")
        print(f"  MAE:  {result['mae']:.4f} °C")
        print(f"  RMSE: {result['rmse']:.4f} °C")

    # The master comparison script expects predictions for test_set.csv only.
    test_result = next(result for result in results if result["name"] == "test_set.csv")
    prediction_frame = pd.DataFrame(
        {
            TIME_COLUMN: test_result["time"],
            "Measured_Temp": test_result["measured"],
            "WhiteBox_Temp": test_result["predicted"],
        }
    )
    prediction_frame.to_csv(PREDICTION_PATH, index=False)
    print(f"\nSaved test-set predictions to: {PREDICTION_PATH}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.write_text(
        json.dumps(
            {
                "model_type": "white_box_lumped_capacitance",
                "version": 1,
                "C_th_J_per_K": C_TH,
                "R_th_K_per_W": R_TH,
                "Q_max_W": Q_MAX,
                "median_filter_seconds": MEDIAN_FILTER_SECONDS,
                "moving_average_seconds": MOVING_AVERAGE_SECONDS,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved reusable white-box model to: {MODEL_PATH}")

    plot_comparisons(results)


if __name__ == "__main__":
    main()
