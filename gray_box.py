"""FOPDT gray-box model and teaching exercise for the FrED heat block.

Workflow
--------
1. Load and filter the temperatures in both experiments.
2. Display the raw and filtered response so a student can estimate K, tau,
   and theta from the graph.
3. Ask for the student's estimates (or accept them as command-line options).
4. Reveal parameters calculated with the 28.3/63.2 percent reaction-curve
   method.
5. Compare the student and calculated FOPDT models against both experiments.
6. Export only the calculated-parameter predictions for ``test_set.csv``.

The model is

    G(s) = K / (tau*s + 1) * exp(-theta*s)

or, in the time domain for temperature deviation from its baseline,

    tau * d(Delta T)/dt + Delta T = K * u(t - theta).

Here, K has units of degrees C per percent PWM, while tau and theta are in
seconds.  The script uses no black-box machine-learning model.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Files and column names
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
PREDICTION_DIR = PROJECT_DIR / "predictions"
MODEL_DIR = PROJECT_DIR / "models"

STEP_DATA_PATH = DATA_DIR / "step_PWM10.csv"
TEST_DATA_PATH = DATA_DIR / "test_set.csv"
PREDICTION_PATH = PREDICTION_DIR / "gray_box_preds.csv"
MODEL_PATH = MODEL_DIR / "gray_box_model.json"

TIME_COLUMN = "Timestamp (s)"
TEMPERATURE_COLUMN = "Temperature (C)"
DUTY_COLUMN = "Temperature Duty Cycle %"
PID_OUTPUT_COLUMN = "Temperature PID output"
DT_COLUMN = "dt (s)"

# Filtering durations are specified in seconds, then converted to sample
# counts using the median sampling interval.  Both filters are centered, so
# they smooth the identification data without adding an artificial time lag.
MEDIAN_FILTER_SECONDS = 3.0
MOVING_AVERAGE_SECONDS = 8.0

# Portions used to estimate the initial and final steady temperatures.
INITIAL_BASELINE_SECONDS = 3.0
FINAL_STEADY_FRACTION = 0.10


@dataclass(frozen=True)
class FOPDTParameters:
    """Parameters of G(s) = K/(tau*s + 1) * exp(-theta*s)."""

    gain: float  # degrees C / percent PWM
    tau: float  # time constant, s
    theta: float  # dead time, s


def read_temperature_section(csv_path: Path) -> pd.DataFrame:
    """Read only the leading TEMPERATURE DATA table from a FrED log.

    A FrED CSV can contain additional diameter, motor, and spooler tables after
    the temperature table.  Reading only the first section prevents those
    differently shaped rows from contaminating the thermal experiment.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    lines = csv_path.read_text(encoding="utf-8-sig").splitlines()
    try:
        label_index = next(
            index
            for index, line in enumerate(lines)
            if line.strip().upper() == "TEMPERATURE DATA"
        )
    except StopIteration as exc:
        raise ValueError(f"No TEMPERATURE DATA section found in {csv_path}") from exc

    header_index = label_index + 1
    while header_index < len(lines) and not lines[header_index].strip():
        header_index += 1

    section_lines: list[str] = []
    for line in lines[header_index:]:
        if not line.strip():
            break
        section_lines.append(line)

    if len(section_lines) < 2:
        raise ValueError(f"Temperature section in {csv_path} has no data rows")

    frame = pd.read_csv(StringIO("\n".join(section_lines)))
    frame.columns = frame.columns.str.strip()
    return frame


def prepare_experiment(csv_path: Path) -> dict[str, object]:
    """Load one experiment and return validated NumPy arrays."""
    frame = read_temperature_section(csv_path)

    # Newer step logs use a duty-cycle name.  The older PID test stores the
    # same heater percentage as "Temperature PID output".
    if DUTY_COLUMN in frame.columns:
        input_column = DUTY_COLUMN
    elif PID_OUTPUT_COLUMN in frame.columns:
        input_column = PID_OUTPUT_COLUMN
    else:
        raise ValueError(
            f"{csv_path.name} must contain {DUTY_COLUMN!r} or "
            f"{PID_OUTPUT_COLUMN!r}"
        )

    required = [TIME_COLUMN, TEMPERATURE_COLUMN, input_column]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_path.name}: {missing}")

    numeric_columns = required + ([DT_COLUMN] if DT_COLUMN in frame.columns else [])
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    invalid = frame[required].isna().any(axis=1)
    if invalid.any():
        # A partially written final sensor row has no usable timestamp/input.
        csv_rows = (np.flatnonzero(invalid.to_numpy()) + 2).tolist()
        print(
            f"Warning: dropping incomplete/non-numeric rows from {csv_path.name}: "
            f"{csv_rows[:10]}"
        )
        frame = frame.loc[~invalid].reset_index(drop=True)

    if len(frame) < 2:
        raise ValueError(f"{csv_path.name} needs at least two valid samples")

    time = frame[TIME_COLUMN].to_numpy(dtype=float)
    raw_temperature = frame[TEMPERATURE_COLUMN].to_numpy(dtype=float)
    pwm = frame[input_column].to_numpy(dtype=float)

    if not all(
        np.all(np.isfinite(values)) for values in (time, raw_temperature, pwm)
    ):
        raise ValueError(f"{csv_path.name} contains non-finite required values")
    if np.any(np.diff(time) <= 0.0):
        raise ValueError(f"Timestamps in {csv_path.name} must strictly increase")
    if np.any((pwm < 0.0) | (pwm > 100.0)):
        raise ValueError(f"Heater commands in {csv_path.name} must be 0 to 100 percent")

    # All later identification, initialization, metrics, plots, and exports
    # receive only this filtered temperature signal.
    measured = filter_temperature(time, raw_temperature)

    if DT_COLUMN in frame.columns:
        logged_dt = frame[DT_COLUMN].to_numpy(dtype=float)[1:]
        if not np.all(np.isfinite(logged_dt)) or np.any(logged_dt <= 0.0):
            raise ValueError(f"Invalid dt values in {csv_path.name}")
        integration_steps = logged_dt
        step_source = DT_COLUMN
    else:
        integration_steps = np.diff(time)
        step_source = f"differences of {TIME_COLUMN}"

    return {
        "name": csv_path.name,
        "time": time,
        "elapsed_time": time - time[0],
        "measured": measured,
        "pwm": pwm,
        "integration_steps": integration_steps,
        "input_column": input_column,
        "step_source": step_source,
    }


def odd_window_size(duration: float, sample_interval: float) -> int:
    """Convert a duration in seconds to an odd filter-window sample count."""
    size = max(3, int(round(duration / sample_interval)))
    return size if size % 2 == 1 else size + 1


def filter_temperature(time: np.ndarray, temperature: np.ndarray) -> np.ndarray:
    """Remove temperature spikes and noise while preserving response timing.

    A centered rolling median rejects isolated thermocouple spikes.  A centered
    moving average then smooths the remaining high-frequency noise.  Centering
    is important for dead-time identification because a causal moving average
    would shift the response to the right and inflate theta.
    """
    median_dt = float(np.median(np.diff(time)))
    median_window = odd_window_size(MEDIAN_FILTER_SECONDS, median_dt)
    average_window = odd_window_size(MOVING_AVERAGE_SECONDS, median_dt)

    series = pd.Series(temperature)
    filtered = series.rolling(
        window=median_window, center=True, min_periods=1
    ).median()
    filtered = filtered.rolling(
        window=average_window, center=True, min_periods=1
    ).mean()
    return filtered.to_numpy(dtype=float)


def first_crossing_time(
    time: np.ndarray, signal: np.ndarray, target: float
) -> float:
    """Find a rising target crossing and linearly interpolate between samples."""
    crossing_indices = np.flatnonzero(signal >= target)
    if crossing_indices.size == 0:
        raise ValueError(
            f"Filtered step response never reaches the target {target:.3f} °C"
        )

    index = int(crossing_indices[0])
    if index == 0:
        return float(time[0])

    t0, t1 = time[index - 1], time[index]
    y0, y1 = signal[index - 1], signal[index]
    if np.isclose(y1, y0):
        return float(t1)
    fraction = (target - y0) / (y1 - y0)
    return float(t0 + fraction * (t1 - t0))


def calculate_fopdt_parameters(
    step_experiment: dict[str, object], temperature: np.ndarray
) -> tuple[FOPDTParameters, dict[str, float]]:
    """Calculate FOPDT parameters with the 28.3/63.2% two-point method.

    For an ideal FOPDT step response:

        t_28.3 = theta + tau/3
        t_63.2 = theta + tau

    Therefore tau = 1.5*(t_63.2 - t_28.3), and
    theta = t_63.2 - tau.  Gain is Delta temperature / Delta PWM.
    """
    elapsed = step_experiment["elapsed_time"]
    pwm = step_experiment["pwm"]
    median_dt = float(np.median(np.diff(elapsed)))

    initial_count = max(3, int(round(INITIAL_BASELINE_SECONDS / median_dt)))
    final_count = max(20, int(round(FINAL_STEADY_FRACTION * len(elapsed))))
    initial_temperature = float(np.median(temperature[:initial_count]))
    final_temperature = float(np.median(temperature[-final_count:]))
    temperature_change = final_temperature - initial_temperature

    # The recording begins when the step is applied, so its pre-step command is
    # assumed to be 0 percent.  Use the median command to reject any odd sample.
    pre_step_pwm = 0.0
    post_step_pwm = float(np.median(pwm))
    pwm_change = post_step_pwm - pre_step_pwm

    if temperature_change <= 0.0:
        raise ValueError("The step experiment does not contain a rising response")
    if pwm_change <= 0.0:
        raise ValueError("The step experiment does not contain a positive PWM step")

    target_28 = initial_temperature + 0.283 * temperature_change
    target_63 = initial_temperature + 0.632 * temperature_change
    time_28 = first_crossing_time(elapsed, temperature, target_28)
    time_63 = first_crossing_time(elapsed, temperature, target_63)
    if time_63 <= time_28:
        raise ValueError("Invalid reaction-curve crossings: t63.2 must exceed t28.3")

    tau = 1.5 * (time_63 - time_28)
    raw_theta = time_63 - tau
    # Noise and a nearly delay-free response can produce a tiny negative
    # intercept.  Negative transport delay is nonphysical, so constrain it.
    theta = max(0.0, raw_theta)
    gain = temperature_change / pwm_change

    details = {
        "initial_temperature": initial_temperature,
        "final_temperature": final_temperature,
        "pre_step_pwm": pre_step_pwm,
        "post_step_pwm": post_step_pwm,
        "target_28": target_28,
        "target_63": target_63,
        "time_28": time_28,
        "time_63": time_63,
        "raw_theta": raw_theta,
    }
    return FOPDTParameters(gain, tau, theta), details


def show_identification_plot(step_experiment: dict[str, object]) -> None:
    """Show the step-response graph before requesting the student's estimate."""
    elapsed = step_experiment["elapsed_time"]
    measured = step_experiment["measured"]
    pwm = step_experiment["pwm"]

    fig, temperature_axis = plt.subplots(figsize=(12, 6), constrained_layout=True)
    temperature_axis.plot(
        elapsed,
        measured,
        color="tab:blue",
        linewidth=2.0,
        label="Filtered measured temperature",
    )
    temperature_axis.set_xlabel("Elapsed time since PWM step (s)")
    temperature_axis.set_ylabel("Temperature (°C)")
    temperature_axis.grid(True, alpha=0.3)

    input_axis = temperature_axis.twinx()
    input_axis.step(
        elapsed,
        pwm,
        where="post",
        color="tab:red",
        alpha=0.65,
        label="Heater command",
    )
    input_axis.set_ylabel("Heater command (% PWM)", color="tab:red")
    input_axis.tick_params(axis="y", labelcolor="tab:red")
    input_axis.set_ylim(bottom=0.0)

    lines_1, labels_1 = temperature_axis.get_legend_handles_labels()
    lines_2, labels_2 = input_axis.get_legend_handles_labels()
    temperature_axis.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")
    temperature_axis.set_title(
        "FrED 10% PWM Step Response — Estimate K, tau, and theta, then close the plot"
    )
    fig.suptitle(
        "Hints: K = ΔT/ΔPWM; theta is the response delay; "
        "tau reaches 63.2% of ΔT after theta",
        fontsize=10,
    )
    plt.show(block=True)


def prompt_for_positive_float(prompt: str, allow_zero: bool = False) -> float:
    """Prompt until the student enters a physically valid number."""
    while True:
        try:
            value = float(input(prompt))
        except ValueError:
            print("Please enter a numeric value.")
            continue
        except EOFError as exc:
            raise RuntimeError(
                "No interactive input is available. Supply --student-k, "
                "--student-tau, and --student-theta on the command line."
            ) from exc

        valid = value >= 0.0 if allow_zero else value > 0.0
        if valid and np.isfinite(value):
            return value
        comparison = "non-negative" if allow_zero else "greater than zero"
        print(f"The value must be finite and {comparison}.")


def obtain_student_parameters(args: argparse.Namespace) -> FOPDTParameters:
    """Use command-line estimates when complete; otherwise prompt the student."""
    supplied = [args.student_k, args.student_tau, args.student_theta]
    if any(value is not None for value in supplied):
        if not all(value is not None for value in supplied):
            raise ValueError(
                "Supply all three options: --student-k, --student-tau, "
                "and --student-theta"
            )
        parameters = FOPDTParameters(*supplied)
        if parameters.gain <= 0.0 or parameters.tau <= 0.0 or parameters.theta < 0.0:
            raise ValueError("Student K and tau must be positive; theta cannot be negative")
        return parameters

    print("\nEnter your estimates from the step-response plot.")
    gain = prompt_for_positive_float("Estimated K (°C per % PWM): ")
    tau = prompt_for_positive_float("Estimated tau (s): ")
    theta = prompt_for_positive_float("Estimated theta (s): ", allow_zero=True)
    return FOPDTParameters(gain, tau, theta)


def simulate_fopdt(
    experiment: dict[str, object], parameters: FOPDTParameters
) -> np.ndarray:
    """Simulate the FOPDT model with irregular sample times and dead time.

    PWM is treated as a zero-order-held input.  For each recorded interval, the
    command at t - theta is found from the input history.  Before the first
    recorded sample, the heater command is assumed to be zero.  The exponential
    update is the exact first-order solution for a constant input over a sample
    interval and is more stable than a forward-Euler update.
    """
    elapsed = experiment["elapsed_time"]
    measured = experiment["measured"]
    pwm = experiment["pwm"]
    integration_steps = experiment["integration_steps"]

    prediction = np.empty_like(measured, dtype=float)
    prediction[0] = measured[0]
    baseline_temperature = float(measured[0])

    interval_start_times = elapsed[:-1]
    delayed_query_times = interval_start_times - parameters.theta
    delayed_indices = np.searchsorted(elapsed, delayed_query_times, side="right") - 1
    delayed_pwm = np.zeros_like(interval_start_times, dtype=float)
    available = delayed_indices >= 0
    delayed_pwm[available] = pwm[delayed_indices[available]]

    for index in range(1, measured.size):
        equilibrium_temperature = (
            baseline_temperature + parameters.gain * delayed_pwm[index - 1]
        )
        decay = np.exp(-integration_steps[index - 1] / parameters.tau)
        prediction[index] = equilibrium_temperature + (
            prediction[index - 1] - equilibrium_temperature
        ) * decay

    return prediction


def error_metrics(measured: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    """Return mean absolute error and root mean squared error in degrees C."""
    residual = predicted - measured
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual**2)))
    return mae, rmse


def evaluate_models(
    experiment: dict[str, object],
    student_parameters: FOPDTParameters,
    calculated_parameters: FOPDTParameters,
) -> dict[str, object]:
    """Simulate and score both FOPDT parameter sets on one experiment."""
    measured = experiment["measured"]
    student_prediction = simulate_fopdt(experiment, student_parameters)
    calculated_prediction = simulate_fopdt(experiment, calculated_parameters)
    student_mae, student_rmse = error_metrics(measured, student_prediction)
    calculated_mae, calculated_rmse = error_metrics(measured, calculated_prediction)

    return {
        **experiment,
        "student_prediction": student_prediction,
        "calculated_prediction": calculated_prediction,
        "student_mae": student_mae,
        "student_rmse": student_rmse,
        "calculated_mae": calculated_mae,
        "calculated_rmse": calculated_rmse,
    }


def print_parameter_comparison(
    student: FOPDTParameters,
    calculated: FOPDTParameters,
    calculation_details: dict[str, float],
) -> None:
    """Reveal calculated parameters after the student has committed estimates."""
    print("\nFOPDT parameter comparison")
    print("                         Student       Calculated")
    print(f"  K (°C/% PWM)       {student.gain:12.4f} {calculated.gain:16.4f}")
    print(f"  tau (s)            {student.tau:12.4f} {calculated.tau:16.4f}")
    print(f"  theta (s)          {student.theta:12.4f} {calculated.theta:16.4f}")
    print("\nCalculated reaction-curve details")
    print(
        f"  Initial/final filtered temperature: "
        f"{calculation_details['initial_temperature']:.3f} / "
        f"{calculation_details['final_temperature']:.3f} °C"
    )
    print(
        f"  PWM step: {calculation_details['pre_step_pwm']:.3f} -> "
        f"{calculation_details['post_step_pwm']:.3f}%"
    )
    print(
        f"  28.3%/63.2% crossing times: "
        f"{calculation_details['time_28']:.3f} / "
        f"{calculation_details['time_63']:.3f} s"
    )
    if calculation_details["raw_theta"] < 0.0:
        print(
            "  Note: the two-point method produced a slightly negative delay "
            f"({calculation_details['raw_theta']:.3f} s), so theta was constrained to 0."
        )


def print_error_summary(results: list[dict[str, object]]) -> None:
    """Print MAE and RMSE for both models on both datasets."""
    print("\nError metrics against filtered measured temperature")
    for result in results:
        print(f"\n{result['name']}")
        print(f"  Samples: {len(result['time'])}")
        print(f"  Input column: {result['input_column']}")
        print(f"  Time steps: {result['step_source']}")
        print("  Student parameters:")
        print(f"    MAE:  {result['student_mae']:.4f} °C")
        print(f"    RMSE: {result['student_rmse']:.4f} °C")
        print("  Calculated parameters:")
        print(f"    MAE:  {result['calculated_mae']:.4f} °C")
        print(f"    RMSE: {result['calculated_rmse']:.4f} °C")


def plot_model_comparisons(results: list[dict[str, object]]) -> None:
    """Plot actual, student-model, and calculated-model temperatures."""
    fig, axes = plt.subplots(len(results), 1, figsize=(12, 9), constrained_layout=True)
    axes = np.atleast_1d(axes)
    titles = {
        "step_PWM10.csv": "10% PWM Step-Response Test",
        "test_set.csv": "Dynamic Multi-Setpoint PID Test",
    }

    for axis, result in zip(axes, results):
        axis.plot(
            result["elapsed_time"],
            result["measured"],
            color="tab:blue",
            linewidth=1.2,
            label="Filtered measured temperature",
        )
        axis.plot(
            result["elapsed_time"],
            result["student_prediction"],
            color="tab:green",
            linewidth=1.8,
            linestyle="--",
            label="FOPDT model — student parameters",
        )
        axis.plot(
            result["elapsed_time"],
            result["calculated_prediction"],
            color="tab:orange",
            linewidth=2.0,
            label="FOPDT model — calculated parameters",
        )
        axis.set_title(titles.get(result["name"], result["name"]))
        axis.set_xlabel("Elapsed time (s)")
        axis.set_ylabel("Temperature (°C)")
        axis.grid(True, alpha=0.3)
        axis.legend()

    fig.suptitle("FrED Heat-Block FOPDT Gray-Box Model Validation", fontsize=14)
    plt.show()


def export_test_predictions(test_result: dict[str, object]) -> None:
    """Export only the calculated-parameter prediction for the PID test."""
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    output = pd.DataFrame(
        {
            TIME_COLUMN: test_result["time"],
            "Measured_Temp": test_result["measured"],
            "GrayBox_Temp": test_result["calculated_prediction"],
        }
    )
    output.to_csv(PREDICTION_PATH, index=False)
    print(f"\nSaved calculated test-set predictions to: {PREDICTION_PATH}")


def export_calculated_model(parameters: FOPDTParameters) -> None:
    """Save the computer-identified FOPDT model for PID tuning tools."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.write_text(
        json.dumps(
            {
                "model_type": "gray_box_fopdt",
                "version": 1,
                "gain_C_per_percent_PWM": parameters.gain,
                "tau_seconds": parameters.tau,
                "theta_seconds": parameters.theta,
                "median_filter_seconds": MEDIAN_FILTER_SECONDS,
                "moving_average_seconds": MOVING_AVERAGE_SECONDS,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved reusable calculated gray-box model to: {MODEL_PATH}")


def parse_arguments() -> argparse.Namespace:
    """Parse optional non-interactive student estimates."""
    parser = argparse.ArgumentParser(
        description="Identify and validate a FrED heat-block FOPDT model."
    )
    parser.add_argument("--student-k", type=float, help="Student gain estimate, °C/% PWM")
    parser.add_argument("--student-tau", type=float, help="Student time-constant estimate, s")
    parser.add_argument("--student-theta", type=float, help="Student dead-time estimate, s")
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plots (useful for automated testing; student parameters are still required)",
    )
    return parser.parse_args()


def main() -> None:
    """Run the complete educational identification and validation workflow."""
    args = parse_arguments()
    step_experiment = prepare_experiment(STEP_DATA_PATH)
    test_experiment = prepare_experiment(TEST_DATA_PATH)

    filtered_step = step_experiment["measured"]
    calculated_parameters, details = calculate_fopdt_parameters(
        step_experiment, filtered_step
    )
    export_calculated_model(calculated_parameters)

    if not args.no_plots:
        print(
            "The identification plot will open first. Estimate K, tau, and theta, "
            "then close the plot to enter your values."
        )
        show_identification_plot(step_experiment)

    # The calculated values are intentionally not printed until after the
    # student supplies estimates.
    student_parameters = obtain_student_parameters(args)
    print_parameter_comparison(student_parameters, calculated_parameters, details)

    results = [
        evaluate_models(step_experiment, student_parameters, calculated_parameters),
        evaluate_models(test_experiment, student_parameters, calculated_parameters),
    ]
    print_error_summary(results)

    test_result = next(result for result in results if result["name"] == "test_set.csv")
    export_test_predictions(test_result)

    if not args.no_plots:
        plot_model_comparisons(results)


if __name__ == "__main__":
    main()
