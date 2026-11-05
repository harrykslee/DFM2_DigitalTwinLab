"""Automatically optimize PID gains independently for all three plant models.

A bounded derivative-free pattern search is used because the closed-loop
objective contains saturation and has no convenient analytical derivatives.
"""

from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np

from pid_tuning_common import (
    MODEL_DIR,
    PIDGains,
    load_models,
    load_recipe,
    simulate_pid,
    tracking_metrics,
)


# Conservative bounds for the FrED controller.  They can be changed here if a
# later hardware safety review establishes a different permissible range.
LOWER_BOUNDS = np.array([0.0, 0.0, 0.0])
UPPER_BOUNDS = np.array([5.0, 0.05, 2.0])


def optimize_model(
    name: str,
    artifact: object,
    recipe: object,
    initial: PIDGains,
    max_evaluations: int,
) -> tuple[PIDGains, np.ndarray, np.ndarray, dict[str, float], int]:
    """Optimize one model and return gains, response, duty, metrics, and calls."""
    starting_point = np.clip(
        np.array([initial.kp, initial.ki, initial.kd], dtype=float),
        LOWER_BOUNDS,
        UPPER_BOUNDS,
    )
    evaluations = 0

    def objective(values: np.ndarray) -> float:
        nonlocal evaluations
        evaluations += 1
        gains = PIDGains(*map(float, values))
        try:
            temperature, duty = simulate_pid(name, artifact, gains, recipe)
            if not np.all(np.isfinite(temperature)):
                return 1.0e9
            score = tracking_metrics(temperature, duty, recipe)["Objective"]
            if evaluations == 1 or evaluations % 10 == 0:
                print(f"    {evaluations}/{max_evaluations} evaluations; score={score:.3f}", flush=True)
            return score
        except Exception:
            return 1.0e9

    # Coordinate pattern search: test +/- one coordinate at a time, keep the
    # best improvement, and halve the step after an unsuccessful sweep.
    point = starting_point.copy()
    best_score = objective(point)
    step = 0.25 * (UPPER_BOUNDS - LOWER_BOUNDS)
    minimum_step = 1.0e-4 * (UPPER_BOUNDS - LOWER_BOUNDS)
    while evaluations < max_evaluations and np.any(step > minimum_step):
        improved = False
        best_candidate = point
        for dimension in range(3):
            for direction in (-1.0, 1.0):
                if evaluations >= max_evaluations:
                    break
                candidate = point.copy()
                candidate[dimension] = np.clip(
                    candidate[dimension] + direction * step[dimension],
                    LOWER_BOUNDS[dimension],
                    UPPER_BOUNDS[dimension],
                )
                if np.array_equal(candidate, point):
                    continue
                score = objective(candidate)
                if score < best_score:
                    best_score = score
                    best_candidate = candidate
                    improved = True
            if evaluations >= max_evaluations:
                break
        if improved:
            point = best_candidate.copy()
        else:
            step *= 0.5

    optimized = PIDGains(*map(float, point))
    temperature, duty = simulate_pid(name, artifact, optimized, recipe)
    metrics = tracking_metrics(temperature, duty, recipe)
    return optimized, temperature, duty, metrics, evaluations


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize FrED PID gains")
    parser.add_argument(
        "--max-evaluations",
        type=int,
        default=60,
        help="Maximum objective calls per plant model",
    )
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if args.max_evaluations < 4:
        raise ValueError("--max-evaluations must be at least 4")

    recipe = load_recipe()
    models = load_models()
    actual_metrics = tracking_metrics(
        recipe.measured_temperature, recipe.recorded_duty, recipe
    )
    print("Recorded old-PID response")
    print(f"  Gains: {recipe.old_gains.as_dict()}")
    print(
        f"  Tracking MAE={actual_metrics['MAE']:.4f} °C, "
        f"RMSE={actual_metrics['RMSE']:.4f} °C"
    )

    optimized_results: dict[str, dict[str, object]] = {}
    for name, artifact in models.items():
        print(f"\nOptimizing {name}...", flush=True)
        old_temperature, old_duty = simulate_pid(name, artifact, recipe.old_gains, recipe)
        old_metrics = tracking_metrics(old_temperature, old_duty, recipe)
        gains, temperature, duty, metrics, evaluations = optimize_model(
            name,
            artifact,
            recipe,
            recipe.old_gains,
            args.max_evaluations,
        )
        optimized_results[name] = {
            "gains": gains,
            "temperature": temperature,
            "duty": duty,
            "metrics": metrics,
        }
        print(
            f"  Kp={gains.kp:.6g}, Ki={gains.ki:.6g}, Kd={gains.kd:.6g} "
            f"({evaluations} evaluations)"
        )
        print(
            f"  MAE={metrics['MAE']:.4f} °C, RMSE={metrics['RMSE']:.4f} °C, "
            f"mean overshoot={metrics['Mean overshoot']:.4f} °C"
        )
        print(
            f"  Same model with recorded gains: MAE={old_metrics['MAE']:.4f} °C, "
            f"RMSE={old_metrics['RMSE']:.4f} °C"
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    output_path = MODEL_DIR / "optimized_pid_gains.json"
    output_path.write_text(
        json.dumps(
            {
                name: {
                    "gains": result["gains"].as_dict(),
                    "metrics": result["metrics"],
                }
                for name, result in optimized_results.items()
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved optimized gains to: {output_path}")

    if not args.no_plot:
        plt.figure(figsize=(13, 7), constrained_layout=True)
        plt.plot(
            recipe.elapsed,
            recipe.setpoint,
            "k:",
            linewidth=1.4,
            label="Temperature recipe",
        )
        plt.plot(
            recipe.elapsed,
            recipe.measured_temperature,
            color="0.35",
            linewidth=1.5,
            label="Old PID filtered response",
        )
        colors = {"White Box": "tab:orange", "Gray Box": "tab:green", "Black Box": "tab:red"}
        for name, result in optimized_results.items():
            gains = result["gains"]
            plt.plot(
                recipe.elapsed,
                result["temperature"],
                color=colors[name],
                linewidth=1.6,
                label=(
                    f"{name}: Kp={gains.kp:.3g}, Ki={gains.ki:.3g}, "
                    f"Kd={gains.kd:.3g}"
                ),
            )
        plt.title("Automatically Optimized PID Gains by Plant Model")
        plt.xlabel("Elapsed time (s)")
        plt.ylabel("Temperature (°C)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
