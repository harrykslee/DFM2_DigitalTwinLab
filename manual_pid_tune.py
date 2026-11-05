"""Interactive manual PID tuning against the three exported thermal models."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox

from pid_tuning_common import PIDGains, load_models, load_recipe, simulate_pid, tracking_metrics


def main() -> None:
    """Open a three-input UI and model candidate gains when the button is pressed."""
    recipe = load_recipe()
    models = load_models()
    old = recipe.old_gains

    figure, axis = plt.subplots(figsize=(13, 7))
    figure.subplots_adjust(bottom=0.23, right=0.78)
    axis.plot(
        recipe.elapsed,
        recipe.setpoint,
        color="black",
        linestyle=":",
        linewidth=1.4,
        label="Temperature recipe",
    )
    axis.plot(
        recipe.elapsed,
        recipe.measured_temperature,
        color="0.35",
        linewidth=1.5,
        label=(
            "Old PID filtered response "
            f"(Kp={old.kp:g}, Ki={old.ki:g}, Kd={old.kd:g})"
        ),
    )
    colors = {"White Box": "tab:orange", "Gray Box": "tab:green", "Black Box": "tab:red"}
    modeled_lines = {
        name: axis.plot([], [], color=colors[name], linewidth=1.7, label=name)[0]
        for name in models
    }
    axis.set_title("Manual PID Tuning — enter gains and press Model")
    axis.set_xlabel("Elapsed time (s)")
    axis.set_ylabel("Temperature (°C)")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))

    kp_box = TextBox(figure.add_axes([0.08, 0.10, 0.16, 0.055]), "Kp  ", initial=f"{old.kp:g}")
    ki_box = TextBox(figure.add_axes([0.31, 0.10, 0.16, 0.055]), "Ki  ", initial=f"{old.ki:g}")
    kd_box = TextBox(figure.add_axes([0.54, 0.10, 0.16, 0.055]), "Kd  ", initial=f"{old.kd:g}")
    model_button = Button(figure.add_axes([0.76, 0.095, 0.13, 0.065]), "Model")
    status = figure.text(0.08, 0.025, "Ready", fontsize=9, va="bottom")

    def run_models(_event: object) -> None:
        try:
            gains = PIDGains(float(kp_box.text), float(ki_box.text), float(kd_box.text))
            if not all(math.isfinite(value) and value >= 0.0 for value in (gains.kp, gains.ki, gains.kd)):
                raise ValueError("PID gains must be finite and non-negative")
        except ValueError as exc:
            status.set_text(f"Invalid input: {exc}")
            figure.canvas.draw_idle()
            return

        status.set_text("Running all three closed-loop models...")
        figure.canvas.draw_idle()
        summaries: list[str] = []
        try:
            for name, artifact in models.items():
                temperature, duty = simulate_pid(name, artifact, gains, recipe)
                metrics = tracking_metrics(temperature, duty, recipe)
                modeled_lines[name].set_data(recipe.elapsed, temperature)
                summaries.append(
                    f"{name}: MAE={metrics['MAE']:.2f}°C, "
                    f"overshoot={metrics['Mean overshoot']:.2f}°C"
                )
            axis.relim()
            axis.autoscale_view()
            axis.set_title(
                f"Candidate PID: Kp={gains.kp:g}, Ki={gains.ki:g}, Kd={gains.kd:g}"
            )
            status.set_text(" | ".join(summaries))
        except Exception as exc:  # Keep the interactive window alive on errors.
            status.set_text(f"Modeling failed: {exc}")
        figure.canvas.draw_idle()

    model_button.on_clicked(run_models)
    plt.show()


if __name__ == "__main__":
    main()
