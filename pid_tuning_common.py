"""Shared model loading, recipe parsing, PID simulation, and scoring utilities."""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from black_box import NARXNetwork, Standardizer


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"
TEST_DATA_PATH = DATA_DIR / "test_set.csv"

TIME_COLUMN = "Timestamp (s)"
TEMPERATURE_COLUMN = "Temperature (C)"
SETPOINT_COLUMN = "Temperature setpoint (C)"
PID_OUTPUT_COLUMN = "Temperature PID output"
KP_COLUMN = "Temperature Kp"
KI_COLUMN = "Temperature Ki"
KD_COLUMN = "Temperature Kd"

MEDIAN_FILTER_SECONDS = 3.0
MOVING_AVERAGE_SECONDS = 8.0


@dataclass(frozen=True)
class PIDGains:
    kp: float
    ki: float
    kd: float

    def as_dict(self) -> dict[str, float]:
        return {"Kp": self.kp, "Ki": self.ki, "Kd": self.kd}


@dataclass(frozen=True)
class Recipe:
    time: np.ndarray
    elapsed: np.ndarray
    dt: np.ndarray
    setpoint: np.ndarray
    measured_temperature: np.ndarray
    recorded_duty: np.ndarray
    old_gains: PIDGains


@dataclass(frozen=True)
class BlackBoxArtifact:
    model: NARXNetwork
    lookback: int
    history_stride: int
    temperature_standardizer: Standardizer


def read_temperature_section(csv_path: Path) -> pd.DataFrame:
    """Read only the leading temperature table from a FrED multi-table CSV."""
    lines = csv_path.read_text(encoding="utf-8-sig").splitlines()
    try:
        label_index = next(
            i for i, line in enumerate(lines) if line.strip().upper() == "TEMPERATURE DATA"
        )
    except StopIteration as exc:
        raise ValueError(f"No TEMPERATURE DATA section found in {csv_path}") from exc

    header_index = label_index + 1
    while header_index < len(lines) and not lines[header_index].strip():
        header_index += 1
    section: list[str] = []
    for line in lines[header_index:]:
        if not line.strip():
            break
        section.append(line)
    frame = pd.read_csv(StringIO("\n".join(section)))
    frame.columns = frame.columns.str.strip()
    return frame


def _odd_window(duration: float, median_dt: float) -> int:
    size = max(3, int(round(duration / median_dt)))
    return size if size % 2 else size + 1


def filter_temperature(time: np.ndarray, temperature: np.ndarray) -> np.ndarray:
    """Use the same centered median and moving-average filter as all models."""
    median_dt = float(np.median(np.diff(time)))
    median_window = _odd_window(MEDIAN_FILTER_SECONDS, median_dt)
    average_window = _odd_window(MOVING_AVERAGE_SECONDS, median_dt)
    filtered = pd.Series(temperature).rolling(
        median_window, center=True, min_periods=1
    ).median()
    filtered = filtered.rolling(average_window, center=True, min_periods=1).mean()
    return filtered.to_numpy(dtype=float)


def load_recipe(csv_path: Path = TEST_DATA_PATH) -> Recipe:
    """Load setpoint recipe, filtered legacy response, duty, and old PID gains."""
    frame = read_temperature_section(csv_path)
    required = [
        TIME_COLUMN,
        TEMPERATURE_COLUMN,
        SETPOINT_COLUMN,
        PID_OUTPUT_COLUMN,
        KP_COLUMN,
        KI_COLUMN,
        KD_COLUMN,
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_path.name}: {missing}")
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(
        subset=[TIME_COLUMN, TEMPERATURE_COLUMN, SETPOINT_COLUMN, PID_OUTPUT_COLUMN]
    ).reset_index(drop=True)

    time = frame[TIME_COLUMN].to_numpy(dtype=float)
    if len(time) < 2 or np.any(np.diff(time) <= 0.0):
        raise ValueError("Recipe timestamps must be strictly increasing")
    measured = filter_temperature(
        time, frame[TEMPERATURE_COLUMN].to_numpy(dtype=float)
    )
    gains = PIDGains(
        kp=float(frame[KP_COLUMN].dropna().median()),
        ki=float(frame[KI_COLUMN].dropna().median()),
        kd=float(frame[KD_COLUMN].dropna().median()),
    )
    return Recipe(
        time=time,
        elapsed=time - time[0],
        dt=np.diff(time),
        setpoint=frame[SETPOINT_COLUMN].to_numpy(dtype=float),
        measured_temperature=measured,
        recorded_duty=frame[PID_OUTPUT_COLUMN].to_numpy(dtype=float),
        old_gains=gains,
    )


def load_models() -> dict[str, object]:
    """Load all exported plant models and fail clearly if one is missing."""
    white_path = MODEL_DIR / "white_box_model.json"
    gray_path = MODEL_DIR / "gray_box_model.json"
    black_path = MODEL_DIR / "black_box_model.pt"
    missing = [path for path in (white_path, gray_path, black_path) if not path.exists()]
    if missing:
        names = ", ".join(path.name for path in missing)
        raise FileNotFoundError(
            f"Missing model artifacts: {names}. Run white_box.py, gray_box.py, "
            "and black_box.py first."
        )

    white = json.loads(white_path.read_text(encoding="utf-8"))
    gray = json.loads(gray_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(black_path, map_location="cpu", weights_only=False)
    black_model = NARXNetwork(int(checkpoint["lookback"]))
    black_model.load_state_dict(checkpoint["state_dict"])
    black_model.eval()
    black = BlackBoxArtifact(
        model=black_model,
        lookback=int(checkpoint["lookback"]),
        history_stride=int(checkpoint["history_stride"]),
        temperature_standardizer=Standardizer(
            float(checkpoint["temperature_mean"]),
            float(checkpoint["temperature_std"]),
        ),
    )
    return {"White Box": white, "Gray Box": gray, "Black Box": black}


def _pid_command(
    setpoint: float,
    temperature: float,
    dt: float,
    gains: PIDGains,
    integral: float,
    previous_error: float | None,
) -> tuple[float, float, float]:
    """Calculate a saturated PID command with conditional anti-windup."""
    error = setpoint - temperature
    derivative = 0.0 if previous_error is None else (error - previous_error) / dt
    candidate_integral = integral + error * dt
    raw = gains.kp * error + gains.ki * candidate_integral + gains.kd * derivative
    duty = float(np.clip(raw, 0.0, 100.0))

    pushes_further_into_saturation = (raw > 100.0 and error > 0.0) or (
        raw < 0.0 and error < 0.0
    )
    if pushes_further_into_saturation:
        raw = gains.kp * error + gains.ki * integral + gains.kd * derivative
        duty = float(np.clip(raw, 0.0, 100.0))
    else:
        integral = candidate_integral
    return duty, integral, error


def _lag_offsets(lookback: int, stride: int) -> np.ndarray:
    if stride == 1:
        return np.arange(1, lookback + 1, dtype=int)
    return np.concatenate((np.array([1]), stride * np.arange(1, lookback)))


def simulate_pid(
    model_name: str,
    artifact: object,
    gains: PIDGains,
    recipe: Recipe,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the specified exported plant in closed loop with candidate PID gains."""
    if min(gains.kp, gains.ki, gains.kd) < 0.0:
        raise ValueError("PID gains must be non-negative")
    temperature = np.empty_like(recipe.measured_temperature)
    duty = np.zeros_like(temperature)
    temperature[0] = recipe.measured_temperature[0]
    integral = 0.0
    previous_error: float | None = None

    if model_name == "White Box":
        c_th = float(artifact["C_th_J_per_K"])
        r_th = float(artifact["R_th_K_per_W"])
        q_max = float(artifact["Q_max_W"])
        ambient = float(temperature[0])
    elif model_name == "Gray Box":
        gain = float(artifact["gain_C_per_percent_PWM"])
        tau = float(artifact["tau_seconds"])
        theta = float(artifact["theta_seconds"])
        baseline = float(temperature[0])
    elif model_name == "Black Box":
        lag_offsets = _lag_offsets(artifact.lookback, artifact.history_stride)
    else:
        raise KeyError(f"Unknown model: {model_name}")

    with torch.inference_mode():
        for index in range(1, temperature.size):
            interval = float(recipe.dt[index - 1])
            command, integral, previous_error = _pid_command(
                recipe.setpoint[index - 1],
                temperature[index - 1],
                interval,
                gains,
                integral,
                previous_error,
            )
            duty[index - 1] = command

            if model_name == "White Box":
                equilibrium = ambient + r_th * q_max * command / 100.0
                decay = np.exp(-interval / (c_th * r_th))
                temperature[index] = equilibrium + (
                    temperature[index - 1] - equilibrium
                ) * decay
            elif model_name == "Gray Box":
                delayed_time = recipe.elapsed[index - 1] - theta
                delayed_index = int(
                    np.searchsorted(recipe.elapsed[:index], delayed_time, side="right")
                    - 1
                )
                delayed_command = duty[delayed_index] if delayed_index >= 0 else 0.0
                equilibrium = baseline + gain * delayed_command
                decay = np.exp(-interval / tau)
                temperature[index] = equilibrium + (
                    temperature[index - 1] - equilibrium
                ) * decay
            else:
                history_indices = index - lag_offsets
                temperature_history = np.where(
                    history_indices >= 0,
                    temperature[np.maximum(history_indices, 0)],
                    temperature[0],
                ).astype(np.float32)
                command_history = np.where(
                    history_indices >= 0,
                    duty[np.maximum(history_indices, 0)],
                    0.0,
                ).astype(np.float32)
                features = np.concatenate(
                    (
                        artifact.temperature_standardizer.normalize(
                            temperature_history
                        ),
                        command_history / 100.0,
                    )
                ).astype(np.float32)[None, :]
                normalized = artifact.model(torch.from_numpy(features)).item()
                temperature[index] = float(
                    artifact.temperature_standardizer.denormalize(
                        np.asarray(normalized)
                    )
                )

    duty[-1] = duty[-2]
    return temperature, duty


def tracking_metrics(
    temperature: np.ndarray, duty: np.ndarray, recipe: Recipe
) -> dict[str, float]:
    """Calculate consistent recipe-tracking and control-effort metrics."""
    error = temperature - recipe.setpoint
    time_weighted_mae = float(
        np.sum(np.abs(error[:-1]) * recipe.dt) / np.sum(recipe.dt)
    )
    rmse = float(np.sqrt(np.mean(error**2)))
    overshoot = float(np.mean(np.maximum(error, 0.0)))
    control_variation = float(np.mean(np.abs(np.diff(duty))))
    score = time_weighted_mae + 2.0 * overshoot + 0.02 * control_variation
    return {
        "MAE": time_weighted_mae,
        "RMSE": rmse,
        "Mean overshoot": overshoot,
        "Control variation": control_variation,
        "Objective": score,
    }

