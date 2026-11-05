"""PyTorch NARX model for the FrED heat block.

This script trains a nonlinear autoregressive model with exogenous inputs
(NARX) on historical PID experiments and evaluates it on the strictly held-out
``data/test_set.csv`` deployment test.

For each time t, the model input is

    [y(t-1), ..., y(t-p), u(t-1), ..., u(t-p)]

where y is filtered temperature, u is heater command in percent PWM, and p is
the tapped-delay length.  The model predicts absolute filtered temperature.
The most recent sample t-1 is always retained; the remaining taps can be
dilated to cover a longer thermal memory without enlarging the network.  At
the beginning of every file, missing temperature history is padded with that
file's initial ambient temperature and missing PWM history with 0 percent.

Internal validation is one-step-ahead (teacher-forced), which is useful while
training.  The final test is closed-loop: after the known initial temperature,
each predicted temperature is fed back into the next NARX history window.  The
held-out test measurements are never used as model inputs.
"""

from __future__ import annotations

import argparse
import copy
import ctypes
from dataclasses import dataclass
from io import StringIO
import os
from pathlib import Path
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Some Windows CPU wheels contain all required DLLs but do not load their
# dependency chain before importing torch._C.  Keep the DLL directory handle
# alive and preload only the three bundled libraries in dependency order.  The
# block is a no-op on Linux/macOS and on Windows installations without this
# wheel layout.
_TORCH_DLL_DIRECTORY = None
if sys.platform == "win32":
    _torch_library_directory = (
        Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib"
    )
    if _torch_library_directory.is_dir():
        _TORCH_DLL_DIRECTORY = os.add_dll_directory(str(_torch_library_directory))
        for _dll_name in ("c10.dll", "torch_cpu.dll", "torch_python.dll"):
            _dll_path = _torch_library_directory / _dll_name
            if _dll_path.exists():
                ctypes.WinDLL(str(_dll_path))

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Paths, file discovery, and CSV columns
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
TEST_DATA_PATH = DATA_DIR / "test_set.csv"
PREDICTION_DIR = PROJECT_DIR / "predictions"
PREDICTION_PATH = PREDICTION_DIR / "black_box_preds.csv"
MODEL_DIR = PROJECT_DIR / "models"
MODEL_PATH = MODEL_DIR / "black_box_model.pt"

# Patterns are deliberately narrow so unrelated step-response files are not
# silently included.  test_set.csv is explicitly excluded again below.
TRAINING_FILE_PATTERNS = (
    "training_data*.csv",
    "train_*.csv",
)

TIME_COLUMN = "Timestamp (s)"
TEMPERATURE_COLUMN = "Temperature (C)"
DUTY_COLUMN = "Temperature Duty Cycle %"
PID_OUTPUT_COLUMN = "Temperature PID output"

# Same filtering method and durations used by white_box.py and gray_box.py.
MEDIAN_FILTER_SECONDS = 3.0
MOVING_AVERAGE_SECONDS = 8.0

# Conservative defaults for a modest-sized thermal dataset.
DEFAULT_LOOKBACK = 30
DEFAULT_HISTORY_STRIDE = 15
DEFAULT_EPOCHS = 30
DEFAULT_BATCH_SIZE = 512
DEFAULT_LEARNING_RATE = 1.0e-3
DEFAULT_VALIDATION_FRACTION = 0.20
DEFAULT_RANDOM_SEED = 42


@dataclass(frozen=True)
class Standardizer:
    """Mean and standard deviation learned from training data only."""

    mean: float
    std: float

    def normalize(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std

    def denormalize(self, values: np.ndarray) -> np.ndarray:
        return values * self.std + self.mean


class NARXNetwork(nn.Module):
    """Small residual MLP that predicts normalized absolute temperature."""

    def __init__(self, lookback: int) -> None:
        super().__init__()
        input_size = 2 * lookback
        self.correction_network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(p=0.05),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # Feature zero is always y(t-1), including when older taps are dilated.
        return features[:, :1] + self.correction_network(features)


def discover_training_files() -> list[Path]:
    """Find historical PID CSVs without ever including the deployment test."""
    discovered: set[Path] = set()
    for pattern in TRAINING_FILE_PATTERNS:
        discovered.update(DATA_DIR.glob(pattern))

    test_resolved = TEST_DATA_PATH.resolve()
    files = sorted(
        path
        for path in discovered
        if path.is_file() and path.resolve() != test_resolved
    )
    if not files:
        patterns = ", ".join(TRAINING_FILE_PATTERNS)
        raise FileNotFoundError(
            f"No training CSVs found in {DATA_DIR}. Expected one of: {patterns}"
        )
    return files


def read_temperature_section(csv_path: Path) -> pd.DataFrame:
    """Read only the leading TEMPERATURE DATA table from a multi-table log."""
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
        raise ValueError(f"Temperature section in {csv_path} contains no data rows")

    frame = pd.read_csv(StringIO("\n".join(section_lines)))
    frame.columns = frame.columns.str.strip()
    return frame


def odd_window_size(duration: float, sample_interval: float) -> int:
    """Convert a filter duration in seconds to an odd number of samples."""
    size = max(3, int(round(duration / sample_interval)))
    return size if size % 2 == 1 else size + 1


def filter_temperature(time: np.ndarray, temperature: np.ndarray) -> np.ndarray:
    """Apply the shared centered median-plus-moving-average temperature filter."""
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


def load_experiment(csv_path: Path) -> dict[str, object]:
    """Parse, validate, and filter one PID experiment."""
    frame = read_temperature_section(csv_path)
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

    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    invalid = frame[required].isna().any(axis=1)
    if invalid.any():
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
    heater_command = frame[input_column].to_numpy(dtype=float)
    if not all(
        np.all(np.isfinite(values))
        for values in (time, raw_temperature, heater_command)
    ):
        raise ValueError(f"{csv_path.name} contains non-finite required values")
    if np.any(np.diff(time) <= 0.0):
        raise ValueError(f"Timestamps in {csv_path.name} must strictly increase")
    if np.any((heater_command < 0.0) | (heater_command > 100.0)):
        raise ValueError(f"Heater commands in {csv_path.name} must be 0 to 100 percent")

    filtered_temperature = filter_temperature(time, raw_temperature)
    return {
        "name": csv_path.name,
        "time": time,
        "raw_temperature": raw_temperature,
        "temperature": filtered_temperature,
        "heater_command": heater_command,
        "input_column": input_column,
    }


def exact_overlap_count(
    candidate: dict[str, object], held_out: dict[str, object]
) -> int:
    """Count exact raw samples shared with the held-out deployment experiment.

    Timestamp, unfiltered temperature, and heater command must all match.  The
    full-precision triple makes accidental matches between independent logs
    extremely unlikely.  Raw values are used so filtering cannot conceal or
    manufacture an overlap.
    """
    held_out_samples = set(
        zip(
            held_out["time"].tolist(),
            held_out["raw_temperature"].tolist(),
            held_out["heater_command"].tolist(),
        )
    )
    candidate_samples = zip(
        candidate["time"].tolist(),
        candidate["raw_temperature"].tolist(),
        candidate["heater_command"].tolist(),
    )
    return sum(sample in held_out_samples for sample in candidate_samples)


def make_narx_samples(
    temperature: np.ndarray,
    heater_command: np.ndarray,
    lookback: int,
    history_stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build padded, optionally dilated histories and absolute targets.

    Lag one is always present.  With stride greater than one, older offsets are
    ``[1, stride, 2*stride, ...]``.  Temperature and PWM use identical offsets,
    and no feature can see y(t) or u(t).  A fresh padded history is constructed
    for every file, so windows cannot cross experiment boundaries.
    """
    if lookback < 1:
        raise ValueError("lookback must be at least 1")
    if history_stride < 1:
        raise ValueError("history_stride must be at least 1")
    if temperature.size != heater_command.size:
        raise ValueError("Temperature and heater command lengths must match")

    if history_stride == 1:
        lag_offsets = np.arange(1, lookback + 1, dtype=int)
    else:
        lag_offsets = np.concatenate(
            (np.array([1], dtype=int), history_stride * np.arange(1, lookback))
        )
    maximum_lag = int(lag_offsets[-1])
    padded_temperature = np.concatenate(
        (np.full(maximum_lag, temperature[0]), temperature)
    )
    padded_command = np.concatenate((np.zeros(maximum_lag), heater_command))
    sample_positions = maximum_lag + np.arange(temperature.size)
    history_indices = sample_positions[:, None] - lag_offsets[None, :]
    temperature_history = padded_temperature[history_indices]
    command_history = padded_command[history_indices]
    features = np.concatenate((temperature_history, command_history), axis=1)
    targets = temperature.reshape(-1, 1)
    return features, targets


def sanity_check_lag_indexing() -> None:
    """Prove that a feature row cannot see current y(t) or current u(t).

    At t=2 with stride two, the lag offsets are [1, 2, 4].  The newest command
    must be u(1)=30—not current u(2)=40—and the oldest tap must be padding.
    """
    temperature = np.array([10.0, 11.0, 13.0])
    command = np.array([20.0, 30.0, 40.0])
    features, targets = make_narx_samples(
        temperature, command, lookback=3, history_stride=2
    )
    expected_t2_features = np.array([11.0, 10.0, 10.0, 30.0, 20.0, 0.0])
    if not np.array_equal(features[2], expected_t2_features):
        raise RuntimeError(
            "NARX lag-index sanity check failed: features must contain only "
            "y(t-1..t-p) and u(t-1..t-p)."
        )
    if not np.isclose(targets[2, 0], 13.0):
        raise RuntimeError("NARX absolute-target sanity check failed")


def split_training_experiments(
    experiments: list[dict[str, object]],
    lookback: int,
    history_stride: int,
    validation_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Chronologically split each historical experiment, then concatenate."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")

    train_features: list[np.ndarray] = []
    train_targets: list[np.ndarray] = []
    validation_features: list[np.ndarray] = []
    validation_targets: list[np.ndarray] = []

    for experiment in experiments:
        features, targets = make_narx_samples(
            experiment["temperature"],
            experiment["heater_command"],
            lookback,
            history_stride,
        )
        split_index = int(round((1.0 - validation_fraction) * len(targets)))
        split_index = min(max(1, split_index), len(targets) - 1)
        train_features.append(features[:split_index])
        train_targets.append(targets[:split_index])
        validation_features.append(features[split_index:])
        validation_targets.append(targets[split_index:])

    return (
        np.concatenate(train_features),
        np.concatenate(train_targets),
        np.concatenate(validation_features),
        np.concatenate(validation_targets),
    )


def fit_standardizer(training_values: np.ndarray) -> Standardizer:
    """Fit scaling from training values, never validation or test data."""
    mean = float(np.mean(training_values))
    std = float(np.std(training_values))
    if not np.isfinite(std) or std < 1.0e-8:
        raise ValueError("Training values do not have usable variance")
    return Standardizer(mean, std)


def normalize_samples(
    features: np.ndarray,
    targets: np.ndarray,
    lookback: int,
    temperature_standardizer: Standardizer,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize temperature histories/targets and scale PWM to 0..1."""
    normalized_features = np.asarray(features, dtype=np.float32).copy()
    normalized_features[:, :lookback] = temperature_standardizer.normalize(
        normalized_features[:, :lookback]
    )
    normalized_features[:, lookback:] /= 100.0
    normalized_targets = temperature_standardizer.normalize(targets).astype(np.float32)
    return normalized_features, normalized_targets


def make_loader(
    features: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Create a PyTorch DataLoader without copying tensors unnecessarily."""
    dataset = TensorDataset(torch.from_numpy(features), torch.from_numpy(targets))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)




def train_model(
    model: NARXNetwork,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
) -> tuple[list[float], list[float]]:
    """Train one-step absolute temperature MSE and restore the best checkpoint."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1.0e-6)
    model.to(device)

    training_history: list[float] = []
    validation_history: list[float] = []
    best_validation_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    early_stopping_patience = 12

    print("\nTraining NARX model (normalized absolute-temperature MSE)")
    for epoch in range(1, epochs + 1):
        model.train()
        training_loss_sum = 0.0
        training_count = 0
        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features), targets)
            loss.backward()
            optimizer.step()
            training_loss_sum += loss.item() * targets.size(0)
            training_count += targets.size(0)

        model.eval()
        validation_loss_sum = 0.0
        validation_count = 0
        with torch.no_grad():
            for features, targets in validation_loader:
                features = features.to(device)
                targets = targets.to(device)
                loss = criterion(model(features), targets)
                validation_loss_sum += loss.item() * targets.size(0)
                validation_count += targets.size(0)

        training_loss = training_loss_sum / training_count
        validation_loss = validation_loss_sum / validation_count
        training_history.append(training_loss)
        validation_history.append(validation_loss)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(
                f"  Epoch {epoch:3d}/{epochs}: "
                f"train MSE={training_loss:.6f}, validation MSE={validation_loss:.6f}"
            )

        if epochs_without_improvement >= early_stopping_patience:
            print(f"  Early stopping at epoch {epoch}; restoring best validation model.")
            break

    model.load_state_dict(best_state)
    return training_history, validation_history


def predict_one_step(
    model: NARXNetwork,
    loader: DataLoader,
    device: torch.device,
    temperature_standardizer: Standardizer,
) -> tuple[np.ndarray, np.ndarray]:
    """Return denormalized one-step temperature predictions and targets."""
    model.eval()
    predicted_batches: list[np.ndarray] = []
    target_batches: list[np.ndarray] = []
    with torch.no_grad():
        for features, targets in loader:
            predicted_batches.append(model(features.to(device)).cpu().numpy().ravel())
            target_batches.append(targets.numpy().ravel())

    return (
        temperature_standardizer.denormalize(np.concatenate(predicted_batches)),
        temperature_standardizer.denormalize(np.concatenate(target_batches)),
    )


def predict_closed_loop(
    model: NARXNetwork,
    experiment: dict[str, object],
    lookback: int,
    history_stride: int,
    temperature_standardizer: Standardizer,
    device: torch.device,
) -> np.ndarray:
    """Run iterative deployment prediction using only predicted output history."""
    measured = experiment["temperature"]
    heater_command = experiment["heater_command"]
    predictions = np.empty_like(measured, dtype=float)

    # The first filtered temperature is the known deployment initial condition.
    predictions[0] = measured[0]
    if history_stride == 1:
        lag_offsets = np.arange(1, lookback + 1, dtype=int)
    else:
        lag_offsets = np.concatenate(
            (np.array([1], dtype=int), history_stride * np.arange(1, lookback))
        )

    model.eval()
    with torch.no_grad():
        for index in range(1, measured.size):
            history_indices = index - lag_offsets
            recent_temperature = np.where(
                history_indices >= 0,
                predictions[np.maximum(history_indices, 0)],
                measured[0],
            ).astype(np.float32)
            recent_command = np.where(
                history_indices >= 0,
                heater_command[np.maximum(history_indices, 0)],
                0.0,
            ).astype(np.float32)
            features = np.concatenate(
                (
                    temperature_standardizer.normalize(recent_temperature),
                    recent_command / 100.0,
                )
            ).reshape(1, -1)
            normalized_temperature = model(
                torch.from_numpy(features).to(device)
            ).item()
            predictions[index] = float(
                temperature_standardizer.denormalize(
                    np.asarray(normalized_temperature)
                )
            )

    return predictions


def error_metrics(measured: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    """Return MAE and RMSE in degrees C."""
    residual = predicted - measured
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual**2)))
    return mae, rmse


def plot_test_result(
    time: np.ndarray, measured: np.ndarray, predicted: np.ndarray
) -> None:
    """Plot held-out filtered temperature against closed-loop NARX prediction."""
    plt.figure(figsize=(12, 6), constrained_layout=True)
    plt.plot(
        time,
        measured,
        color="tab:blue",
        linewidth=1.5,
        label="Actual Filtered Temperature (Test Set)",
    )
    plt.plot(
        time,
        predicted,
        color="tab:red",
        linewidth=1.8,
        label="NARX Model Prediction (Closed Loop)",
    )
    plt.title("FrED Heat-Block NARX Deployment Test")
    plt.xlabel("Timestamp (s)")
    plt.ylabel("Temperature (°C)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()


def export_predictions(
    time: np.ndarray, measured: np.ndarray, predicted: np.ndarray
) -> None:
    """Write held-out deployment predictions for the master comparison script."""
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    output = pd.DataFrame(
        {
            TIME_COLUMN: time,
            "Measured_Temp": measured,
            "BlackBox_Temp": predicted,
        }
    )
    output.to_csv(PREDICTION_PATH, index=False)
    print(f"Saved deployment predictions to: {PREDICTION_PATH}")


def set_random_seed(seed: int) -> None:
    """Make CPU/GPU initialization and data shuffling reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_arguments() -> argparse.Namespace:
    """Expose useful training controls while keeping classroom defaults simple."""
    parser = argparse.ArgumentParser(description="Train and test a PyTorch NARX model")
    parser.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK)
    parser.add_argument(
        "--history-stride",
        type=int,
        default=DEFAULT_HISTORY_STRIDE,
        help="Spacing between older history taps; lag t-1 is always retained",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument(
        "--validation-fraction", type=float, default=DEFAULT_VALIDATION_FRACTION
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument(
        "--no-plot", action="store_true", help="Skip the final interactive plot"
    )
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    """Fail early on invalid hyperparameters."""
    if args.lookback < 1:
        raise ValueError("--lookback must be at least 1")
    if args.history_stride < 1:
        raise ValueError("--history-stride must be at least 1")
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be positive")
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be between 0 and 1")


def main() -> None:
    """Train, validate, deploy, plot, and export the NARX model."""
    args = parse_arguments()
    validate_arguments(args)
    set_random_seed(args.seed)
    sanity_check_lag_indexing()
    print("NARX lag-index sanity check: PASS (only past y and past u are visible)")

    training_paths = discover_training_files()
    print("Candidate historical training files:")
    for path in training_paths:
        print(f"  - {path.name}")
    print(f"Strictly held-out deployment file: {TEST_DATA_PATH.name}")

    test_experiment = load_experiment(TEST_DATA_PATH)
    training_experiments: list[dict[str, object]] = []
    for path in training_paths:
        experiment = load_experiment(path)
        overlap = exact_overlap_count(experiment, test_experiment)
        if overlap:
            print(
                f"Warning: excluding {path.name} because it contains {overlap:,} "
                "exact held-out test samples."
            )
            continue
        training_experiments.append(experiment)

    if not training_experiments:
        raise ValueError(
            "Every discovered training file overlaps test_set.csv; provide an "
            "independent historical PID training file."
        )
    print("Training files accepted after held-out overlap checks:")
    for experiment in training_experiments:
        print(f"  - {experiment['name']}")
    train_x, train_y, validation_x, validation_y = split_training_experiments(
        training_experiments,
        args.lookback,
        args.history_stride,
        args.validation_fraction,
    )

    # Fit temperature scaling from the training partition only.
    temperature_standardizer = fit_standardizer(train_y)
    train_x, train_y = normalize_samples(
        train_x, train_y, args.lookback, temperature_standardizer
    )
    validation_x, validation_y = normalize_samples(
        validation_x,
        validation_y,
        args.lookback,
        temperature_standardizer,
    )
    train_loader = make_loader(train_x, train_y, args.batch_size, shuffle=True)
    validation_loader = make_loader(
        validation_x, validation_y, args.batch_size, shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"\nSamples: {len(train_y):,} training, {len(validation_y):,} validation; "
        f"lookback={args.lookback}; stride={args.history_stride}; device={device}"
    )
    model = NARXNetwork(args.lookback)
    train_model(
        model,
        train_loader,
        validation_loader,
        device,
        args.epochs,
        args.learning_rate,
    )

    # Persist the best validation checkpoint and its input scaling so the PID
    # tuning scripts can run the same NARX model without retraining it.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_type": "black_box_narx",
            "version": 1,
            "state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
            "lookback": args.lookback,
            "history_stride": args.history_stride,
            "temperature_mean": temperature_standardizer.mean,
            "temperature_std": temperature_standardizer.std,
            "median_filter_seconds": MEDIAN_FILTER_SECONDS,
            "moving_average_seconds": MOVING_AVERAGE_SECONDS,
        },
        MODEL_PATH,
    )
    print(f"Saved reusable black-box model to: {MODEL_PATH}")

    validation_prediction, validation_target = predict_one_step(
        model,
        validation_loader,
        device,
        temperature_standardizer,
    )
    validation_mae, validation_rmse = error_metrics(
        validation_target, validation_prediction
    )
    print("\nInternal chronological validation (one-step ahead)")
    print(f"  MAE:  {validation_mae:.4f} °C")
    print(f"  RMSE: {validation_rmse:.4f} °C")

    test_prediction = predict_closed_loop(
        model,
        test_experiment,
        args.lookback,
        args.history_stride,
        temperature_standardizer,
        device,
    )
    test_measured = test_experiment["temperature"]
    test_mae, test_rmse = error_metrics(test_measured, test_prediction)
    print("\nHeld-out deployment test (closed-loop multi-step prediction)")
    print(f"  File: {TEST_DATA_PATH.name}")
    print(f"  Samples: {len(test_measured):,}")
    print(f"  MAE:  {test_mae:.4f} °C")
    print(f"  RMSE: {test_rmse:.4f} °C")

    export_predictions(test_experiment["time"], test_measured, test_prediction)
    if not args.no_plot:
        plot_test_result(test_experiment["time"], test_measured, test_prediction)


if __name__ == "__main__":
    main()
