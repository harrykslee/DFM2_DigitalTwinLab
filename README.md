# FrED Heat Block Digital Twin and PID Tuning Lab

**Duration:** 90 minutes

**Goal:** Build three models of the FrED heat block, use them to propose PID gains, and check one proposed controller against a new physical experiment.

## Why this lab?

A digital twin predicts how a physical system responds to an input. Here, the input is heater duty cycle (percent PWM), and the output is heat block temperature. You will work with three ways to describe the same system:

| Model | What it knows | Main idea |
| --- | --- | --- |
| White box | Heat storage, heat loss, heater power | Calculate temperature from an energy balance. |
| Gray box | First order response and delay | Estimate a few meaningful parameters from a 10% PWM step experiment. |
| Black box | Historical temperature and heater data | Train a neural network to predict the next temperature. |

The first four activities ask how well each model reproduces recorded experiments. The PID activities pose an **inverse problem**: instead of predicting temperature from known heater input, choose controller gains $(K_p,K_i,K_d)$ that produce a desirable temperature response. The manual tool lets you try gains; the automatic tool searches for gains on each model. A prediction is a proposal until a new FrED run tests it.

**Error measures:** Mean absolute error (MAE) is the average size of a temperature error; root mean squared error (RMSE) gives larger errors more weight. In model comparison, the error is *prediction minus measured temperature*. In PID tuning, the error is *temperature minus setpoint*. Keep those two comparisons distinct in your write up.

## Before you start (about 5 minutes)

Work from the repository root in a terminal. On Windows PowerShell, set up and activate a virtual environment if one is not already available:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The commands below assume the environment is active. The model scripts expect `data/test_set.csv` and **`data/step_PWM10.csv`**. If the 10% step file is missing, obtain it from your instructor before Steps 1 and 2. The black box also needs one or more independent historical files named `training_data*.csv`, `train_*.csv`, or `pid_train*.csv`. Never use `test_set.csv` or your new validation run for neural network training.

The historical `test_set.csv` recipe lasts about **139 minutes**. It is used for computer simulations today. For the physical trial in Step 7, use a shorter instructor approved recipe that fits the remaining lab time; `pid_validation.py` reads the setpoints and timestamps from that new run.

Suggested pacing: setup 5 min; white box 10; gray box 12; black box 12; compare 8; manual tuning 10; automatic tuning 10; physical trial and validation 23. Start the computer steps promptly so the final run has time to finish.

## 1. White box model (about 10 minutes)

The heat balance treats the block as one temperature:

$$
C_{th}\frac{dT}{dt}=\frac{u}{100}Q_{max}-\frac{T-T_{\infty}}{R_{th}}.
$$

Here $C_{th}$ is thermal capacitance, $R_{th}$ is thermal resistance, $Q_{max}$ is heater power at 100% PWM, and $u$ is the recorded duty cycle. The initial measured temperature supplies the ambient/initial condition. This script uses fixed physical parameters; it does not fit them to the test data.

```powershell
python white_box.py
```

Inspect the 10% step and multi setpoint test plots. Record the MAE and RMSE for both experiments. Save or screenshot the figures for your write up. The script saves `models/white_box_model.json` and `predictions/white_box_preds.csv`.

**Reflect:** Where does the fixed physics model miss the measured response most clearly: during heating, near steady temperature, or after setpoint changes?

## 2. Gray box model (about 12 minutes)

The first order plus dead time (FOPDT) model is

$$
G(s)=\frac{K}{\tau s+1}e^{-\theta s}.
$$

$K$ is temperature gain per percent PWM, $\tau$ is the response time constant, and $\theta$ is the delay. The script smooths the 10% step temperature with a centered median filter followed by a moving average. It then estimates the parameters mathematically using the 28.3% and 63.2% response times.

```powershell
python gray_box.py
```

Look at the first step response plot **before closing it**. Estimate $K\approx\Delta T/\Delta u$, the delay $\theta$, and the time $\tau$ needed to reach about 63.2% of the full rise *after the delay*. Record your three estimates, then close that plot and enter them at the prompts. Compare your curves and errors with the calculated parameter curves for both experiments. Save or screenshot the identification and comparison plots. The calculated model is saved as `models/gray_box_model.json`; only its test prediction is saved in `predictions/gray_box_preds.csv`.

| Parameter | Your estimate | Calculated value |
| --- | ---: | ---: |
| $K$ (°C per % PWM) |  |  |
| $\tau$ (s) |  |  |
| $\theta$ (s) |  |  |

**Reflect:** Which parameter was hardest to judge from the plot, and how did that difference affect the predicted curve?

## 3. Black box model (about 12 minutes)

The neural network is a NARX model: it uses earlier filtered temperatures and earlier heater commands to predict temperature. It keeps the most recent sample and spaces older history taps to cover a longer thermal response. Training and internal validation use historical PID files; `test_set.csv` is held out for the final multi step prediction.

```powershell
python black_box.py
```

Watch the training and internal validation losses, then record the held out test MAE and RMSE. Save or screenshot the final prediction plot. The script saves `models/black_box_model.pt` and `predictions/black_box_preds.csv`. Training may take several minutes; use the provided defaults for this short lab.

**Reflect:** Does a low one step validation loss guarantee a low error after the network repeatedly uses its own predictions? What do you see here?

## 4. Compare the models (about 8 minutes)

Run this only after the three prediction files have been regenerated from the same `test_set.csv`:

```powershell
python compare_models.py
```

The script checks timestamp alignment, compares each prediction with the filtered measured temperature, and reports model to model disagreement. Record the MAE, RMSE, and bias for each model. Save or screenshot the comparison and residual plot. A metrics table is also saved as `predictions/model_comparison_metrics.csv`.

**Reflect:** Which model would you trust most for trying new PID gains, and what evidence in the error plot supports your choice?

## 5. Manually tune PID gains (about 10 minutes)

The PID controller combines proportional, integral, and derivative action. In the simulation, a candidate controller sees the setpoint recipe from `test_set.csv`, computes its own duty cycle, and drives each saved model. The gray line is the *recorded, filtered response with the old gains*; the colored lines are simulations with your proposed gains.

```powershell
python manual_pid_tune.py
```

Enter three nonnegative numbers in the **Kp**, **Ki**, and **Kd** boxes and press **Model**. Try the recorded gains first, then one or two deliberate changes. Compare tracking error, overshoot, and the shape of each response. Record one candidate set and save or screenshot the plot with that set visible.

**Reflect:** Did the same gain change help all three models? Describe one difference.

## 6. Automatically tune PID gains (about 10 minutes)

The automatic tool solves the inverse problem with a bounded coordinate search. It tries gains separately on each saved model, reruns the simulated controller, and seeks to reduce a score made from time weighted tracking MAE, mean temperature above setpoint, and changes in heater command:

$$
J=\text{tracking MAE}+2\,(\text{mean overshoot})+0.02\,(\text{mean duty change}).
$$

```powershell
python auto_pid_tune.py
```

The default search uses at most 60 evaluations per model. It plots the three optimized simulated responses against the recipe and old recorded response, and saves gains in `models/optimized_pid_gains.json`. Record all three suggestions below. The search is limited to $0\leq K_p\leq5$, $0\leq K_i\leq0.05$, and $0\leq K_d\leq2$. A result at a limit is the best found **within those limits**, not proof of a global optimum. Save or screenshot the plot.

| Gain set | $K_p$ | $K_i$ | $K_d$ |
| --- | ---: | ---: | ---: |
| Recorded old controller |  |  |  |
| Your manual candidate |  |  |  |
| White box suggestion |  |  |  |
| Gray box suggestion |  |  |  |
| Black box suggestion |  |  |  |
| **Chosen for physical trial** |  |  |  |

**Reflect:** Which gains will you try on FrED, and why? Use both the model comparison and tuning plots in your answer.

## 7. Test on FrED and validate (about 23 minutes)

With your instructor, enter the **chosen** gains on FrED and run the shorter approved temperature recipe. Record the actual gains used, including any change made at the bench. Export the *new* recording as `data/pid_validation.csv`. The required columns are `Timestamp (s)`, `Temperature (C)`, and `Temperature setpoint (C)`; the FrED sectioned export also works. A recorded heater command column is useful for checking the controller simulation.

```powershell
python pid_validation.py
```

Enter the gains **actually used** when prompted. The script filters the new measured temperature, runs each saved model with the new recording's setpoints and timestamps, prints prediction errors against the physical response, plots all four temperature curves, and saves `predictions/pid_validation_preds.csv`. Save or screenshot the validation plot.

**Reflect:** Which model best predicted the new run? If the physical response differs from every prediction, name one plausible source of that difference (plant model, controller behavior, heater command, or measurement).

## What to submit

Submit a short write up with your recorded parameters, model errors, manual and automatic gain table, chosen physical gains, and brief answers to the reflection prompts. Include saved plots or screenshots with captions identifying the dataset, model, and gains. Distinguish **simulated tracking** from the **new measured FrED response** when drawing your conclusion.

`models/`, `predictions/`, and `data/pid_validation.csv` are ignored by Git. Keep your figures and new recording with your lab submission rather than relying on a Git push to include them.

