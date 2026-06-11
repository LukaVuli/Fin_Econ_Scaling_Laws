# Scaling Law Estimator

A configurable TensorFlow/Keras experiment runner for neural-network scaling-law
studies. The estimator trains the same model family at many target parameter
counts, records validation/test performance, estimates compute in PF-days, and
optionally evaluates portfolio strategies from model forecasts.

This repository currently ships the estimator as one Python module:

```text
Scaling_Law_Estimator.py
```

When published as a package, the examples below can use the package import.
Until then, run the examples from this folder and import directly from
`Scaling_Law_Estimator`.

## What The Code Does

At a high level, the estimator:

1. Builds dense neural networks at requested parameter counts.
2. Trains each network on train/validation/test data.
3. Saves results incrementally, so long experiments can resume safely.
4. Tracks training curves, final losses, R2, wall-clock time, and compute.
5. Optionally converts predictions into panel or time-series portfolio returns.
6. Fits and plots scaling-law curves from saved experiment outputs.

The main public classes are:

```python
ScalingLawConfig       # all experiment configuration
ScalingLawExperiment   # train models and save results
ScalingLawPlotter      # load saved results and make plots
PortfolioAnalyzer      # standalone portfolio analysis helpers
DataSplitter           # dataframe split and missing-data logic
ResultsManager         # output artifact persistence
```

## Installation

For a future published package:

```bash
pip install scaling-law-estimator
```

For the current single-file repository, install the scientific dependencies and
run scripts from this directory:

```bash
pip install tensorflow numpy pandas scipy matplotlib psutil
```

`psutil` is optional. It is only used for memory diagnostics.

## Imports

Use this import style in the current folder:

```python
from Scaling_Law_Estimator import (
    ScalingLawConfig,
    ScalingLawExperiment,
    ScalingLawPlotter,
    NormalizationType,
    ArchitectureMode,
    InitializerType,
    ResumeMode,
    SplitMode,
    MissingDataPolicy,
)
```

After packaging, the same API should be exported by the package:

```python
from scaling_law_estimator import ScalingLawConfig, ScalingLawExperiment
```

## Simplest Possible Run

The lowest-friction path is to pass already-split NumPy arrays.

```python
import numpy as np

from Scaling_Law_Estimator import ScalingLawConfig, ScalingLawExperiment

rng = np.random.default_rng(42)

X_train = rng.normal(size=(1000, 20)).astype("float32")
y_train = rng.normal(size=1000).astype("float32")

X_val = rng.normal(size=(200, 20)).astype("float32")
y_val = rng.normal(size=200).astype("float32")

X_test = rng.normal(size=(200, 20)).astype("float32")
y_test = rng.normal(size=200).astype("float32")

config = ScalingLawConfig(
    param_sizes=["1K", "10K"],
    epochs=5,
    output_dir="./Output/quickstart",
    resume=False,
    save_models=False,
)

experiment = ScalingLawExperiment(config)
results = experiment.run(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
)

print(results)
```

This creates:

```text
./Output/quickstart/
  scaling_results.pkl
  scaling_results.json
```

If portfolio analysis is enabled and dates are supplied, it also creates:

```text
portfolio_returns.csv
```

## Simplest DataFrame Run

For most research workflows, you will have a single DataFrame with feature
columns, a target column, and a date column. The estimator can split it for you
without look-ahead leakage.

```python
import numpy as np
import pandas as pd

from Scaling_Law_Estimator import ScalingLawConfig, ScalingLawExperiment

rng = np.random.default_rng(42)
n = 5000

df = pd.DataFrame({
    "date": pd.date_range("2000-01-01", periods=n, freq="D"),
    "x1": rng.normal(size=n),
    "x2": rng.normal(size=n),
    "x3": rng.normal(size=n),
    "target": rng.normal(size=n),
})

config = ScalingLawConfig(
    param_sizes=["1K", "10K", "100K"],
    epochs=10,
    test_size=0.2,
    val_size=0.125,
    output_dir="./Output/dataframe_run",
    resume=False,
)

experiment = ScalingLawExperiment(config)
results = experiment.run_from_dataframe(
    df=df,
    feature_cols=["x1", "x2", "x3"],
    target_col="target",
    date_col="date",
)
```

With numeric `test_size` and `val_size`, splitting is done by unique dates, not
by individual rows. That preserves the time ordering.

## Date Cutoff Splits

Use string cutoffs when you want exact time periods:

```python
config = ScalingLawConfig(
    test_size="2020-01-01",
    val_size="2018-01-01",
    param_sizes=["10K", "100K"],
    epochs=20,
    output_dir="./Output/date_cutoffs",
)

experiment = ScalingLawExperiment(config)
results = experiment.run_from_dataframe(
    df=df,
    feature_cols=["x1", "x2", "x3"],
    target_col="target",
    date_col="date",
)
```

This means:

```text
train: date < 2018-01-01
val:   2018-01-01 <= date < 2020-01-01
test:  date >= 2020-01-01
```

## Configuration Styles

The package supports two styles.

### Flat Compatibility Style

This matches the older script API and is still fully supported:

```python
config = ScalingLawConfig(
    normalization="layer",
    architecture_mode="fixed_depth",
    fixed_depth_layers=5,
    dropout_rate=0.2,
    epochs=100,
    batch_size=65536,
    learning_rate=0.001,
    test_size="1994-12-01",
    val_size="1992-12-01",
    output_dir="./Output/gfd",
    resume=True,
    random_state=42,
)
```

### Nested Package Style

This is better for package users because each concern has its own config object:

```python
from Scaling_Law_Estimator import (
    ScalingLawConfig,
    ArchitectureConfig,
    TrainingConfig,
    SplitConfig,
    OutputConfig,
    RuntimeConfig,
    ArchitectureMode,
    NormalizationType,
)

config = ScalingLawConfig(
    architecture=ArchitectureConfig(
        normalization=NormalizationType.LAYER,
        architecture_mode=ArchitectureMode.FIXED_DEPTH,
        fixed_depth_layers=5,
        dropout_rate=0.2,
    ),
    training=TrainingConfig(
        epochs=100,
        train_batch_size=65536,
        prediction_batch_size=262144,
        learning_rate=0.001,
    ),
    split=SplitConfig(
        test_size="1994-12-01",
        val_size="1992-12-01",
    ),
    output=OutputConfig(
        output_dir="./Output/gfd",
        save_models=True,
    ),
    runtime=RuntimeConfig(
        resume=True,
        random_state=42,
    ),
)
```

Nested configs may also be passed as dictionaries:

```python
config = ScalingLawConfig(
    architecture={
        "architecture_mode": "fixed_depth",
        "fixed_depth_layers": 3,
        "normalization": "batch",
    },
    training={
        "epochs": 50,
        "train_batch_size": 8192,
    },
    output={
        "output_dir": "./Output/dict_config",
    },
)
```

## Parameter Sizes

Parameter sizes can be strings or generated programmatically.

```python
config = ScalingLawConfig(
    param_sizes=["100", "1K", "10K", "100K", "1M"],
    epochs=50,
)
```

Useful helpers:

```python
from Scaling_Law_Estimator import ScalingLawExperiment

print(ScalingLawExperiment.parse_size("10K"))       # 10000
print(ScalingLawExperiment.parse_size("1.5M"))      # 1500000
print(ScalingLawExperiment.format_params(1000000))  # 1M

sizes = ScalingLawExperiment.make_param_sizes(
    jump=50,
    min_size=100,
    max_size=1_000_000,
)
```

You can also train only part of a size grid:

```python
config = ScalingLawConfig(
    param_sizes=ScalingLawExperiment.make_param_sizes(jump=50),
    start_at_size="10K",
    stop_at_size="1M",
)
```

## Size-Dependent Epochs

`epochs` can be an integer or a callable that receives the target parameter
count.

```python
def get_epochs(size: int) -> int:
    return max(int(0.1 * (size ** 0.75)), 1) + 10

config = ScalingLawConfig(
    param_sizes=["1K", "10K", "100K", "1M"],
    epochs=get_epochs,
)
```

## Architecture Options

The estimator has two built-in architecture modes.

### Tapered

Tapered mode changes depth/width patterns as target parameter counts grow.

```python
config = ScalingLawConfig(
    architecture_mode="tapered",
    param_sizes=["1K", "10K", "100K", "1M"],
)
```

### Fixed Depth

Fixed-depth mode solves for an approximately uniform hidden width at a fixed
number of layers.

```python
config = ScalingLawConfig(
    architecture_mode="fixed_depth",
    fixed_depth_layers=5,
    param_sizes=["10K", "100K", "1M"],
)
```

Normalization and dropout:

```python
config = ScalingLawConfig(
    normalization="layer",        # "layer", "batch", or "none"
    dropout_rate=0.1,
    dropout_middle_only=True,
    initializer="he_normal",      # "he_normal" or "glorot_uniform"
    use_input_normalization=True,
)
```

## Training Options

Common training controls:

```python
config = ScalingLawConfig(
    epochs=500,
    batch_size=8192,              # legacy alias for train_batch_size
    learning_rate=0.001,
    optimizer="adam",
    clip_norm=1.0,
)
```

Package-style training config:

```python
from Scaling_Law_Estimator import TrainingConfig

config = ScalingLawConfig(
    training=TrainingConfig(
        epochs=100,
        train_batch_size=65536,
        validation_batch_size=None,
        prediction_batch_size=262144,
        learning_rate=0.001,
        clip_norm=1.0,
    )
)
```

Learning-rate scheduler:

```python
config = ScalingLawConfig(
    lr_scheduler_enabled=True,
    lr_scheduler_factor=0.5,
    lr_scheduler_patience=None,   # defaults to max(epochs // 5, 50)
    lr_scheduler_min_lr=1e-10,
)
```

## Runtime And Reproducibility

```python
config = ScalingLawConfig(
    random_state=42,
    enable_determinism=True,
    precision=32,
    debug_memory=False,
    show_live_plots=False,
)
```

Seeds are used for NumPy, TensorFlow, initializers, and dropout layer seeds. By
default, per-layer seeds are `random_state + layer_index`, preserving the old
default behavior of `42 + layer_index`.

## Resume Modes

Long scaling experiments are often interrupted. Resume behavior is explicit.

```python
from Scaling_Law_Estimator import ResumeMode

config = ScalingLawConfig(
    output_dir="./Output/resume_demo",
    resume=ResumeMode.UPDATE_EXISTING,
)
```

Available modes:

```text
UPDATE_EXISTING  current old resume=True behavior; retrain and upsert results
OVERWRITE        current old resume=False behavior; reset result artifacts
SKIP_EXISTING    skip any model_name already present in saved results
FAIL_IF_EXISTS   raise if result artifacts already contain data
```

Legacy booleans still work:

```python
ScalingLawConfig(resume=True)   # UPDATE_EXISTING
ScalingLawConfig(resume=False)  # OVERWRITE
```

## Output Artifacts

Default output files:

```text
scaling_results.pkl
scaling_results.json
portfolio_returns.csv
test_sample.csv
Models/
```

Customize artifact names:

```python
from Scaling_Law_Estimator import ArtifactNames, OutputConfig

config = ScalingLawConfig(
    output=OutputConfig(
        output_dir="./Output/custom_names",
        save_pickle=True,
        save_json=True,
        save_csv=True,
        save_models=True,
        artifacts=ArtifactNames(
            results_pickle="results.pkl",
            results_json="results.json",
            portfolio_returns_csv="returns.csv",
            test_sample_csv="test_rows.csv",
            models_dir="keras_models",
        ),
    )
)
```

## Missing Data

Default behavior is `drop_any`, which matches the original script: rows are
dropped if any required feature, target, or date value is missing.

```python
from Scaling_Law_Estimator import MissingDataConfig

config = ScalingLawConfig(
    missing_data=MissingDataConfig(policy="drop_any")
)
```

Other policies:

```python
# Drop rows only when target/date is missing. Feature NaNs remain.
config = ScalingLawConfig(
    missing_data={"policy": "drop_target_only"}
)

# Raise if any required model column has missing values.
config = ScalingLawConfig(
    missing_data={"policy": "error"}
)

# Drop target/date NaNs, then fill feature NaNs with feature means.
config = ScalingLawConfig(
    missing_data={"policy": "impute_mean"}
)
```

## Explicit Mask Splits

Masks must align with the original DataFrame index. They are applied after the
missing-data policy, while preserving the original row alignment.

```python
from Scaling_Law_Estimator import SplitConfig

train_mask = df["date"] < "2018-01-01"
val_mask = (df["date"] >= "2018-01-01") & (df["date"] < "2020-01-01")
test_mask = df["date"] >= "2020-01-01"

config = ScalingLawConfig(
    split=SplitConfig(
        mode="masks",
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )
)
```

## Pre-Split Arrays

If your project already owns splitting, pass arrays directly to `run()`:

```python
experiment.run(X_train, y_train, X_val, y_val, X_test, y_test)
```

Or store them in `PreSplitData` for a DataSplitter-style workflow:

```python
from Scaling_Law_Estimator import PreSplitData, SplitConfig

config = ScalingLawConfig(
    split=SplitConfig(
        mode="pre_split",
        pre_split=PreSplitData(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            test_dates=test_dates,
        ),
    )
)
```

## R2 Benchmarks

R2 is calculated relative to a benchmark prediction series. Built-in modes:

```text
historical_mean
historical_mean_updating
ar1
ar1_updating
```

Example:

```python
config = ScalingLawConfig(
    benchmark={"mode": "historical_mean_updating"}
)

results = experiment.run_from_dataframe(
    df=df,
    feature_cols=["x1", "x2"],
    target_col="target",
    date_col="date",
    benchmark="ar1_updating",  # optional runtime override
)
```

You can also supply a callable benchmark:

```python
import numpy as np

def my_benchmark(y_train, y_val, y_test):
    val_pred = np.full(len(y_val), np.mean(y_train))
    test_pred = np.full(len(y_test), np.mean(np.concatenate([y_train, y_val])))
    return val_pred, test_pred

config = ScalingLawConfig(
    benchmark=my_benchmark
)
```

Callable benchmarks may return either:

```python
(val_predictions, test_predictions)
```

or:

```python
{
    "val_predictions": val_predictions,
    "test_predictions": test_predictions,
}
```

## Compute Accounting

By default, FLOPs per epoch are estimated as:

```text
6 * actual_params * n_train
```

You can replace this with a custom callable:

```python
from Scaling_Law_Estimator import ComputeConfig

def custom_flops(actual_params, train_samples, input_dim, architecture, model):
    return 8 * actual_params * train_samples

config = ScalingLawConfig(
    compute=ComputeConfig(
        flop_estimator=custom_flops,
    )
)
```

Saved result keys remain:

```text
flops_per_epoch
total_flops
pf_days
training_curve.cumulative_pf_days
```

## Panel Portfolio Analysis

Panel mode ranks cross-sectional predictions within each date and builds
long-short portfolios.

```python
results = experiment.run_from_dataframe(
    df=df,
    feature_cols=["x1", "x2", "x3"],
    target_col="next_return",
    date_col="date",
    portfolio="panel",
)
```

Panel outputs include:

```text
D1 ... D10
Forecast_Weighted
LS_50
LS_30
LS_10
```

Annualization defaults to monthly data:

```python
config = ScalingLawConfig(
    annualization_periods=12
)
```

For daily data, choose something like:

```python
config = ScalingLawConfig(
    annualization_periods=252
)
```

### Optional Panel Transaction Costs

Panel transaction-cost accounting requires asset identifiers.

```python
config = ScalingLawConfig(
    transaction_cost_rate=0.001,
    portfolio={"asset_id_col": "permno"},
)

results = experiment.run_from_dataframe(
    df=df,
    feature_cols=["x1", "x2"],
    target_col="next_return",
    date_col="date",
    portfolio="panel",
    asset_id_col="permno",
)
```

If no `asset_id_col` is supplied, panel behavior remains unchanged.

## Time-Series Portfolio Analysis

Time-series mode treats the predictions as forecasts for one asset or one
aggregate return series. It creates a forecast-timed strategy.

```python
results = experiment.run_from_dataframe(
    df=df,
    feature_cols=["x1", "x2"],
    target_col="market_return",
    date_col="date",
    portfolio="ts",
)
```

Configure the strategy:

```python
from Scaling_Law_Estimator import TSStrategyConfig, TradingConfig

config = ScalingLawConfig(
    ts_strategy=TSStrategyConfig(
        kappa=2.0,
        min_periods=12,
        winsorize_weights=True,
        weight_floor=-1.0,
        weight_cap=3.0,
        signal_lag=1,
        standardize_signal=False,
    ),
    trading=TradingConfig(
        transaction_cost_rate=0.0005,
        leverage_cap=2.0,
        long_only=False,
        allow_short=True,
    ),
)
```

Legacy runtime override still works:

```python
results = experiment.run_from_dataframe(
    df=df,
    feature_cols=["x1", "x2"],
    target_col="market_return",
    date_col="date",
    portfolio="ts",
    kappa=1.5,
)
```

TS output includes:

```text
actual_return
prediction
hist_mean
hist_std
z_score
weight
turnover
transaction_cost
gross_strategy_return
strategy_return
```

## Plotting

After an experiment finishes, create the standard plot bundle:

```python
experiment.create_plots(dpi=300)
```

Or use the plotter directly:

```python
from Scaling_Law_Estimator import ScalingLawPlotter

plotter = ScalingLawPlotter("./Output/dataframe_run")

plotter.plot_scaling_curves(
    loss_type="val_loss",
    x_axis="compute",
)

plotter.plot_final_performance(
    metric="test_loss",
    x_axis="compute",
)

plotter.plot_sharpe_ratio_scaling(
    breakpoint="50",
    x_axis="compute",
)
```

If you customized artifact names, pass the same artifact config:

```python
plotter = ScalingLawPlotter(
    "./Output/custom_names",
    artifacts=config.output.artifacts,
)
```

## A Complete Minimal Research Script

```python
import numpy as np
import pandas as pd

from Scaling_Law_Estimator import (
    ScalingLawConfig,
    ScalingLawExperiment,
    ArchitectureMode,
    NormalizationType,
    ResumeMode,
)

rng = np.random.default_rng(42)
n_dates = 120
n_assets = 100

dates = np.repeat(pd.date_range("2010-01-31", periods=n_dates, freq="ME"), n_assets)
permno = np.tile(np.arange(n_assets), n_dates)

df = pd.DataFrame({
    "date": dates,
    "permno": permno,
    "value": rng.normal(size=n_dates * n_assets),
    "momentum": rng.normal(size=n_dates * n_assets),
    "quality": rng.normal(size=n_dates * n_assets),
})

df["next_return"] = (
    0.01 * df["value"]
    + 0.02 * df["momentum"]
    - 0.01 * df["quality"]
    + rng.normal(scale=0.05, size=len(df))
)

config = ScalingLawConfig(
    normalization=NormalizationType.LAYER,
    architecture_mode=ArchitectureMode.FIXED_DEPTH,
    fixed_depth_layers=3,
    dropout_rate=0.1,
    param_sizes=["1K", "10K", "100K"],
    epochs=20,
    batch_size=8192,
    learning_rate=0.001,
    test_size="2018-01-31",
    val_size="2016-01-31",
    output_dir="./Output/minimal_research",
    resume=ResumeMode.OVERWRITE,
    random_state=42,
    annualization_periods=12,
)

experiment = ScalingLawExperiment(config)
results = experiment.run_from_dataframe(
    df=df,
    feature_cols=["value", "momentum", "quality"],
    target_col="next_return",
    date_col="date",
    portfolio="panel",
    asset_id_col="permno",
)

experiment.create_plots(dpi=300)
```

## Result Dictionary

Each trained model stores a result dictionary with keys such as:

```text
model_name
target_params
actual_params
architecture
train_loss
val_loss
test_loss
val_r2
test_r2
train_time
total_flops
pf_days
epochs
batch_size
learning_rate
flops_per_epoch
normalization
architecture_mode
portfolio_mode
benchmark
annualization_periods
portfolio_stats
training_curve
```

`training_curve` contains:

```text
epochs
train_loss
val_loss
cumulative_flops
cumulative_pf_days
```

## Common Recipes

### Train One Model Size Only

```python
config = ScalingLawConfig(
    param_sizes=["100K"],
    epochs=50,
)
```

### Train A Range But Start In The Middle

```python
config = ScalingLawConfig(
    param_sizes=ScalingLawExperiment.make_param_sizes(jump=50),
    start_at_size="100K",
    stop_at_size="10M",
)
```

### Skip Models Already Completed

```python
config = ScalingLawConfig(
    output_dir="./Output/long_run",
    resume="skip_existing",
)
```

### Fail Rather Than Accidentally Reuse An Output Directory

```python
config = ScalingLawConfig(
    output_dir="./Output/fresh_run",
    resume="fail_if_exists",
)
```

### Use No Dropout

```python
config = ScalingLawConfig(
    dropout_rate=0.0,
)
```

### Disable Input Normalization

```python
config = ScalingLawConfig(
    use_input_normalization=False,
)
```

### Use Batch Normalization

```python
config = ScalingLawConfig(
    normalization="batch",
)
```

### Use No Hidden-Layer Normalization

```python
config = ScalingLawConfig(
    normalization="none",
)
```

### Save Keras Models

```python
config = ScalingLawConfig(
    save_models=True,
    output_dir="./Output/save_models_demo",
)
```

Models are saved under:

```text
./Output/save_models_demo/Models/model_<size>.keras
```

### Use Mixed Precision

```python
config = ScalingLawConfig(
    precision=16,
)
```

Supported precision values are `8`, `16`, `32`, and `64`. The default is `32`.
Use reduced precision only when your hardware and numerical setup are appropriate.

## Troubleshooting

### ImportError: No module named tensorflow

Install TensorFlow in the environment where you run the experiment:

```bash
pip install tensorflow
```

### No GPU Found

The code can run on CPU, but scaling-law sweeps are usually expensive. The
experiment prints detected GPU devices at startup.

### Empty Split Error

If you see an empty train/validation/test split error, check:

```python
config.test_size
config.val_size
config.split.mode
df["date"].min(), df["date"].max()
```

For date cutoffs, `val_size` must be before `test_size`.

### Portfolio Returns Are Missing

Portfolio returns are only saved when `test_dates` are available. If you call
`run(...)` directly with arrays, pass `test_dates`:

```python
experiment.run(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    test_dates=test_dates,
    portfolio="panel",
)
```

`run_from_dataframe(...)` handles this automatically.

### Existing Results Are Being Updated

That is the default old behavior. Use:

```python
resume="skip_existing"
```

or:

```python
resume=False
```

depending on whether you want to skip completed models or overwrite artifacts.

## Design Principle

The defaults are intentionally conservative: they preserve the original research
script's methods and behavior. New configuration options are opt-in. This makes
the module suitable for package use without silently changing existing
experiments.
