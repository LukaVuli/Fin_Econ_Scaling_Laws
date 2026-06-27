# Financial Economics Scaling Laws

Financial Economics Scaling Laws is a Python package for estimating scaling laws in
economics and finance forecasting problems.

The package lets researchers train neural networks across many model sizes, record
forecast accuracy and compute, and fit empirical power-law relationships between
compute/model scale and predictive performance. It was built from the methods in
Timmermann and Vulicevic (2026). It is a methods
package meant to help other researchers apply, modify, and extend scaling-law tools
on their own time-series, panel, and return-predictability data.

The PyPI/project name is `Fin_Econ_Scaling_Laws`. The Python import module is
`scaling_laws`:

```python
from scaling_laws import ScalingLawConfig, ScalingLawExperiment
```

## Overview

Scaling laws describe how forecast performance changes as computational scale
increases. In this package, compute is tracked during model training and related to
out-of-sample performance using power-law curves. The same workflow can be used to
study:

1. How forecast loss changes with model scale and training compute.
2. Whether performance appears to approach an asymptotic limit.
3. How economic measures such as portfolio Sharpe ratios scale with compute.
4. Whether richer predictor sets improve the attainable performance frontier.
5. How much compute is required before larger models or larger datasets become useful.

The package is designed for researchers who want to estimate these relationships on
their own economics or finance data, not only on the data used in the paper.

## Authors

- [Allan Timmermann](https://rady.ucsd.edu/faculty-research/faculty/allan-timmermann.html)
- [Luka Vulicevic](https://www.lukavulicevic.com/)

## Academic Paper

The methods are based on:

**Compute, Complexity, and the Scaling Laws of Return Predictability**

Allan Timmermann and Luka Vulicevic
SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6105327

If you use this package or the methodology in academic work, please cite:

```bibtex
@article{timmermann2026compute,
  title={Compute, Complexity, and the Scaling Laws of Return Predictability},
  author={Timmermann, Allan and Vulicevic, Luka},
  journal={Available at SSRN 6105327},
  year={2026}
}
```

## Key Features

### Estimate scaling laws from forecasting experiments

Train the same model family across a grid of target parameter counts, save
out-of-sample performance, and fit scaling-law plots against cumulative training
compute.


### Flexible model-size grids

Parameter grids can be supplied directly, for example `["250", "1K", "10K",
"100K", "1M"]`, or generated programmatically with
`ScalingLawExperiment.make_param_sizes(...)`.

### Portfolio evaluation

When test dates are supplied, predictions can be evaluated as:

1. **Panel portfolios**: cross-sectional sorts by predicted returns, including
   decile returns, forecast-weighted returns, and long-short portfolios.
2. **Time-series strategies**: forecast-timed strategy returns using configurable
   risk scaling, lags, weights, and trading constraints.

### Plotting tools

After a run, create standard scaling-law figures for losses, R2, and portfolio
performance with `experiment.create_plots(...)`.

### Simulation example

The repository includes a simulation script that creates synthetic firm-month
characteristics and next-month returns, then runs the package on the simulated
panel. This is meant to show how to work with the methodology before applying it
to real data.

## Project Structure

```text
Fin_Econ_Scaling_Laws/
|-- README.md
|-- LICENSE
|-- requirements.txt
|-- pyproject.toml
|-- llms.txt
|-- Testing/
|   `-- simulation_example.py
`-- scaling_laws/
    |-- __init__.py
    |-- config.py
    |-- experiment.py
    |-- model_builder.py
    |-- data_splitter.py
    |-- portfolio.py
    |-- plotting.py
    |-- results.py
    |-- callbacks.py
    |-- enums.py
    `-- utils/
        |-- format.py
        |-- memory.py
        `-- system.py
```

## Installation & Requirements

### Platform support

This package is currently supported for macOS / Apple Silicon workflows using
TensorFlow with `tensorflow-metal`. NVIDIA/CUDA GPU systems may run the code, but
they are not a supported target for now because we have observed training
instability and backend-dependent behavior on those systems.

### From PyPI

Install the package with:

```bash
pip install Fin_Econ_Scaling_Laws
```

### From this repository

For local development:

```bash
pip install -e .
```

To install the full runtime stack listed in this repository:

```bash
pip install -r requirements.txt
```

### Core dependencies

```text
tensorflow
numpy
pandas
scipy
matplotlib
```

The requirements file includes `tensorflow-metal` for Apple Silicon Macs through
a platform marker. Optional memory diagnostics use `psutil`.

## Package Imports

Most users start with:

```python
from scaling_laws import (
    ScalingLawConfig,
    ScalingLawExperiment,
    TrainingConfig,
    SplitConfig,
    OutputConfig,
    RuntimeConfig,
    ResumeMode,
)
```

Additional configuration classes and enums are also exported from `scaling_laws`,
including:

```python
from scaling_laws import (
    ArchitectureConfig,
    ArchitectureMode,
    AnnualizationConfig,
    BenchmarkConfig,
    ComputeConfig,
    FuzzyStopConfig,
    InitializerType,
    MissingDataConfig,
    MissingDataPolicy,
    NormalizationType,
    PortfolioConfig,
    PortfolioMode,
    PreSplitData,
    SchedulerConfig,
    TradingConfig,
    TSStrategyConfig,
    format_params,
    parse_size,
    print_system_info,
)
```

For direct plotting from saved results:

```python
from scaling_laws.plotting import ScalingLawPlotter
```

## Data Requirements

The standard DataFrame workflow expects:

- A date column, such as `date`
- One or more feature columns observed at time `t`
- A target column observed at the forecast horizon, such as `ret_exc`
- Optionally, an asset identifier column, such as `permno`, for panel portfolio
  accounting

For example:

```text
date        permno   char_01   char_02   ...   ret_exc
2000-01-31  10001   0.421    -0.038          0.014
2000-01-31  10002  -0.107     0.552         -0.021
2000-02-29  10001   0.390     0.012          0.008
```

The package does not require stock returns specifically. The same structure can
be used for other economics or finance forecasting problems where predictors are
observed before the target.

## Usage

### 1. Minimal DataFrame Workflow

Use `ScalingLawExperiment.run(...)` when your data are in a single DataFrame.

```python
import numpy as np
import pandas as pd

from scaling_laws import (
    OutputConfig,
    ResumeMode,
    RuntimeConfig,
    ScalingLawConfig,
    ScalingLawExperiment,
    SplitConfig,
    TrainingConfig,
)

rng = np.random.default_rng(42)
n = 2_000

df = pd.DataFrame({
    "date": pd.date_range("2000-01-31", periods=n, freq="D"),
    "x1": rng.normal(size=n),
    "x2": rng.normal(size=n),
    "x3": rng.normal(size=n),
    "ret_exc": rng.normal(scale=0.05, size=n),
})

config = ScalingLawConfig(
    training=TrainingConfig(epochs=5, train_batch_size=512),
    split=SplitConfig(test_size=0.20, val_size=0.125),
    output=OutputConfig(output_dir="./Output/readme_dataframe"),
    runtime=RuntimeConfig(resume=ResumeMode.OVERWRITE, random_state=42),
    param_sizes=["1K", "10K"],
)

experiment = ScalingLawExperiment(config)
results = experiment.run(
    df,
    X=["x1", "x2", "x3"],
    y="ret_exc",
    date_col="date",
)
```

With numeric `test_size` and `val_size`, splitting is done by unique dates, not
by individual rows.

### 2. Exact Date Cutoffs

Use string cutoffs when you want fixed sample periods.

```python
config = ScalingLawConfig(
    split=SplitConfig(
        val_size="2018-01-01",
        test_size="2020-01-01",
    ),
    output=OutputConfig(output_dir="./Output/date_cutoffs"),
    param_sizes=["10K", "100K"],
)
```

This creates:

```text
train: date < 2018-01-01
val:   2018-01-01 <= date < 2020-01-01
test:  date >= 2020-01-01
```

### 3. Pre-Split Arrays

Use `run_from_arrays(...)` when your own pipeline already created the train,
validation, and test arrays.

```python
import numpy as np

from scaling_laws import (
    OutputConfig,
    ResumeMode,
    RuntimeConfig,
    ScalingLawConfig,
    ScalingLawExperiment,
    TrainingConfig,
)

rng = np.random.default_rng(42)

X_train = rng.normal(size=(1_000, 20)).astype("float32")
y_train = rng.normal(size=1_000).astype("float32")
X_val = rng.normal(size=(250, 20)).astype("float32")
y_val = rng.normal(size=250).astype("float32")
X_test = rng.normal(size=(250, 20)).astype("float32")
y_test = rng.normal(size=250).astype("float32")

config = ScalingLawConfig(
    training=TrainingConfig(epochs=5, train_batch_size=512),
    output=OutputConfig(output_dir="./Output/readme_arrays"),
    runtime=RuntimeConfig(resume=ResumeMode.OVERWRITE),
    param_sizes=["1K", "10K"],
)

experiment = ScalingLawExperiment(config)
results = experiment.run_from_arrays(
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    X_test=X_test,
    y_test=y_test,
)
```

For portfolio analysis from arrays, also pass `test_dates`. For panel portfolio
analysis, pass `asset_ids`.

### 4. Panel Portfolio Evaluation

Panel mode sorts cross-sectional predictions within each date.

```python
from scaling_laws import AnnualizationConfig, PortfolioConfig, ScalingLawConfig

config = ScalingLawConfig(
    portfolio=PortfolioConfig(mode="panel", asset_id_col="permno"),
    annualization=AnnualizationConfig(periods=12),
)

results = experiment.run(
    df,
    X=feature_cols,
    y="ret_exc",
    date_col="date",
    portfolio="panel",
    asset_id_col="permno",
)
```

### 5. Time-Series Strategy Evaluation

Time-series mode treats model predictions as a strategy signal for one return
series or aggregate portfolio.

```python
from scaling_laws import PortfolioConfig, ScalingLawConfig, TSStrategyConfig

config = ScalingLawConfig(
    portfolio=PortfolioConfig(mode="ts"),
    ts_strategy=TSStrategyConfig(
        kappa=1.0,
        min_periods=6,
        signal_lag=1,
        standardize_signal=False,
    ),
)

results = experiment.run(
    df,
    X=feature_cols,
    y="ret_exc",
    date_col="date",
    portfolio="ts",
    kappa=0.5,
)
```

### 6. Plot Results

After an experiment finishes:

```python
experiment.create_plots(dpi=300)
```

To include panel long-short breakpoints:

```python
experiment.create_plots(dpi=300, include_ls_breakpoints=True)
```

To plot directly from a saved output directory:

```python
from scaling_laws.plotting import ScalingLawPlotter

plotter = ScalingLawPlotter("./Output/readme_dataframe")
plotter.create_all_plots(dpi=300)
```

## Configuration Parameters

`ScalingLawConfig` is the top-level configuration object. It nests smaller
configuration objects so researchers can change one part of the workflow without
rewriting the others.

### Common controls

- **`param_sizes`**: model-size grid, e.g. `["1K", "10K", "100K", "1M"]`
- **`start_at_size` / `stop_at_size`**: run only part of the size grid
- **`TrainingConfig.epochs`**: fixed integer or callable schedule by parameter count
- **`TrainingConfig.train_batch_size`**: training batch size
- **`TrainingConfig.shuffle`**: whether Keras shuffles training rows each epoch
- **`TrainingConfig.learning_rate`**: optimizer learning rate
- **`SplitConfig.test_size` / `val_size`**: date proportions or date cutoffs
- **`OutputConfig.output_dir`**: directory for saved outputs
- **`RuntimeConfig.resume`**: behavior when outputs already exist
- **`ComputeConfig.precision`**: numeric precision policy
- **`ComputeConfig.allow_tf32`**: whether NVIDIA GPUs may use TF32 matrix math
- **`PortfolioConfig.mode`**: `"panel"` or `"ts"`

### Example full configuration

```python
from scaling_laws import (
    ArchitectureConfig,
    ArchitectureMode,
    AnnualizationConfig,
    BenchmarkConfig,
    ComputeConfig,
    FuzzyStopConfig,
    InitializerType,
    NormalizationType,
    OutputConfig,
    PortfolioConfig,
    ResumeMode,
    RuntimeConfig,
    ScalingLawConfig,
    SchedulerConfig,
    SplitConfig,
    TrainingConfig,
    TSStrategyConfig,
)

config = ScalingLawConfig(
    architecture=ArchitectureConfig(
        normalization=NormalizationType.LAYER,
        architecture_mode=ArchitectureMode.FIXED_DEPTH,
        fixed_depth_layers=5,
        dropout_rate=0.10,
        dropout_middle_only=True,
        initializer=InitializerType.HE_NORMAL,
        use_input_normalization=True,
    ),
    training=TrainingConfig(
        epochs=100,
        train_batch_size=65_536,
        validation_batch_size=None,
        prediction_batch_size=262_144,
        shuffle=False,
        learning_rate=0.001,
        optimizer="adam",
        clip_norm=1.0,
    ),
    scheduler=SchedulerConfig(
        lr_scheduler_enabled=True,
        lr_scheduler_factor=0.5,
        lr_scheduler_patience=None,
        lr_scheduler_min_lr=1e-10,
    ),
    fuzzy_stop=FuzzyStopConfig(enabled=False),
    split=SplitConfig(
        val_size="2018-01-01",
        test_size="2020-01-01",
    ),
    output=OutputConfig(
        output_dir="./Output/main_run",
        save_pickle=True,
        save_json=True,
        save_csv=True,
        save_models=False,
    ),
    runtime=RuntimeConfig(
        resume=ResumeMode.UPDATE_EXISTING,
        random_state=42,
        show_live_plots=False,
        debug_memory=False,
    ),
    compute=ComputeConfig(
        precision=32,
        enable_determinism=True,
        allow_tf32=False,
    ),
    benchmark=BenchmarkConfig(mode="historical_mean"),
    annualization=AnnualizationConfig(periods=12),
    ts_strategy=TSStrategyConfig(kappa=1.0),
    portfolio=PortfolioConfig(mode="panel", asset_id_col="permno"),
    param_sizes=["1K", "10K", "100K", "1M"],
)
```

Nested configuration fields can also be passed as dictionaries:

```python
config = ScalingLawConfig(
    training={
        "epochs": 50,
        "train_batch_size": 8_192,
    },
    output={
        "output_dir": "./Output/dict_config",
    },
    runtime={
        "resume": "overwrite",
    },
    param_sizes=["1K", "10K"],
)
```

## Parameter Sizes

Parameter sizes can be strings or integers:

```python
from scaling_laws import ScalingLawExperiment, format_params, parse_size

print(parse_size("10K"))          # 10000
print(parse_size("1.5M"))         # 1500000
print(format_params(1_000_000))   # 1M

sizes = ScalingLawExperiment.make_param_sizes(
    min_size=1_000,
    max_size=1_000_000,
    num=8,
)
```

Size-dependent epoch schedules are supported:

```python
def epochs_for_size(size: int) -> int:
    return max(int(0.1 * size ** 0.75), 1) + 10

config = ScalingLawConfig(
    training=TrainingConfig(epochs=epochs_for_size),
    param_sizes=["1K", "10K", "100K"],
)
```

## Resume Modes

Long scaling-law experiments can be interrupted. Resume behavior is controlled
with `RuntimeConfig.resume`.

```text
update_existing  update or add model results as training proceeds
overwrite        reset result artifacts before starting
skip_existing    skip model_name entries already present in saved results
fail_if_exists   raise if existing result artifacts contain data
```

Example:

```python
from scaling_laws import ResumeMode, RuntimeConfig, ScalingLawConfig

config = ScalingLawConfig(
    runtime=RuntimeConfig(resume=ResumeMode.SKIP_EXISTING),
    output={"output_dir": "./Output/resume_demo"},
    param_sizes=["1K", "10K", "100K"],
)
```

## Output Files

By default, outputs are written under `OutputConfig.output_dir`.

```text
scaling_results.pkl       Pickled list of per-model result dictionaries
scaling_results.json      JSON-safe copy of the saved model results
portfolio_returns.csv     Panel or time-series portfolio returns, if available
test_sample.csv           Test-sample rows and metadata for DataFrame runs
Models/                   Saved Keras models, if save_models=True
```

Training-history plots may also be saved under `Models/` as:

```text
Models/<model_name>_training.png
```

Plotting can create files such as:

```text
scaling_curves_val_compute.png
test_loss_vs_compute.png
val_r2_vs_compute.png
test_r2_vs_compute.png
sharpe_ratio_Forecast_Weighted_vs_compute.png
sharpe_ratio_LS50_vs_compute.png
sharpe_ratio_LS30_vs_compute.png
sharpe_ratio_LS10_vs_compute.png
```

## Simulation Example

The repository includes:

```text
Testing/simulation_example.py
```

This script simulates a characteristic-driven firm-month panel. Characteristics
are observed at month `t`, and the target `ret_exc` is the excess return earned
in month `t + 1`.

The hidden expected-return function combines:

1. Linear characteristic levels
2. Nonlinear single-characteristic transforms
3. Pairwise characteristic interactions
4. Nonlinear pairwise interactions
5. Common month-level shocks
6. Idiosyncratic firm-level shocks

The default output directory is:

```text
~/Desktop/characteristic next return simulation
```

The script saves a simulated panel CSV, runs the scaling-law experiment,
performs panel portfolio analysis, saves the standard result files, and then
attempts to create plots.

Edit these top-level constants first:

- **`N_MONTHS`**: number of simulated signal months
- **`N_FIRMS`**: number of firms per month
- **`N_FIRM_CHARS`**: number of characteristics
- **Signal strengths**: linear, nonlinear, pairwise, and nonlinear interaction
  strengths
- **Noise scales**: common and idiosyncratic shock scales
- **`OUTPUT_DIR`**: where files are written
- **Fuzzy-stop controls**: optional training extension behavior
- **`param_sizes`** inside `build_config(...)`: model-size grid

Run the example from the repository root:

```bash
python Testing/simulation_example.py
```

This script is intentionally more substantial than a smoke test. Reduce sample
sizes and model sizes before running it on a laptop.

## Research Applications

Financial Economics Scaling Laws can be used for:

1. Estimating compute-performance scaling laws in return prediction.
2. Comparing forecasting problems with different predictor sets.
3. Studying whether additional data raises the asymptotic performance frontier.
4. Measuring how quickly forecast performance improves with model scale.
5. Comparing statistical forecast metrics with economic portfolio metrics.
6. Stress-testing whether larger models are worth the additional compute.
7. Adapting the scaling-law methodology to macro, asset pricing, and other
   economics time-series settings.

## Performance Considerations

- Scaling-law experiments can be expensive because the same workflow is repeated
  across many model sizes.
- Start with a small parameter grid such as `["250", "1K", "5K"]`.
- Use short epoch schedules for smoke tests and larger schedules for research
  runs.
- CPU execution is possible but can be slow for large neural networks.
- Apple Silicon with TensorFlow and `tensorflow-metal` is the primary supported
  runtime for now.
- Use `ResumeMode.SKIP_EXISTING` or `ResumeMode.UPDATE_EXISTING` for long runs.
- Saving models can consume substantial disk space; leave `save_models=False`
  unless you need the trained Keras models.

## Contributing and Modifying

This package is intended to be modified and extended. Researchers are free to
change architectures, loss functions, data splits, portfolio construction,
compute accounting, plotting, or any other part of the workflow for their own
research questions.

The defaults are meant to provide a clear starting point based on the paper's
methods, not to constrain how the package can be used.

## License

See `LICENSE` for project licensing terms.
