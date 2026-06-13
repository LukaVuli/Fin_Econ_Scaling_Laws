"""
Characteristic-driven next-month return scaling-law simulation.

This script mirrors the empirical timing in the paper more directly than the
factor-beta simulation:

- firm-month characteristics are observed at month t
- the target is the excess return earned in month t + 1
- expected returns are a hidden nonlinear function of the observed
  characteristics
- the last 32 characteristics are irrelevant Gaussian-style characteristics,
  giving a simple GFD-Synthetic-64 analogue

Outputs are written to:

    ~/Desktop/characteristic next return simulation
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Scaling_Law_Estimator import (  # noqa: E402
    ArchitectureMode,
    InitializerType,
    NormalizationType,
    ResumeMode,
    ScalingLawConfig,
    ScalingLawExperiment,
    print_system_info,
)


N_MONTHS = 200
N_FIRMS = 200
N_SIGNAL_CHARS = 32
N_NOISE_CHARS = 32
N_FIRM_CHARS = N_SIGNAL_CHARS + N_NOISE_CHARS
CHAR_PERSISTENCE = 0.92
FEATURE_SET = "synthetic_64"  # use "signal_32" to train only on informative chars
RANDOM_STATE = 42

OUTPUT_DIR = Path.home() / "Desktop" / "characteristic next return simulation"


def simulate_characteristic_next_return_panel(
        n_months: int = N_MONTHS,
        n_firms: int = N_FIRMS,
        n_signal_chars: int = N_SIGNAL_CHARS,
        n_noise_chars: int = N_NOISE_CHARS,
        char_persistence: float = CHAR_PERSISTENCE,
        feature_set: str = FEATURE_SET,
        random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Simulate firm-month characteristics and next-month returns.

    Rows are dated by the signal month t. The target column, ret_exc, is the
    return earned in month t + 1, generated from characteristics known at t.

    Returns:
        A tuple of (panel_dataframe, feature_columns).
    """
    rng = np.random.default_rng(random_state)
    n_firm_chars = n_signal_chars + n_noise_chars

    dates = pd.date_range("2000-01-31", periods=n_months + 1, freq="ME")
    signal_dates = dates[:-1]
    return_dates = dates[1:]

    signal_characteristics = np.empty(
        (n_months, n_firms, n_signal_chars),
        dtype=np.float64,
    )
    signal_characteristics[0] = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(n_firms, n_signal_chars),
    )
    innovation_scale = np.sqrt(1.0 - char_persistence ** 2)
    for month in range(1, n_months):
        signal_characteristics[month] = (
            char_persistence * signal_characteristics[month - 1]
            + innovation_scale * rng.normal(size=(n_firms, n_signal_chars))
        )

    noise_characteristics = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(n_months, n_firms, n_noise_chars),
    )
    characteristics = np.concatenate(
        [signal_characteristics, noise_characteristics],
        axis=2,
    )
    signal_chars = signal_characteristics

    linear_weights = rng.normal(size=n_signal_chars)
    linear_weights /= np.linalg.norm(linear_weights)

    quad_count = min(8, n_signal_chars)
    quadratic_weights = rng.normal(size=quad_count)
    quadratic_weights /= np.linalg.norm(quadratic_weights)

    interaction_pairs = [
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7),
        (8, 9),
        (10, 11),
        (12, 13),
        (14, 15),
    ]
    interaction_pairs = [
        (i, j) for i, j in interaction_pairs
        if i < n_signal_chars and j < n_signal_chars
    ]
    interaction_weights = rng.normal(size=len(interaction_pairs))
    interaction_weights /= np.linalg.norm(interaction_weights)

    linear_component = 0.012 * np.tensordot(
        signal_chars,
        linear_weights,
        axes=([2], [0]),
    )
    quadratic_component = 0.006 * np.tensordot(
        signal_chars[:, :, :quad_count] ** 2 - 1.0,
        quadratic_weights,
        axes=([2], [0]),
    )
    interaction_terms = np.stack([
        signal_chars[:, :, i] * signal_chars[:, :, j]
        for i, j in interaction_pairs
    ], axis=2)
    interaction_component = 0.008 * np.tensordot(
        interaction_terms,
        interaction_weights,
        axes=([2], [0]),
    )
    smooth_component = (
        0.004 * np.sin(signal_chars[:, :, 0])
        - 0.003 * np.tanh(signal_chars[:, :, 1] * signal_chars[:, :, 2])
        + 0.003 * np.cos(signal_chars[:, :, 3])
    )

    conditional_mean = (
        linear_component
        + quadratic_component
        + interaction_component
        + smooth_component
    )

    next_month_common_shock = rng.normal(loc=0.0, scale=0.010, size=n_months)
    idiosyncratic_error = rng.normal(
        loc=0.0,
        scale=0.045,
        size=(n_months, n_firms),
    )
    excess_return = (
        conditional_mean
        + next_month_common_shock[:, None]
        + idiosyncratic_error
    )

    month_idx = np.repeat(np.arange(n_months), n_firms)
    firm_idx = np.tile(np.arange(n_firms), n_months)
    panel_chars = characteristics[month_idx, firm_idx]

    log_market_equity = (
        10.0
        + panel_chars[:, 0]
        + 0.001 * month_idx
        + rng.normal(loc=0.0, scale=0.10, size=n_months * n_firms)
    )

    data = {
        "date": signal_dates[month_idx],
        "return_date": return_dates[month_idx],
        "permno": firm_idx.astype(int),
        "ret_exc": excess_return.reshape(-1).astype(np.float32),
        "log_market_equity": log_market_equity.astype(np.float32),
    }

    char_cols = []
    for k in range(n_firm_chars):
        col = f"char_{k + 1:02d}"
        data[col] = panel_chars[:, k].astype(np.float32)
        char_cols.append(col)

    df = pd.DataFrame(data)

    if feature_set == "signal_32":
        feature_cols = char_cols[:n_signal_chars]
    elif feature_set == "synthetic_64":
        feature_cols = char_cols
    else:
        raise ValueError(
            "feature_set must be either 'signal_32' or 'synthetic_64', "
            f"got {feature_set!r}"
        )

    return df, feature_cols


def get_epochs(size: int) -> int:
    return max(int((0.1 * (size ** 0.75))), 1) + 100


def resolve_split_cutoffs(
        dates: pd.Series,
        val_months: int = 36,
        test_months: int = 60,
) -> Tuple[str, str]:
    """Choose date cutoffs that fit inside the simulated sample."""
    unique_dates = sorted(pd.to_datetime(pd.Series(dates)).unique())
    n_dates = len(unique_dates)
    total_holdout_months = val_months + test_months

    if n_dates <= total_holdout_months:
        test_months = max(int(round(n_dates * 0.20)), 1)
        remaining_dates = max(n_dates - test_months, 2)
        val_months = max(int(round(n_dates * 0.12)), 1)
        val_months = min(val_months, remaining_dates - 1)

    val_cutoff_idx = n_dates - (val_months + test_months)
    test_cutoff_idx = n_dates - test_months

    if val_cutoff_idx <= 0 or test_cutoff_idx <= val_cutoff_idx:
        raise ValueError(
            "Could not construct non-empty train/validation/test date cutoffs "
            f"from {n_dates} unique dates"
        )

    val_cutoff = pd.Timestamp(unique_dates[val_cutoff_idx]).strftime("%Y-%m-%d")
    test_cutoff = pd.Timestamp(unique_dates[test_cutoff_idx]).strftime("%Y-%m-%d")
    return val_cutoff, test_cutoff


def build_config(val_cutoff: str, test_cutoff: str) -> ScalingLawConfig:
    """Create the same size-aware scaling-law run configuration as the example."""
    return ScalingLawConfig(
        normalization=NormalizationType.LAYER,
        architecture_mode=ArchitectureMode.FIXED_DEPTH,
        fixed_depth_layers=5,
        dropout_rate=0.1,
        dropout_middle_only=True,
        initializer=InitializerType.HE_NORMAL,
        use_input_normalization=True,
        param_sizes=[
            "1K",
            "5K",
            "10K",
            "20K",
            # "50K",
            # "100K",
            # "500K",
            # "1M"
        ],
        epochs=get_epochs,
        batch_size=65536,
        prediction_batch_size=65536,
        learning_rate=0.1,
        clip_norm=1.0,
        lr_scheduler_enabled=True,
        lr_scheduler_factor=0.5,
        lr_scheduler_patience=None,
        lr_scheduler_min_lr=1e-10,
        test_size=test_cutoff,
        val_size=val_cutoff,
        output_dir=str(OUTPUT_DIR),
        save_pickle=True,
        save_json=True,
        save_csv=True,
        save_models=False,
        resume=ResumeMode.OVERWRITE,
        random_state=RANDOM_STATE,
        precision=16,
        enable_determinism=True,
        show_live_plots=True,
        debug_memory=False,
        annualization_periods=12,
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print_system_info()
    print("=" * 80)
    print("CHARACTERISTIC NEXT-RETURN SCALING LAW SIMULATION")
    print("=" * 80)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Signal months: {N_MONTHS:,}")
    print(f"Firms: {N_FIRMS:,}")
    print(f"Observations: {N_MONTHS * N_FIRMS:,}")
    print(f"Informative characteristics: {N_SIGNAL_CHARS:,}")
    print(f"Irrelevant characteristics: {N_NOISE_CHARS:,}")
    print(f"Feature set: {FEATURE_SET}")
    print("Timing: characteristics at month t predict ret_exc earned in month t + 1")

    df, feature_cols = simulate_characteristic_next_return_panel()

    panel_path = OUTPUT_DIR / "simulated_characteristic_next_return_panel.csv"
    df.to_csv(panel_path, index=False)
    print(f"Saved simulated panel to: {panel_path}")
    print(f"Feature columns: {len(feature_cols)}")
    print(
        "Return diagnostics: "
        f"mean={df['ret_exc'].mean():.6f}, "
        f"std={df['ret_exc'].std():.6f}, "
        f"min={df['ret_exc'].min():.6f}, "
        f"max={df['ret_exc'].max():.6f}"
    )

    val_cutoff, test_cutoff = resolve_split_cutoffs(df["date"])
    print(f"Validation cutoff: {val_cutoff}")
    print(f"Test cutoff: {test_cutoff}")

    config = build_config(val_cutoff=val_cutoff, test_cutoff=test_cutoff)
    experiment = ScalingLawExperiment(config)
    results = experiment.run_from_dataframe(
        df=df,
        feature_cols=feature_cols,
        target_col="ret_exc",
        date_col="date",
        portfolio="panel",
        asset_id_col="permno",
    )

    print("=" * 80)
    print(f"Finished {len(results):,} model result(s).")
    print(f"Results saved in: {OUTPUT_DIR}")
    print("=" * 80)

    try:
        experiment.create_plots(dpi=200)
    except Exception as exc:
        print(f"Plot creation skipped because of an error: {exc}")


if __name__ == "__main__":
    main()
