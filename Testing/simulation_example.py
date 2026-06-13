"""
Synthetic firm-month scaling-law example.

This script builds a small but nontrivial panel of simulated firm-month returns.
The data-generating process is intentionally hidden from the estimator:

- 20 monthly Gaussian factors over 200 months by default
- firm-specific latent characteristics
- nonlinear firm betas on the factor levels
- nonlinear firm betas on factor square terms
- nonlinear firm betas on a handful of factor interactions
- an iid idiosyncratic return shock

The neural network only sees raw monthly factors and noisy firm characteristics.
It is not given the true betas, square terms, or interaction terms directly.

Outputs are written to:

    ~/Desktop/scaling law simulation
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from Scaling_Law_Estimator import (
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
N_FACTORS = 20
N_FIRM_CHARS = 12
RANDOM_STATE = 42

OUTPUT_DIR = Path.home() / "Desktop" / "scaling law simulation"


def make_interaction_pairs(n_factors: int) -> List[Tuple[int, int]]:
    """Choose a fixed set of low-overlap factor interaction pairs."""
    candidate_pairs = [
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7),
        (8, 9),
        (10, 11),
        (12, 13),
        (14, 15),
    ]
    return [(i, j) for i, j in candidate_pairs if i < n_factors and j < n_factors]


def simulate_firm_month_panel(
        n_months: int = N_MONTHS,
        n_firms: int = N_FIRMS,
        n_factors: int = N_FACTORS,
        n_firm_chars: int = N_FIRM_CHARS,
        random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Simulate a firm-month return panel with a hidden nonlinear factor structure.

    Returns:
        A tuple of (panel_dataframe, feature_columns).
    """
    rng = np.random.default_rng(random_state)
    dates = pd.date_range("2000-01-31", periods=n_months, freq="ME")

    # Monthly systematic shocks observed by the estimator.
    factors = rng.normal(loc=0.0, scale=1.0, size=(n_months, n_factors))

    # Persistent firm traits observed with noise by the estimator.
    firm_chars = rng.normal(loc=0.0, scale=1.0, size=(n_firms, n_firm_chars))
    firm_alpha = rng.normal(loc=0.0, scale=0.004, size=n_firms)
    firm_log_size = rng.normal(loc=10.0, scale=1.0, size=n_firms)

    interaction_pairs = make_interaction_pairs(n_factors)
    n_interactions = len(interaction_pairs)

    # Unknown maps from firm traits to true factor loadings.
    w_linear = rng.normal(scale=0.55, size=(n_firm_chars, n_factors))
    w_square = rng.normal(scale=0.40, size=(n_firm_chars, n_factors))
    w_interaction = rng.normal(scale=0.45, size=(n_firm_chars, n_interactions))

    beta_linear = np.tanh(firm_chars @ w_linear / np.sqrt(n_firm_chars))
    beta_square = 0.65 * np.tanh(firm_chars @ w_square / np.sqrt(n_firm_chars))
    beta_interaction = 0.75 * np.tanh(firm_chars @ w_interaction / np.sqrt(n_firm_chars))

    month_idx = np.repeat(np.arange(n_months), n_firms)
    firm_idx = np.tile(np.arange(n_firms), n_months)

    panel_factors = factors[month_idx]
    panel_chars = firm_chars[firm_idx] + rng.normal(
        loc=0.0,
        scale=0.08,
        size=(n_months * n_firms, n_firm_chars),
    )

    centered_factor_squares = panel_factors ** 2 - 1.0
    interaction_terms = np.column_stack([
        panel_factors[:, i] * panel_factors[:, j]
        for i, j in interaction_pairs
    ])

    linear_component = (
        0.025
        * np.sum(beta_linear[firm_idx] * panel_factors, axis=1)
        / np.sqrt(n_factors)
    )
    square_component = (
        0.014
        * np.sum(beta_square[firm_idx] * centered_factor_squares, axis=1)
        / np.sqrt(n_factors)
    )
    interaction_component = (
        0.018
        * np.sum(beta_interaction[firm_idx] * interaction_terms, axis=1)
        / np.sqrt(n_interactions)
    )
    direct_characteristic_component = (
        0.006 * np.sin(panel_chars[:, 0])
        - 0.004 * panel_chars[:, 1] ** 2
        + 0.003 * panel_chars[:, 2] * panel_chars[:, 3]
    )
    idiosyncratic_error = rng.normal(
        loc=0.0,
        scale=0.040,
        size=n_months * n_firms,
    )

    excess_return = (
        firm_alpha[firm_idx]
        + linear_component
        + square_component
        + interaction_component
        + direct_characteristic_component
        + idiosyncratic_error
    )

    log_market_equity = (
        firm_log_size[firm_idx]
        + 0.001 * month_idx
        + 0.10 * panel_factors[:, 0]
        + rng.normal(loc=0.0, scale=0.10, size=n_months * n_firms)
    )

    data = {
        "date": dates[month_idx],
        "permno": firm_idx.astype(int),
        "ret_exc": excess_return.astype(np.float32),
        "log_market_equity": log_market_equity.astype(np.float32),
    }

    factor_cols = []
    for k in range(n_factors):
        col = f"factor_{k + 1:02d}"
        data[col] = panel_factors[:, k].astype(np.float32)
        factor_cols.append(col)

    char_cols = []
    for k in range(n_firm_chars):
        col = f"char_{k + 1:02d}"
        data[col] = panel_chars[:, k].astype(np.float32)
        char_cols.append(col)

    df = pd.DataFrame(data)
    feature_cols = factor_cols + char_cols + ["log_market_equity"]

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
    """Create a small size-aware scaling-law run configuration."""
    return ScalingLawConfig(
        normalization=NormalizationType.LAYER,
        architecture_mode=ArchitectureMode.FIXED_DEPTH,
        fixed_depth_layers=5,
        dropout_rate=0.1,
        dropout_middle_only=True,
        initializer=InitializerType.HE_NORMAL,
        use_input_normalization=True,
        param_sizes=[
            "100",
            "500",
            "1K",
            "5K",
            "10K",
            "20K",
            #"50K",
            #"100K",
            #"500K",
            #"1M"
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
    print("SYNTHETIC FIRM-MONTH SCALING LAW SIMULATION")
    print("=" * 80)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Months: {N_MONTHS:,}")
    print(f"Firms: {N_FIRMS:,}")
    print(f"Observations: {N_MONTHS * N_FIRMS:,}")
    print(f"Monthly factors: {N_FACTORS:,}")
    print(f"Firm characteristics: {N_FIRM_CHARS:,}")

    df, feature_cols = simulate_firm_month_panel()

    panel_path = OUTPUT_DIR / "simulated_firm_month_panel.csv"
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
