"""
Characteristic-driven next-month return scaling-law simulation.

This script mirrors the empirical timing in the paper more directly than the
factor-beta simulation:

- 64 firm-month characteristics are observed at month t
- all 64 characteristics enter the hidden expected-return function
- the target is the excess return earned in month t + 1
- expected returns combine linear characteristic levels, nonlinear level
  transforms, pairwise interactions, and nonlinear pairwise interactions

Outputs are written to:

    ~/Desktop/characteristic next return simulation
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Scaling_Law_Estimator import (  # noqa: E402
    ArchitectureMode,
    FuzzyStopConfig,
    InitializerType,
    NormalizationType,
    ResumeMode,
    ScalingLawConfig,
    ScalingLawExperiment,
    print_system_info,
)


N_MONTHS = 200  # More months give longer train/val/test histories.
N_FIRMS = 200  # More firms give a larger cross-section each month.
N_FIRM_CHARS = 64  # More characteristics increase input dimension.
CHAR_PERSISTENCE = 0.92  # Higher values make characteristics move more slowly.
RANDOM_STATE = 42  # Change this to draw a different simulated economy.

# Hidden expected-return controls. Set a strength to 0.0 to remove that block.
LINEAR_CHAR_COUNT = 64  # More linear chars make the signal more additive.
NONLINEAR_LEVEL_CHAR_COUNT = 64  # More chars here add nonlinear single-char effects.
PAIRWISE_INTERACTION_COUNT = 256  # More pairs add more char_i * char_j effects.
NONLINEAR_INTERACTION_COUNT = 256  # More pairs add harder nonlinear interaction effects.

LINEAR_LEVEL_STRENGTH = 0.01  # Higher values make additive effects matter more.
NONLINEAR_LEVEL_STRENGTH = 0.01  # Higher values make nonlinear levels matter more.
PAIRWISE_INTERACTION_STRENGTH = 0.02  # Higher values make simple interactions matter more.
NONLINEAR_INTERACTION_STRENGTH = 0.02  # Higher values make complex interactions matter more.

COMMON_SHOCK_SCALE = 0.01  # Higher values add more month-wide return noise.
IDIOSYNCRATIC_ERROR_SCALE = 0.05  # Higher values add more firm-level return noise.
TARGET_MEAN_MONTHLY_RETURN = 0  # Higher values shift average monthly returns up.
CONDITIONAL_MEAN_CLIP = 0.1  # Lower values cap extreme expected returns more tightly.

OUTPUT_DIR = Path.home() / "Desktop" / "characteristic next return simulation"  # Output location.

# Fuzzy training-stop controls. When enabled, training keeps going past the
# scheduled epoch budget until a causal-median-smoothed validation metric
# confirms a local optimum, then restores weights to that epoch. Any
# parameter left as None is auto-resolved from the scheduled epoch count:
#   smoothing_window -> clip(epochs // 25, 10, 100)
#   patience         -> 2 * smoothing_window
#   max_extra_epochs -> epochs // 2
FUZZY_STOP_ENABLED = True  # Set False to fall back to a hard stop at the scheduled epoch.
FUZZY_STOP_MONITOR = "val_r2_percent"  # Or "val_loss"; auto-flips mode based on the name.
FUZZY_STOP_MODE: Optional[str] = None  # "min", "max", or None to auto-derive from the monitor.
FUZZY_STOP_SMOOTHING_WINDOW: Optional[int] = None  # Median window in epochs; None auto-resolves.
FUZZY_STOP_PATIENCE: Optional[int] = None  # Epochs past best before stopping; None auto-resolves.
FUZZY_STOP_MAX_EXTRA_EPOCHS: Optional[int] = None  # Hard cap on extension; None auto-resolves.
FUZZY_STOP_RESTORE_BEST_WEIGHTS = True  # Snap weights back to the smoothed-best epoch on stop.


def normalized_random_weights(rng: np.random.Generator, size: int) -> np.ndarray:
    """Draw random weights normalized to unit Euclidean length."""
    if size <= 0:
        return np.empty(0, dtype=np.float64)
    weights = rng.normal(size=size)
    norm = np.linalg.norm(weights)
    if norm == 0.0:
        return weights
    return weights / norm


def choose_characteristic_indices(
        n_firm_chars: int,
        count: int,
) -> np.ndarray:
    """Use the first count characteristics for level effects."""
    if count < 0 or count > n_firm_chars:
        raise ValueError(
            f"Characteristic count must be between 0 and {n_firm_chars}, "
            f"got {count}"
        )
    return np.arange(count)


def choose_interaction_pairs(
        rng: np.random.Generator,
        n_firm_chars: int,
        count: int,
) -> List[Tuple[int, int]]:
    """Choose random distinct characteristic pairs for interaction blocks."""
    all_pairs = [
        (i, j)
        for i in range(n_firm_chars)
        for j in range(i + 1, n_firm_chars)
    ]
    if count < 0 or count > len(all_pairs):
        raise ValueError(
            f"Interaction count must be between 0 and {len(all_pairs)}, "
            f"got {count}"
        )
    if count == 0:
        return []
    chosen = rng.choice(len(all_pairs), size=count, replace=False)
    return [all_pairs[int(idx)] for idx in chosen]


def standardize_terms(terms: np.ndarray) -> np.ndarray:
    """Center and scale each hidden term to make strength controls comparable."""
    centered = terms - terms.mean(axis=(0, 1), keepdims=True)
    scale = centered.std(axis=(0, 1), keepdims=True)
    return centered / np.where(scale > 0.0, scale, 1.0)


def simulate_characteristic_next_return_panel(
        n_months: int = N_MONTHS,
        n_firms: int = N_FIRMS,
        n_firm_chars: int = N_FIRM_CHARS,
        char_persistence: float = CHAR_PERSISTENCE,
        linear_char_count: int = LINEAR_CHAR_COUNT,
        nonlinear_level_char_count: int = NONLINEAR_LEVEL_CHAR_COUNT,
        pairwise_interaction_count: int = PAIRWISE_INTERACTION_COUNT,
        nonlinear_interaction_count: int = NONLINEAR_INTERACTION_COUNT,
        linear_level_strength: float = LINEAR_LEVEL_STRENGTH,
        nonlinear_level_strength: float = NONLINEAR_LEVEL_STRENGTH,
        pairwise_interaction_strength: float = PAIRWISE_INTERACTION_STRENGTH,
        nonlinear_interaction_strength: float = NONLINEAR_INTERACTION_STRENGTH,
        common_shock_scale: float = COMMON_SHOCK_SCALE,
        idiosyncratic_error_scale: float = IDIOSYNCRATIC_ERROR_SCALE,
        target_mean_monthly_return: float = TARGET_MEAN_MONTHLY_RETURN,
        conditional_mean_clip: Optional[float] = CONDITIONAL_MEAN_CLIP,
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

    dates = pd.date_range("2000-01-31", periods=n_months + 1, freq="ME")
    signal_dates = dates[:-1]
    return_dates = dates[1:]

    if not -1.0 < char_persistence < 1.0:
        raise ValueError(
            "char_persistence must be strictly between -1 and 1, "
            f"got {char_persistence}"
        )

    characteristics = np.empty(
        (n_months, n_firms, n_firm_chars),
        dtype=np.float64,
    )
    characteristics[0] = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(n_firms, n_firm_chars),
    )
    innovation_scale = np.sqrt(1.0 - char_persistence ** 2)
    for month in range(1, n_months):
        characteristics[month] = (
            char_persistence * characteristics[month - 1]
            + innovation_scale * rng.normal(size=(n_firms, n_firm_chars))
        )

    zero_component = np.zeros((n_months, n_firms), dtype=np.float64)

    linear_indices = choose_characteristic_indices(
        n_firm_chars,
        linear_char_count,
    )
    if linear_level_strength != 0.0 and len(linear_indices) > 0:
        linear_terms = standardize_terms(characteristics[:, :, linear_indices])
        linear_weights = normalized_random_weights(rng, linear_terms.shape[2])
        linear_component = linear_level_strength * np.tensordot(
            linear_terms,
            linear_weights,
            axes=([2], [0]),
        )
    else:
        linear_component = zero_component.copy()

    nonlinear_level_indices = choose_characteristic_indices(
        n_firm_chars,
        nonlinear_level_char_count,
    )
    if nonlinear_level_strength != 0.0 and len(nonlinear_level_indices) > 0:
        level_chars = characteristics[:, :, nonlinear_level_indices]
        nonlinear_level_terms = np.concatenate(
            [
                np.sin(1.3 * level_chars),
                np.tanh(level_chars),
                level_chars ** 2,
                level_chars ** 3 / (1.0 + level_chars ** 2),
            ],
            axis=2,
        )
        nonlinear_level_terms = standardize_terms(nonlinear_level_terms)
        nonlinear_level_weights = normalized_random_weights(
            rng,
            nonlinear_level_terms.shape[2],
        )
        nonlinear_level_component = nonlinear_level_strength * np.tensordot(
            nonlinear_level_terms,
            nonlinear_level_weights,
            axes=([2], [0]),
        )
    else:
        nonlinear_level_component = zero_component.copy()

    interaction_pairs = choose_interaction_pairs(
        rng,
        n_firm_chars,
        pairwise_interaction_count,
    )
    if pairwise_interaction_strength != 0.0 and interaction_pairs:
        pair_i = np.array([i for i, _ in interaction_pairs])
        pair_j = np.array([j for _, j in interaction_pairs])
        interaction_terms = (
            characteristics[:, :, pair_i] * characteristics[:, :, pair_j]
        )
        interaction_terms = standardize_terms(interaction_terms)
        interaction_weights = normalized_random_weights(
            rng,
            interaction_terms.shape[2],
        )
        interaction_component = pairwise_interaction_strength * np.tensordot(
            interaction_terms,
            interaction_weights,
            axes=([2], [0]),
        )
    else:
        interaction_component = zero_component.copy()

    nonlinear_interaction_pairs = choose_interaction_pairs(
        rng,
        n_firm_chars,
        nonlinear_interaction_count,
    )
    if nonlinear_interaction_strength != 0.0 and nonlinear_interaction_pairs:
        pair_i = np.array([i for i, _ in nonlinear_interaction_pairs])
        pair_j = np.array([j for _, j in nonlinear_interaction_pairs])
        left = characteristics[:, :, pair_i]
        right = characteristics[:, :, pair_j]
        nonlinear_interaction_terms = np.concatenate(
            [
                np.tanh(left * right),
                np.sin(left + right),
                np.tanh(left ** 2 - right ** 2),
                (left * right) / (1.0 + np.abs(left - right)),
            ],
            axis=2,
        )
        nonlinear_interaction_terms = standardize_terms(
            nonlinear_interaction_terms
        )
        nonlinear_interaction_weights = normalized_random_weights(
            rng,
            nonlinear_interaction_terms.shape[2],
        )
        nonlinear_interaction_component = (
            nonlinear_interaction_strength
            * np.tensordot(
                nonlinear_interaction_terms,
                nonlinear_interaction_weights,
                axes=([2], [0]),
            )
        )
    else:
        nonlinear_interaction_component = zero_component.copy()

    conditional_mean = (
        linear_component
        + nonlinear_level_component
        + interaction_component
        + nonlinear_interaction_component
    )
    conditional_mean = (
        conditional_mean
        - conditional_mean.mean()
        + target_mean_monthly_return
    )
    if conditional_mean_clip is not None and conditional_mean_clip > 0.0:
        conditional_mean = np.clip(
            conditional_mean,
            -conditional_mean_clip,
            conditional_mean_clip,
        )

    next_month_common_shock = rng.normal(
        loc=0.0,
        scale=common_shock_scale,
        size=n_months,
    )
    idiosyncratic_error = rng.normal(
        loc=0.0,
        scale=idiosyncratic_error_scale,
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
    feature_cols = char_cols

    return df, feature_cols


def get_epochs(size: int) -> int:
    return max(int((0.75 * (size ** 0.75))), 1) + 100


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
            "250",
            "500",
            "1K",
            "5K",
            "10K",
            "20K",
            "50K",
            #"100K",
            #"500K",
            #"1M"
        ],
        epochs=get_epochs,
        batch_size=65536,
        prediction_batch_size=65536,
        learning_rate=0.01,
        clip_norm=1.0,
        lr_scheduler_enabled=False,
        lr_scheduler_factor=0.5,
        lr_scheduler_patience=None,
        lr_scheduler_min_lr=1e-10,
        fuzzy_stop=FuzzyStopConfig(
            enabled=FUZZY_STOP_ENABLED,
            monitor=FUZZY_STOP_MONITOR,
            mode=FUZZY_STOP_MODE,
            smoothing_window=FUZZY_STOP_SMOOTHING_WINDOW,
            patience=FUZZY_STOP_PATIENCE,
            max_extra_epochs=FUZZY_STOP_MAX_EXTRA_EPOCHS,
            restore_best_weights=FUZZY_STOP_RESTORE_BEST_WEIGHTS,
        ),
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
    print(f"Firm characteristics: {N_FIRM_CHARS:,}")
    print(f"Linear characteristic levels: {LINEAR_CHAR_COUNT:,}")
    print(f"Nonlinear level transforms: {NONLINEAR_LEVEL_CHAR_COUNT:,}")
    print(f"Pairwise interactions: {PAIRWISE_INTERACTION_COUNT:,}")
    print(f"Nonlinear interactions: {NONLINEAR_INTERACTION_COUNT:,}")
    print(
        "Strengths: "
        f"linear={LINEAR_LEVEL_STRENGTH:.4f}, "
        f"nonlinear_levels={NONLINEAR_LEVEL_STRENGTH:.4f}, "
        f"interactions={PAIRWISE_INTERACTION_STRENGTH:.4f}, "
        f"nonlinear_interactions={NONLINEAR_INTERACTION_STRENGTH:.4f}"
    )
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
