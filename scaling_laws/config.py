"""Configuration dataclasses for the scaling_laws package.

All experiment configuration lives here. The top-level :class:`ScalingLawConfig`
nests every other config dataclass and is the user-facing entry point.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .enums import (
    ArchitectureMode,
    InitializerType,
    MissingDataPolicy,
    NormalizationType,
    PortfolioMode,
    R2_FAMILIES,
    ResumeMode,
    SplitMode,
    VALID_BENCHMARK_MODES,
)


# ============================================================================
# DEFAULT SCHEDULES
# ============================================================================

def default_taper_schedule(target_params: int) -> List[float]:
    """Default architecture-width schedule for tapered models."""
    if target_params < 1_000:
        return [1.0, 0.5]
    elif target_params < 10_000:
        return [1.0, 1.0, 0.5]
    elif target_params < 100_000:
        return [1.0, 1.0, 0.5, 0.25]
    elif target_params < 1_000_000:
        return [1.0, 1.0, 1.0, 0.5, 0.5, 0.25]
    else:
        return [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.25, 0.25, 0.125]


def default_epochs_schedule(size: int) -> int:
    """Default epochs schedule. `size` is the model parameter count."""
    return max(int(0.1 * (size ** 0.75)), 1) + 100


# ============================================================================
# NESTED CONFIG DATACLASSES
# ============================================================================

@dataclass
class ArchitectureConfig:
    """Architecture-specific model configuration."""
    normalization: NormalizationType = NormalizationType.LAYER
    architecture_mode: ArchitectureMode = ArchitectureMode.TAPERED
    fixed_depth_layers: int = 3
    dropout_rate: float = 0.1
    dropout_middle_only: bool = True
    initializer: InitializerType = InitializerType.HE_NORMAL
    use_input_normalization: bool = True
    taper_schedule: Optional[Callable[[int], List[float]]] = None
    activation: str = "relu"
    output_units: int = 1
    output_activation: Optional[str] = None

    def __post_init__(self):
        if isinstance(self.normalization, str):
            self.normalization = NormalizationType(self.normalization.lower())
        if isinstance(self.architecture_mode, str):
            self.architecture_mode = ArchitectureMode(self.architecture_mode.lower())
        if isinstance(self.initializer, str):
            self.initializer = InitializerType(self.initializer.lower())
        if self.taper_schedule is None:
            self.taper_schedule = default_taper_schedule


@dataclass
class TrainingConfig:
    """Training loop configuration."""
    epochs: Union[int, Callable[[int], int]] = default_epochs_schedule
    train_batch_size: int = 8192
    validation_batch_size: Optional[int] = None
    prediction_batch_size: int = 262144
    learning_rate: float = 0.001
    optimizer: str = "adam"
    clip_norm: Optional[float] = 1.0
    shuffle: bool = True


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration."""
    lr_scheduler_enabled: bool = True
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: Optional[int] = None
    lr_scheduler_min_lr: float = 1e-10


@dataclass
class FuzzyStopConfig:
    """Fuzzy training-stop configuration.

    The scheduled epoch count (from ``TrainingConfig.epochs``) is treated
    as a **hard minimum**: training always runs at least that many epochs
    before fuzzy stop is allowed to do anything. After the minimum is hit,
    training continues for up to ``max_extra_epochs`` further epochs (or
    ``max_extra_fraction * scheduled_epochs`` if set), and fuzzy stop only
    terminates inside that extension window — either because a causal-
    median-smoothed validation metric has stalled for ``patience``
    consecutive epochs, or because the extra-epoch budget is exhausted.
    On stop, weights are optionally restored to the smoothed-best epoch
    found in the extension window.

    The upper budget can be specified two ways (mutually exclusive):
    ``max_extra_epochs`` is an absolute integer; ``max_extra_fraction`` is
    a fraction of the scheduled epoch count (e.g. ``0.5`` for 50% more).
    If both are ``None``, the cap auto-resolves to 50% of the scheduled
    count. Any other field left as ``None`` is auto-resolved at training
    start via :meth:`resolve`.

    Median smoothing is used (rather than mean/EMA) because it survives
    the large isolated spikes typical of chaotic training phases.
    """
    enabled: bool = False
    monitor: str = "val_r2_percent"
    mode: Optional[str] = None
    smoothing_window: Optional[int] = None
    patience: Optional[int] = None
    max_extra_epochs: Optional[int] = None
    max_extra_fraction: Optional[float] = None
    restore_best_weights: bool = True

    def __post_init__(self):
        self.enabled = bool(self.enabled)
        if not isinstance(self.monitor, str) or not self.monitor.strip():
            raise ValueError("FuzzyStopConfig.monitor must be a non-empty string")
        self.monitor = self.monitor.strip()
        if self.mode is not None:
            mode_normalized = str(self.mode).lower()
            if mode_normalized not in ("min", "max"):
                raise ValueError(
                    f"FuzzyStopConfig.mode must be 'min', 'max', or None; got {self.mode!r}"
                )
            self.mode = mode_normalized
        for name in ("smoothing_window", "patience", "max_extra_epochs"):
            value = getattr(self, name)
            if value is None:
                continue
            value = int(value)
            if value < 1:
                raise ValueError(
                    f"FuzzyStopConfig.{name} must be >= 1 when set, got {value}"
                )
            setattr(self, name, value)
        if self.max_extra_fraction is not None:
            fraction = float(self.max_extra_fraction)
            if fraction <= 0:
                raise ValueError(
                    f"FuzzyStopConfig.max_extra_fraction must be > 0 when set, got {fraction}"
                )
            self.max_extra_fraction = fraction
        if self.max_extra_epochs is not None and self.max_extra_fraction is not None:
            raise ValueError(
                "FuzzyStopConfig: set either max_extra_epochs or max_extra_fraction, not both"
            )
        self.restore_best_weights = bool(self.restore_best_weights)

    def resolve(self, scheduled_epochs: int) -> "FuzzyStopConfig":
        """Materialize all auto fields against a concrete epoch budget.

        Returns the receiver unchanged when disabled, so callers can rely
        on the resolved object's flags without an extra branch.
        """
        if not self.enabled:
            return self
        scheduled_epochs = max(1, int(scheduled_epochs))
        mode = self.mode
        if mode is None:
            mode = "min" if "loss" in self.monitor.lower() else "max"
        window = self.smoothing_window
        if window is None:
            window = max(10, min(100, scheduled_epochs // 25))
        patience = self.patience
        if patience is None:
            patience = 2 * window
        cap = self.max_extra_epochs
        if cap is None:
            fraction = self.max_extra_fraction if self.max_extra_fraction is not None else 0.5
            cap = max(1, int(fraction * scheduled_epochs))
        return FuzzyStopConfig(
            enabled=True,
            monitor=self.monitor,
            mode=mode,
            smoothing_window=window,
            patience=patience,
            max_extra_epochs=cap,
            max_extra_fraction=None,
            restore_best_weights=self.restore_best_weights,
        )


@dataclass
class RuntimeConfig:
    """Runtime behavior and reproducibility configuration."""
    show_live_plots: bool = False
    show_live_r2: bool = True
    # Which R² family streams per-epoch as ``r2_percent`` / ``val_r2_percent``,
    # driving BOTH the live training plot and the fuzzy-stop monitor. One of
    # "square_corr", "r2_zero", "r2_classic", "r2_histmean" (see R2_FAMILIES).
    live_r2_metric: str = "square_corr"
    # Hard bottom (percent) for the live/saved R² axis when an UNBOUNDED family
    # is selected (r2_zero / r2_classic / r2_histmean). The top still auto-tracks
    # the running max; only this floor is fixed so the view doesn't jump around.
    # Ignored for square_corr (already bounded to [0, 100]).
    live_r2_floor: float = -5.0
    debug_memory: bool = False
    resume: Union[ResumeMode, str] = ResumeMode.UPDATE_EXISTING
    random_state: int = 42
    # When True, model i in a sweep is seeded with ``random_state + i`` (the
    # original per-model behaviour) so each size gets its own init/shuffle
    # stream. When False (default) every model shares ``random_state`` for a
    # clean size-only comparison.
    vary_seed_per_model: bool = False
    run_name: Optional[str] = None
    # Optional prefix prepended to every model name -> "<prefix>_model_<size>".
    # Used to stamp the date a model could earliest have been estimated.
    model_name_prefix: Optional[str] = None

    def __post_init__(self):
        self.resume = ResumeMode.coerce(self.resume)
        self.vary_seed_per_model = bool(self.vary_seed_per_model)
        self.live_r2_metric = str(self.live_r2_metric)
        if self.live_r2_metric not in R2_FAMILIES:
            raise ValueError(
                f"live_r2_metric must be one of {R2_FAMILIES}, "
                f"got {self.live_r2_metric!r}"
            )


@dataclass
class ComputeConfig:
    """Compute policy and accounting configuration."""
    precision: Union[int, str] = 32
    mixed_precision_policy: Optional[str] = None
    enable_determinism: bool = True
    enable_tensor_float_32: bool = False
    flop_estimator: Optional[Callable[[int, int, int, List[int], Any], Union[int, float]]] = None
    _PRECISION_TO_POLICY: ClassVar[Dict[int, str]] = {
        8: "mixed_float8",
        16: "mixed_float16",
        32: "float32",
        64: "float64",
    }

    def __post_init__(self):
        try:
            self.precision = int(self.precision)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "precision must be one of 8, 16, 32, or 64"
            ) from exc

        if self.precision not in self._PRECISION_TO_POLICY:
            raise ValueError(
                f"precision must be one of {sorted(self._PRECISION_TO_POLICY)}, "
                f"got {self.precision!r}"
            )

        if self.mixed_precision_policy is not None:
            self.mixed_precision_policy = str(self.mixed_precision_policy)
        self.enable_determinism = bool(self.enable_determinism)
        self.enable_tensor_float_32 = bool(self.enable_tensor_float_32)

    def resolve_mixed_precision_policy(self) -> str:
        """Return the Keras dtype policy implied by this compute configuration."""
        if self.mixed_precision_policy:
            return self.mixed_precision_policy
        return self._PRECISION_TO_POLICY[int(self.precision)]

    def estimate_flops_per_epoch(
            self,
            actual_params: int,
            train_samples: int,
            input_dim: int,
            architecture: List[int],
            model: Any
    ) -> Union[int, float]:
        """Estimate per-epoch FLOPs, preserving the historical default formula."""
        if self.flop_estimator is None:
            return 6 * actual_params * train_samples
        return self.flop_estimator(
            actual_params,
            train_samples,
            input_dim,
            architecture,
            model
        )


@dataclass
class ArtifactNames:
    """Configurable names for files and directories written by the experiment."""
    results_pickle: str = "scaling_results.pkl"
    results_json: str = "scaling_results.json"
    portfolio_returns_csv: str = "portfolio_returns.csv"
    test_sample_csv: str = "test_sample.csv"
    models_dir: str = "Models"


@dataclass
class OutputConfig:
    """Output behavior and artifact naming configuration."""
    output_dir: str = "./Output/"
    save_pickle: bool = True
    save_json: bool = True
    save_csv: bool = True
    save_models: bool = False
    save_test_sample: bool = True
    artifacts: Union[ArtifactNames, Dict[str, Any]] = field(default_factory=ArtifactNames)

    def __post_init__(self):
        if isinstance(self.artifacts, dict):
            self.artifacts = ArtifactNames(**self.artifacts)
        elif not isinstance(self.artifacts, ArtifactNames):
            raise TypeError(
                f"artifacts must be ArtifactNames or a dict; got {type(self.artifacts).__name__}"
            )


@dataclass
class PreSplitData:
    """Container for already-materialized train/validation/test arrays."""
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    test_dates: Optional[np.ndarray] = None
    test_asset_ids: Optional[np.ndarray] = None
    test_sample: Optional[pd.DataFrame] = None

    def __post_init__(self):
        self.X_train = np.asarray(self.X_train, dtype=np.float32)
        self.y_train = np.asarray(self.y_train, dtype=np.float32)
        self.X_val = np.asarray(self.X_val, dtype=np.float32)
        self.y_val = np.asarray(self.y_val, dtype=np.float32)
        self.X_test = np.asarray(self.X_test, dtype=np.float32)
        self.y_test = np.asarray(self.y_test, dtype=np.float32)
        if self.test_dates is not None:
            self.test_dates = np.asarray(self.test_dates)
        if self.test_asset_ids is not None:
            self.test_asset_ids = np.asarray(self.test_asset_ids)

    @classmethod
    def from_tuple(cls, values: Tuple[Any, ...]) -> "PreSplitData":
        """Create a pre-split container from the historical 6/7-array tuple."""
        if len(values) not in {6, 7, 8}:
            raise ValueError(
                "pre_split tuples must contain X/y train, val, test arrays "
                "and optionally test_dates and test_asset_ids"
            )
        test_dates = values[6] if len(values) >= 7 else None
        test_asset_ids = values[7] if len(values) == 8 else None
        return cls(*values[:6], test_dates=test_dates, test_asset_ids=test_asset_ids)


@dataclass
class SplitConfig:
    """Data splitting configuration."""
    mode: Union[SplitMode, str] = SplitMode.AUTO
    test_size: Union[float, str] = 0.2
    val_size: Union[float, str] = 0.125
    # Earliest date kept in the TRAIN split (date_cutoffs mode only). None =
    # current behaviour: train runs from the earliest date up to val_size.
    train_start: Optional[Union[str, Any]] = None
    train_mask: Optional[Any] = None
    val_mask: Optional[Any] = None
    test_mask: Optional[Any] = None
    pre_split: Optional[Union[PreSplitData, Dict[str, Any], List[Any], Tuple[Any, ...]]] = None

    def __post_init__(self):
        if not isinstance(self.mode, SplitMode):
            self.mode = SplitMode(str(self.mode).lower())
        if isinstance(self.pre_split, dict):
            self.pre_split = PreSplitData(**self.pre_split)
        elif isinstance(self.pre_split, (list, tuple)):
            self.pre_split = PreSplitData.from_tuple(tuple(self.pre_split))
        elif self.pre_split is not None and not isinstance(self.pre_split, PreSplitData):
            raise TypeError(
                "pre_split must be PreSplitData, a dict, a 6/7/8-array sequence, or None; "
                f"got {type(self.pre_split).__name__}"
            )

    def has_masks(self) -> bool:
        return any(mask is not None for mask in (self.train_mask, self.val_mask, self.test_mask))


@dataclass
class MissingDataConfig:
    """Missing-data handling configuration."""
    policy: Union[MissingDataPolicy, str] = MissingDataPolicy.DROP_ANY

    def __post_init__(self):
        if not isinstance(self.policy, MissingDataPolicy):
            self.policy = MissingDataPolicy(str(self.policy).lower())


@dataclass
class BenchmarkConfig:
    """Benchmark model used in the denominator of OOS R² calculations."""
    mode: Union[str, Callable[..., Any]] = "historical_mean"

    def __post_init__(self):
        if isinstance(self.mode, str):
            normalized = self.mode.strip().lower()
            if normalized not in VALID_BENCHMARK_MODES:
                raise ValueError(
                    f"benchmark mode must be one of {sorted(VALID_BENCHMARK_MODES)}, "
                    f"got {self.mode!r}"
                )
            self.mode = normalized
            return
        if not callable(self.mode):
            raise TypeError(
                "benchmark mode must be a supported string or callable; "
                f"got {type(self.mode).__name__}"
            )


@dataclass
class AnnualizationConfig:
    """Annualization convention for portfolio statistics."""
    periods: int = 12

    def __post_init__(self):
        self.periods = int(self.periods)
        if self.periods <= 0:
            raise ValueError(f"annualization periods must be positive, got {self.periods}")


@dataclass
class TradingConfig:
    """Trading costs and constraints applied to portfolio strategies."""
    transaction_cost_rate: float = 0.0
    leverage_cap: Optional[float] = None
    long_only: bool = False
    allow_short: bool = True

    def __post_init__(self):
        self.transaction_cost_rate = float(self.transaction_cost_rate)
        if self.transaction_cost_rate < 0:
            raise ValueError(
                "transaction_cost_rate must be non-negative, "
                f"got {self.transaction_cost_rate}"
            )
        if self.leverage_cap is not None:
            self.leverage_cap = float(self.leverage_cap)
            if self.leverage_cap < 0:
                raise ValueError(f"leverage_cap must be non-negative, got {self.leverage_cap}")
        self.long_only = bool(self.long_only)
        self.allow_short = bool(self.allow_short)

    def uses_default_constraints(self) -> bool:
        return (
            self.leverage_cap is None
            and not self.long_only
            and self.allow_short
        )

    def uses_default_costs_and_constraints(self) -> bool:
        return self.transaction_cost_rate == 0.0 and self.uses_default_constraints()


@dataclass
class TSStrategyConfig:
    """Time-series forecast strategy configuration."""
    kappa: float = 1.0
    min_periods: int = 6
    winsorize_weights: bool = False
    weight_floor: float = -1.0
    weight_cap: float = 3.0
    signal_lag: int = 1
    standardize_signal: bool = False

    def __post_init__(self):
        self.kappa = float(self.kappa)
        self.min_periods = int(self.min_periods)
        if self.min_periods < 1:
            raise ValueError(f"min_periods must be at least 1, got {self.min_periods}")
        self.winsorize_weights = bool(self.winsorize_weights)
        self.weight_floor = float(self.weight_floor)
        self.weight_cap = float(self.weight_cap)
        if self.weight_floor > self.weight_cap:
            raise ValueError(
                f"weight_floor must be <= weight_cap, got {self.weight_floor} > {self.weight_cap}"
            )
        self.signal_lag = int(self.signal_lag)
        if self.signal_lag < 0:
            raise ValueError(f"signal_lag must be non-negative, got {self.signal_lag}")
        self.standardize_signal = bool(self.standardize_signal)


@dataclass
class PortfolioConfig:
    """Portfolio analysis mode and optional panel asset identity configuration."""
    mode: Union[PortfolioMode, str] = PortfolioMode.PANEL
    asset_id_col: Optional[str] = None
    enabled: bool = True

    def __post_init__(self):
        self.enabled = bool(self.enabled)
        if not isinstance(self.mode, PortfolioMode):
            self.mode = PortfolioMode(str(self.mode).lower())
        if self.mode == PortfolioMode.NONE:
            self.enabled = False
        if self.asset_id_col is not None:
            self.asset_id_col = str(self.asset_id_col)


# ============================================================================
# TOP-LEVEL CONFIG
# ============================================================================

@dataclass
class ScalingLawConfig:
    architecture: ArchitectureConfig = field(default_factory=ArchitectureConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    fuzzy_stop: FuzzyStopConfig = field(default_factory=FuzzyStopConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    missing_data: MissingDataConfig = field(default_factory=MissingDataConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    annualization: AnnualizationConfig = field(default_factory=AnnualizationConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    ts_strategy: TSStrategyConfig = field(default_factory=TSStrategyConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)

    # Parameter Sizes
    param_sizes: Optional[List[str]] = None
    start_at_size: Optional[str] = None
    stop_at_size: Optional[str] = None

    _NESTED_CONFIG_TYPES: ClassVar[Dict[str, Any]] = {
        "architecture": ArchitectureConfig,
        "training": TrainingConfig,
        "scheduler": SchedulerConfig,
        "fuzzy_stop": FuzzyStopConfig,
        "runtime": RuntimeConfig,
        "compute": ComputeConfig,
        "output": OutputConfig,
        "split": SplitConfig,
        "missing_data": MissingDataConfig,
        "benchmark": BenchmarkConfig,
        "annualization": AnnualizationConfig,
        "trading": TradingConfig,
        "ts_strategy": TSStrategyConfig,
        "portfolio": PortfolioConfig,
    }

    def __post_init__(self):
        """Coerce dict/scalar nested-config inputs into proper config instances and validate."""
        for fname, ftype in self._NESTED_CONFIG_TYPES.items():
            val = getattr(self, fname)
            if val is None:
                setattr(self, fname, ftype())
            elif isinstance(val, ftype):
                continue
            elif fname == "benchmark" and (isinstance(val, str) or callable(val)):
                setattr(self, fname, BenchmarkConfig(mode=val))
            elif fname == "annualization" and isinstance(val, (int, float, np.integer, np.floating)):
                setattr(self, fname, AnnualizationConfig(periods=int(val)))
            elif fname == "portfolio" and isinstance(val, (str, PortfolioMode)):
                setattr(self, fname, PortfolioConfig(mode=val))
            elif isinstance(val, dict):
                setattr(self, fname, ftype(**val))
            else:
                raise TypeError(
                    f"{fname} must be {ftype.__name__}, a dict, or None; "
                    f"got {type(val).__name__}"
                )

        # Re-run nested __post_init__ on each (dict coercion already invoked it via
        # the ftype(**val) call, but a fresh ftype() instance and an already-existing
        # instance assigned directly may need re-validation too).
        for fname in self._NESTED_CONFIG_TYPES:
            obj = getattr(self, fname)
            if hasattr(obj, "__post_init__"):
                obj.__post_init__()

        # Set default parameter sizes if not provided
        if self.param_sizes is None:
            self.param_sizes = ['1K', '10K', '100K', '1M']

    @staticmethod
    def _serialize_value(value: Any) -> Any:
        if isinstance(value, Enum):
            return value.value
        if is_dataclass(value):
            return {
                config_field.name: ScalingLawConfig._serialize_value(
                    getattr(value, config_field.name)
                )
                for config_field in fields(value)
            }
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (pd.Series, pd.Index)):
            return value.tolist()
        if isinstance(value, pd.DataFrame):
            return value.to_dict(orient="list")
        if callable(value):
            name = getattr(value, "__name__", value.__class__.__name__)
            module = getattr(value, "__module__", None)
            if module and module != "builtins":
                name = f"{module}.{name}"
            return f"<callable: {name}>"
        if isinstance(value, dict):
            return {
                ScalingLawConfig._serialize_value(key): ScalingLawConfig._serialize_value(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [ScalingLawConfig._serialize_value(item) for item in value]
        return value

    def get_epochs(self, param_size: int) -> int:
        """Get number of epochs for a given parameter size."""
        if callable(self.training.epochs):
            return self.training.epochs(param_size)
        return self.training.epochs

    def get_lr_scheduler_patience(self, epochs: int) -> int:
        """Get LR scheduler patience, auto-calculating if not set."""
        if self.scheduler.lr_scheduler_patience is not None:
            return self.scheduler.lr_scheduler_patience
        return max(epochs // 5, 50)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        config_dict = {
            config_field.name: self._serialize_value(getattr(self, config_field.name))
            for config_field in fields(self)
        }
        return config_dict
