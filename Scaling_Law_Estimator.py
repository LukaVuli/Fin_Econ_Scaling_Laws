from __future__ import annotations

import ctypes
import gc
import json
import os
import pickle
import sys
import time
import warnings
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, List, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize


# Suppress warnings
warnings.filterwarnings('ignore')

# TensorFlow imports (deferred to allow configuration)
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau
from tensorflow.keras.initializers import GlorotUniform, HeNormal
from tensorflow.keras.layers import (
    BatchNormalization,
    Dense,
    Dropout,
    Input,
    LayerNormalization,
    Normalization,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


# ============================================================================
# ENUMS AND TYPE DEFINITIONS
# ============================================================================

class NormalizationType(str, Enum):
    """Enumeration of supported normalization types."""
    LAYER = "layer"
    BATCH = "batch"
    NONE = "none"


class ArchitectureMode(str, Enum):
    """Enumeration of supported architecture modes."""
    TAPERED = "tapered"
    FIXED_DEPTH = "fixed_depth"


class InitializerType(str, Enum):
    """Enumeration of supported weight initializers."""
    HE_NORMAL = "he_normal"
    GLOROT_UNIFORM = "glorot_uniform"


class PortfolioMode(str, Enum):
    """Enumeration of supported portfolio analysis modes."""
    PANEL = "panel"
    TS = "ts"


class ResumeMode(str, Enum):
    """Enumeration of supported result-file resume behaviors."""
    UPDATE_EXISTING = "update_existing"
    OVERWRITE = "overwrite"
    SKIP_EXISTING = "skip_existing"
    FAIL_IF_EXISTS = "fail_if_exists"

    @classmethod
    def coerce(cls, value: Union["ResumeMode", bool, str]) -> "ResumeMode":
        """Coerce legacy bools and strings into a resume mode."""
        if isinstance(value, cls):
            return value
        if isinstance(value, bool):
            return cls.UPDATE_EXISTING if value else cls.OVERWRITE
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "yes", "1"}:
                return cls.UPDATE_EXISTING
            if normalized in {"false", "no", "0"}:
                return cls.OVERWRITE
            for mode in cls:
                if normalized in {mode.value, mode.name.lower()}:
                    return mode
        raise ValueError(f"Unsupported resume mode: {value!r}")

    def __bool__(self) -> bool:
        """Keep legacy truthiness sensible for code that checks config.resume."""
        return self is not ResumeMode.OVERWRITE

    def __str__(self) -> str:
        return self.value


class SplitMode(str, Enum):
    """Enumeration of supported DataFrame split strategies."""
    AUTO = "auto"
    DATE_CUTOFFS = "date_cutoffs"
    DATE_PROPORTIONS = "date_proportions"
    MASKS = "masks"
    PRE_SPLIT = "pre_split"


class MissingDataPolicy(str, Enum):
    """Enumeration of supported missing-data policies."""
    DROP_ANY = "drop_any"
    DROP_TARGET_ONLY = "drop_target_only"
    ERROR = "error"
    IMPUTE_MEAN = "impute_mean"


VALID_BENCHMARK_MODES = {
    "historical_mean",
    "historical_mean_updating",
    "ar1",
    "ar1_updating",
}


# ============================================================================
# CONFIGURATION DATACLASS
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

    def __post_init__(self):
        if isinstance(self.normalization, str):
            self.normalization = NormalizationType(self.normalization.lower())
        if isinstance(self.architecture_mode, str):
            self.architecture_mode = ArchitectureMode(self.architecture_mode.lower())
        if isinstance(self.initializer, str):
            self.initializer = InitializerType(self.initializer.lower())


@dataclass(init=False)
class TrainingConfig:
    """Training loop configuration."""
    epochs: Union[int, Callable[[int], int]] = 500
    train_batch_size: int = 8192
    validation_batch_size: Optional[int] = None
    prediction_batch_size: int = 262144
    learning_rate: float = 0.001
    optimizer: str = "adam"
    clip_norm: Optional[float] = 1.0

    def __init__(
            self,
            epochs: Union[int, Callable[[int], int]] = 500,
            train_batch_size: int = 8192,
            validation_batch_size: Optional[int] = None,
            prediction_batch_size: int = 262144,
            learning_rate: float = 0.001,
            optimizer: str = "adam",
            clip_norm: Optional[float] = 1.0,
            batch_size: Optional[int] = None
    ):
        if batch_size is not None:
            train_batch_size = batch_size

        self.epochs = epochs
        self.train_batch_size = train_batch_size
        self.validation_batch_size = validation_batch_size
        self.prediction_batch_size = prediction_batch_size
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.clip_norm = clip_norm

    @property
    def batch_size(self) -> int:
        """Backward-compatible alias for train_batch_size."""
        return self.train_batch_size

    @batch_size.setter
    def batch_size(self, value: int):
        self.train_batch_size = value


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration."""
    lr_scheduler_enabled: bool = True
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: Optional[int] = None
    lr_scheduler_min_lr: float = 1e-10


@dataclass
class RuntimeConfig:
    """Runtime behavior and reproducibility configuration."""
    show_live_plots: bool = False
    debug_memory: bool = False
    resume: Union[ResumeMode, bool, str] = ResumeMode.UPDATE_EXISTING
    random_state: int = 42
    run_name: Optional[str] = None

    def __post_init__(self):
        self.resume = ResumeMode.coerce(self.resume)


@dataclass
class ComputeConfig:
    """Compute policy and accounting configuration."""
    precision: Union[int, str] = 32
    mixed_precision_policy: Optional[str] = None
    enable_determinism: bool = True
    flop_estimator: Optional[Callable[[int, int, int, List[int], Model], Union[int, float]]] = None
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
            model: Model
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

    def __post_init__(self):
        if not isinstance(self.mode, PortfolioMode):
            self.mode = PortfolioMode(str(self.mode).lower())
        if self.asset_id_col is not None:
            self.asset_id_col = str(self.asset_id_col)


@dataclass(init=False)
class ScalingLawConfig:
    architecture: ArchitectureConfig = field(default_factory=ArchitectureConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
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
    _TOP_LEVEL_DEFAULTS: ClassVar[Dict[str, Any]] = {
        "param_sizes": None,
        "start_at_size": None,
        "stop_at_size": None,
    }
    _FLAT_ALIASES: ClassVar[Dict[str, Tuple[str, str]]] = {
        "normalization": ("architecture", "normalization"),
        "architecture_mode": ("architecture", "architecture_mode"),
        "fixed_depth_layers": ("architecture", "fixed_depth_layers"),
        "dropout_rate": ("architecture", "dropout_rate"),
        "dropout_middle_only": ("architecture", "dropout_middle_only"),
        "initializer": ("architecture", "initializer"),
        "use_input_normalization": ("architecture", "use_input_normalization"),
        "epochs": ("training", "epochs"),
        "batch_size": ("training", "train_batch_size"),
        "train_batch_size": ("training", "train_batch_size"),
        "validation_batch_size": ("training", "validation_batch_size"),
        "prediction_batch_size": ("training", "prediction_batch_size"),
        "learning_rate": ("training", "learning_rate"),
        "optimizer": ("training", "optimizer"),
        "clip_norm": ("training", "clip_norm"),
        "lr_scheduler_enabled": ("scheduler", "lr_scheduler_enabled"),
        "lr_scheduler_factor": ("scheduler", "lr_scheduler_factor"),
        "lr_scheduler_patience": ("scheduler", "lr_scheduler_patience"),
        "lr_scheduler_min_lr": ("scheduler", "lr_scheduler_min_lr"),
        "show_live_plots": ("runtime", "show_live_plots"),
        "debug_memory": ("runtime", "debug_memory"),
        "resume": ("runtime", "resume"),
        "random_state": ("runtime", "random_state"),
        "run_name": ("runtime", "run_name"),
        "precision": ("compute", "precision"),
        "mixed_precision_policy": ("compute", "mixed_precision_policy"),
        "enable_determinism": ("compute", "enable_determinism"),
        "flop_estimator": ("compute", "flop_estimator"),
        "output_dir": ("output", "output_dir"),
        "save_pickle": ("output", "save_pickle"),
        "save_json": ("output", "save_json"),
        "save_csv": ("output", "save_csv"),
        "save_models": ("output", "save_models"),
        "test_size": ("split", "test_size"),
        "val_size": ("split", "val_size"),
        "benchmark_mode": ("benchmark", "mode"),
        "annualization_periods": ("annualization", "periods"),
        "transaction_cost_rate": ("trading", "transaction_cost_rate"),
        "leverage_cap": ("trading", "leverage_cap"),
        "long_only": ("trading", "long_only"),
        "allow_short": ("trading", "allow_short"),
        "kappa": ("ts_strategy", "kappa"),
        "min_periods": ("ts_strategy", "min_periods"),
        "winsorize_weights": ("ts_strategy", "winsorize_weights"),
        "weight_floor": ("ts_strategy", "weight_floor"),
        "weight_cap": ("ts_strategy", "weight_cap"),
        "signal_lag": ("ts_strategy", "signal_lag"),
        "standardize_signal": ("ts_strategy", "standardize_signal"),
        "portfolio_mode": ("portfolio", "mode"),
        "asset_id_col": ("portfolio", "asset_id_col"),
    }

    def __init__(
            self,
            architecture: Optional[Union[ArchitectureConfig, Dict[str, Any]]] = None,
            training: Optional[Union[TrainingConfig, Dict[str, Any]]] = None,
            scheduler: Optional[Union[SchedulerConfig, Dict[str, Any]]] = None,
            runtime: Optional[Union[RuntimeConfig, Dict[str, Any]]] = None,
            compute: Optional[Union[ComputeConfig, Dict[str, Any]]] = None,
            output: Optional[Union[OutputConfig, Dict[str, Any]]] = None,
            split: Optional[Union[SplitConfig, Dict[str, Any]]] = None,
            missing_data: Optional[Union[MissingDataConfig, Dict[str, Any]]] = None,
            benchmark: Optional[Union[BenchmarkConfig, Dict[str, Any], str, Callable[..., Any]]] = None,
            annualization: Optional[Union[AnnualizationConfig, Dict[str, Any], int, float]] = None,
            trading: Optional[Union[TradingConfig, Dict[str, Any]]] = None,
            ts_strategy: Optional[Union[TSStrategyConfig, Dict[str, Any]]] = None,
            portfolio: Optional[Union[PortfolioConfig, Dict[str, Any], str, PortfolioMode]] = None,
            **kwargs
    ):
        object.__setattr__(
            self,
            "architecture",
            self._coerce_nested_config("architecture", architecture)
        )
        object.__setattr__(
            self,
            "training",
            self._coerce_nested_config("training", training)
        )
        object.__setattr__(
            self,
            "scheduler",
            self._coerce_nested_config("scheduler", scheduler)
        )
        object.__setattr__(
            self,
            "runtime",
            self._coerce_nested_config("runtime", runtime)
        )
        object.__setattr__(
            self,
            "compute",
            self._coerce_nested_config("compute", compute)
        )
        object.__setattr__(
            self,
            "output",
            self._coerce_nested_config("output", output)
        )
        object.__setattr__(
            self,
            "split",
            self._coerce_nested_config("split", split)
        )
        object.__setattr__(
            self,
            "missing_data",
            self._coerce_nested_config("missing_data", missing_data)
        )
        object.__setattr__(
            self,
            "benchmark",
            self._coerce_nested_config("benchmark", benchmark)
        )
        object.__setattr__(
            self,
            "annualization",
            self._coerce_nested_config("annualization", annualization)
        )
        object.__setattr__(
            self,
            "trading",
            self._coerce_nested_config("trading", trading)
        )
        object.__setattr__(
            self,
            "ts_strategy",
            self._coerce_nested_config("ts_strategy", ts_strategy)
        )
        object.__setattr__(
            self,
            "portfolio",
            self._coerce_nested_config("portfolio", portfolio)
        )

        for key, default_value in self._TOP_LEVEL_DEFAULTS.items():
            object.__setattr__(self, key, kwargs.pop(key, default_value))

        for key, value in list(kwargs.items()):
            if key in self._FLAT_ALIASES:
                setattr(self, key, value)
                kwargs.pop(key)

        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected ScalingLawConfig field(s): {unexpected}")

        self.__post_init__()

    def __post_init__(self):
        """Validate and convert configuration values."""
        self.architecture.__post_init__()
        self.runtime.__post_init__()
        self.compute.__post_init__()
        self.output.__post_init__()
        self.split.__post_init__()
        self.missing_data.__post_init__()
        self.benchmark.__post_init__()
        self.annualization.__post_init__()
        self.trading.__post_init__()
        self.ts_strategy.__post_init__()
        self.portfolio.__post_init__()

        # Set default parameter sizes if not provided
        if self.param_sizes is None:
            self.param_sizes = ['1K', '10K', '100K', '1M']

    def __getattr__(self, name: str) -> Any:
        if name in self._FLAT_ALIASES:
            config_name, attr_name = self._FLAT_ALIASES[name]
            return getattr(getattr(self, config_name), attr_name)
        raise AttributeError(f"{self.__class__.__name__!s} has no attribute {name!r}")

    def __setattr__(self, name: str, value: Any):
        if name in self._NESTED_CONFIG_TYPES:
            object.__setattr__(self, name, self._coerce_nested_config(name, value))
            return
        if name in self._FLAT_ALIASES:
            config_name, attr_name = self._FLAT_ALIASES[name]
            setattr(getattr(self, config_name), attr_name, value)
            if config_name == "architecture":
                self.architecture.__post_init__()
            elif config_name == "runtime":
                self.runtime.__post_init__()
            elif config_name == "compute":
                self.compute.__post_init__()
            elif config_name == "output":
                self.output.__post_init__()
            elif config_name == "split":
                self.split.__post_init__()
            elif config_name == "missing_data":
                self.missing_data.__post_init__()
            elif config_name == "benchmark":
                self.benchmark.__post_init__()
            elif config_name == "annualization":
                self.annualization.__post_init__()
            elif config_name == "trading":
                self.trading.__post_init__()
            elif config_name == "ts_strategy":
                self.ts_strategy.__post_init__()
            elif config_name == "portfolio":
                self.portfolio.__post_init__()
            return
        object.__setattr__(self, name, value)

    @classmethod
    def _coerce_nested_config(cls, name: str, value: Optional[Union[Any, Dict[str, Any]]]):
        config_type = cls._NESTED_CONFIG_TYPES[name]
        if value is None:
            return config_type()
        if isinstance(value, config_type):
            return value
        if name == "benchmark" and (isinstance(value, str) or callable(value)):
            return BenchmarkConfig(mode=value)
        if name == "annualization" and isinstance(value, (int, float, np.integer, np.floating)):
            return AnnualizationConfig(periods=int(value))
        if name == "portfolio" and isinstance(value, (str, PortfolioMode)):
            return PortfolioConfig(mode=value)
        if isinstance(value, dict):
            return config_type(**value)
        raise TypeError(
            f"{name} must be {config_type.__name__}, a dict, or None; got {type(value).__name__}"
        )

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
        if callable(self.epochs):
            return self.epochs(param_size)
        return self.epochs

    def get_lr_scheduler_patience(self, epochs: int) -> int:
        """Get LR scheduler patience, auto-calculating if not set."""
        if self.lr_scheduler_patience is not None:
            return self.lr_scheduler_patience
        return max(epochs // 5, 50)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        config_dict = {
            config_field.name: self._serialize_value(getattr(self, config_field.name))
            for config_field in fields(self)
        }
        config_dict["test_size"] = self._serialize_value(self.test_size)
        config_dict["val_size"] = self._serialize_value(self.val_size)
        return config_dict


# ============================================================================
# MEMORY MANAGEMENT UTILITIES
# ============================================================================

class MemoryManager:
    """Utility class for memory management operations."""

    @staticmethod
    def malloc_trim():
        """Force glibc to return memory to OS (Linux only)."""
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass

    @staticmethod
    def print_memory_usage(label: str = ""):
        """Print current memory usage."""
        if not PSUTIL_AVAILABLE:
            return
        try:
            process = psutil.Process()
            mem_info = process.memory_info()
            print(f"[MEMORY {label}] RSS: {mem_info.rss / 1024 ** 3:.2f} GB, "
                  f"VMS: {mem_info.vms / 1024 ** 3:.2f} GB")
        except Exception:
            pass

    @staticmethod
    def aggressive_cleanup():
        """Perform aggressive memory cleanup."""
        # Clear TensorFlow session
        tf.keras.backend.clear_session()

        # Reset default graph (TF1 compatibility)
        try:
            tf.compat.v1.reset_default_graph()
        except Exception:
            pass

        # Force Python garbage collection - run multiple times for circular refs
        gc.collect()
        gc.collect()
        gc.collect()

        # Return memory to OS (Linux)
        MemoryManager.malloc_trim()

        # Close all matplotlib figures
        plt.close('all')


# ============================================================================
# CALLBACKS
# ============================================================================

class R2PercentMetric(tf.keras.metrics.Metric):
    """Streaming squared-correlation R2 metric reported as a percentage."""

    def __init__(self, name: str = "r2_percent", **kwargs):
        super().__init__(name=name, **kwargs)
        self.sum_y = self.add_weight(name="sum_y", initializer="zeros")
        self.sum_y2 = self.add_weight(name="sum_y2", initializer="zeros")
        self.sum_pred = self.add_weight(name="sum_pred", initializer="zeros")
        self.sum_pred2 = self.add_weight(name="sum_pred2", initializer="zeros")
        self.sum_y_pred = self.add_weight(name="sum_y_pred", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.reshape(tf.cast(y_true, tf.float32), [-1])
        y_pred = tf.reshape(tf.cast(y_pred, tf.float32), [-1])

        if sample_weight is not None:
            sample_weight = tf.reshape(tf.cast(sample_weight, tf.float32), [-1])
            y_for_sums = y_true * sample_weight
            y2_for_sums = tf.square(y_true) * sample_weight
            pred_for_sums = y_pred * sample_weight
            pred2_for_sums = tf.square(y_pred) * sample_weight
            y_pred_for_sums = y_true * y_pred * sample_weight
            count = tf.reduce_sum(sample_weight)
        else:
            y_for_sums = y_true
            y2_for_sums = tf.square(y_true)
            pred_for_sums = y_pred
            pred2_for_sums = tf.square(y_pred)
            y_pred_for_sums = y_true * y_pred
            count = tf.cast(tf.size(y_true), tf.float32)

        self.sum_y.assign_add(tf.reduce_sum(y_for_sums))
        self.sum_y2.assign_add(tf.reduce_sum(y2_for_sums))
        self.sum_pred.assign_add(tf.reduce_sum(pred_for_sums))
        self.sum_pred2.assign_add(tf.reduce_sum(pred2_for_sums))
        self.sum_y_pred.assign_add(tf.reduce_sum(y_pred_for_sums))
        self.count.assign_add(count)

    def result(self):
        count = tf.maximum(self.count, 1.0)
        y_var = self.sum_y2 - tf.square(self.sum_y) / count
        pred_var = self.sum_pred2 - tf.square(self.sum_pred) / count
        cov = self.sum_y_pred - (self.sum_y * self.sum_pred) / count
        r2 = tf.square(cov) / tf.maximum(y_var * pred_var, tf.keras.backend.epsilon())
        return 100.0 * tf.clip_by_value(r2, 0.0, 1.0)

    def reset_state(self):
        for variable in self.variables:
            variable.assign(0.0)


class LivePlotCallback(Callback):
    """Callback that displays training progress in real-time with a live plot."""

    def __init__(self):
        super().__init__()
        self.losses: List[float] = []
        self.val_losses: List[float] = []
        self.epochs_list: List[int] = []
        self.fig = None
        self.ax = None

    def on_train_begin(self, logs=None):
        self.losses = []
        self.val_losses = []
        self.epochs_list = []
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 6))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_list.append(epoch)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

        self.ax.clear()
        self.ax.plot(self.epochs_list, self.losses, 'b-', label='Training Loss', linewidth=2)
        self.ax.plot(self.epochs_list, self.val_losses, 'r-', label='Validation Loss', linewidth=2)
        self.ax.set_xlabel('Epoch')
        self.ax.set_ylabel('Loss (RMSE in Percent)')
        self.ax.set_title('Training Progress')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        self.ax.set_yscale('log')

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def on_train_end(self, logs=None):
        plt.ioff()
        self.cleanup()

    def cleanup(self):
        """Manual cleanup method."""
        if self.fig is not None:
            plt.close(self.fig)
        plt.close('all')
        self.losses = []
        self.val_losses = []
        self.epochs_list = []
        self.fig = None
        self.ax = None


class SingleLineProgressCallback(Callback):
    """Callback that prints training progress on a single updating line with timing."""

    def __init__(self):
        super().__init__()
        self.total_epochs = 0
        self.start_time = 0
        self.epoch_times: List[float] = []

    def on_train_begin(self, logs=None):
        self.total_epochs = self.params['epochs']
        self.start_time = time.time()
        self.epoch_times = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = epoch + 1
        total = self.total_epochs
        elapsed = time.time() - self.start_time

        if current > 1:
            epoch_time = elapsed / current
            self.epoch_times.append(epoch_time)
            recent_avg = sum(self.epoch_times[-10:]) / min(len(self.epoch_times), 10)
            eta = recent_avg * (total - current)
        else:
            eta = 0

        elapsed_str = self._format_time(elapsed)
        eta_str = self._format_time(eta)

        bar_length = 40
        filled = int(bar_length * current / total)
        bar = '█' * filled + '░' * (bar_length - filled)

        percent = 100 * current / total
        msg = f"\r[{bar}] {percent:.1f}% | "
        msg += f"Loss: {logs.get('loss', 0):.6f} Val: {logs.get('val_loss', 0):.6f} | "
        msg += f"Train R2: {logs.get('r2_percent', 0):.2f}% "
        msg += f"Val R2: {logs.get('val_r2_percent', 0):.2f}% | "
        msg += f"{elapsed_str} < {eta_str}"

        sys.stdout.write(msg)
        sys.stdout.flush()

    def on_train_end(self, logs=None):
        print()
        self.epoch_times = []

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds into HH:MM:SS or MM:SS string."""
        if seconds < 0:
            return "..."
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"


# ============================================================================
# MODEL BUILDER
# ============================================================================

class ModelBuilder:
    """
    Builder class for creating neural network models with target parameter counts.

    Supports various normalization strategies and architecture modes.
    """

    def __init__(self, config: ScalingLawConfig):
        """
        Initialize the model builder.

        Args:
            config: ScalingLawConfig instance with model configuration
        """
        self.config = config

    def _get_initializer(self, seed: int):
        """Get the appropriate weight initializer."""
        if self.config.initializer == InitializerType.HE_NORMAL:
            return HeNormal(seed=seed)
        elif self.config.initializer == InitializerType.GLOROT_UNIFORM:
            return GlorotUniform(seed=seed)
        else:
            return HeNormal(seed=seed)

    def _get_layer_seed(self, layer_index: int) -> int:
        """Get deterministic per-layer seeds from the configured base seed."""
        return int(self.config.random_state) + layer_index

    def _add_normalization(self, x, layer_index: int):
        """Add normalization layer based on configuration."""
        if self.config.normalization == NormalizationType.LAYER:
            return LayerNormalization()(x)
        elif self.config.normalization == NormalizationType.BATCH:
            return BatchNormalization()(x)
        else:  # NormalizationType.NONE
            return x

    def _should_apply_dropout(self, layer_index: int, total_layers: int) -> bool:
        """Determine if dropout should be applied at this layer."""
        if self.config.dropout_rate <= 0:
            return False
        if self.config.dropout_middle_only:
            return layer_index > 0 and layer_index < total_layers - 1
        return True

    def _get_tapered_architecture_template(self, target_params: int) -> List[float]:
        """Get the architecture template for tapered mode based on model size."""
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

    def build_tapered_model(
            self,
            input_dim: int,
            target_params: int
    ) -> Tuple[Model, Normalization, int, List[int]]:
        """
        Build a neural network with tapered architecture targeting N parameters.

        Args:
            input_dim: Number of input features
            target_params: Target number of parameters

        Returns:
            Tuple of (model, normalizer, actual_params, architecture)
        """
        layers_template = self._get_tapered_architecture_template(target_params)

        def count_params(width):
            total = input_dim * width
            for i in range(len(layers_template) - 1):
                curr_width = int(width * layers_template[i])
                next_width = int(width * layers_template[i + 1])
                total += curr_width * next_width
            total += int(width * layers_template[-1]) * 1
            return total

        # Binary search for optimal width
        low, high = 1, 10000
        best_width = 1
        while low <= high:
            mid = (low + high) // 2
            params = count_params(mid)
            if abs(params - target_params) < abs(count_params(best_width) - target_params):
                best_width = mid
            if params < target_params:
                low = mid + 1
            else:
                high = mid - 1

        layers_config = [max(1, int(best_width * scale)) for scale in layers_template]

        # Build model
        inputs = Input(shape=(input_dim,))

        if self.config.use_input_normalization:
            normalizer = Normalization(axis=-1)
            x = normalizer(inputs)
        else:
            normalizer = None
            x = inputs

        for i, neurons in enumerate(layers_config):
            layer_seed = self._get_layer_seed(i)
            kernel_init = self._get_initializer(seed=layer_seed)
            x = Dense(neurons, activation='relu', kernel_initializer=kernel_init)(x)
            x = self._add_normalization(x, i)

            if self._should_apply_dropout(i, len(layers_config)):
                x = Dropout(self.config.dropout_rate, seed=layer_seed)(x)

        outputs = Dense(1)(x)
        model = Model(inputs=inputs, outputs=outputs)

        actual_params = model.count_params()

        return model, normalizer, actual_params, layers_config

    def build_fixed_depth_model(
            self,
            input_dim: int,
            target_params: int
    ) -> Tuple[Model, Normalization, int, List[int]]:
        """
        Build a neural network with fixed depth and uniform width.

        Args:
            input_dim: Number of input features
            target_params: Target number of parameters

        Returns:
            Tuple of (model, normalizer, actual_params, architecture)
        """
        n_layers = self.config.fixed_depth_layers

        def count_params(width, n_layers):
            if n_layers == 0:
                return input_dim * 1
            total = input_dim * width
            total += (n_layers - 1) * width * width
            total += width * 1
            return total

        def solve_width(target_params, n_layers):
            if n_layers == 0:
                return 0
            elif n_layers == 1:
                return max(1, int(target_params / (input_dim + 1)))
            else:
                a = n_layers - 1
                b = input_dim + 1
                c = -target_params
                discriminant = b ** 2 - 4 * a * c
                w = (-b + discriminant ** 0.5) / (2 * a)
                return max(1, int(w))

        # Calculate optimal width
        width = solve_width(target_params, n_layers)

        # Fine-tune with binary search
        low, high = max(1, width - 50), width + 50
        best_width = width
        best_diff = abs(count_params(width, n_layers) - target_params)

        while low <= high:
            mid = (low + high) // 2
            params = count_params(mid, n_layers)
            diff = abs(params - target_params)

            if diff < best_diff:
                best_diff = diff
                best_width = mid

            if params < target_params:
                low = mid + 1
            else:
                high = mid - 1

        width = best_width

        # Build model
        inputs = Input(shape=(input_dim,))

        if self.config.use_input_normalization:
            normalizer = Normalization(axis=-1)
            x = normalizer(inputs)
        else:
            normalizer = None
            x = inputs

        for i in range(n_layers):
            layer_seed = self._get_layer_seed(i)
            kernel_init = self._get_initializer(seed=layer_seed)
            x = Dense(width, activation='relu', kernel_initializer=kernel_init)(x)
            x = self._add_normalization(x, i)

            if self._should_apply_dropout(i, n_layers):
                x = Dropout(self.config.dropout_rate, seed=layer_seed)(x)

        outputs = Dense(1)(x)
        model = Model(inputs=inputs, outputs=outputs)

        actual_params = model.count_params()
        layers_config = [width] * n_layers

        return model, normalizer, actual_params, layers_config

    def build_model(
            self,
            input_dim: int,
            target_params: int
    ) -> Tuple[Model, Normalization, int, List[int]]:
        """
        Build a model based on the configured architecture mode.

        Args:
            input_dim: Number of input features
            target_params: Target number of parameters

        Returns:
            Tuple of (model, normalizer, actual_params, architecture)
        """
        if self.config.architecture_mode == ArchitectureMode.TAPERED:
            return self.build_tapered_model(input_dim, target_params)
        else:
            return self.build_fixed_depth_model(input_dim, target_params)


# ============================================================================
# RESULTS MANAGER
# ============================================================================

class ResultsManager:
    """Manager class for saving and loading experiment results."""

    def __init__(self, output_dir: Optional[Union[str, Path]], config: Optional[ScalingLawConfig]):
        """
        Initialize the results manager.

        Args:
            output_dir: Directory for saving results
            config: ScalingLawConfig instance
        """
        self.config = config or ScalingLawConfig(output_dir=str(output_dir or "./Output/"))
        self.output_config = self.config.output
        self.artifacts = self.output_config.artifacts

        output_root = output_dir if output_dir is not None else self.output_config.output_dir
        self.output_path = Path(output_root)
        self.output_path.mkdir(parents=True, exist_ok=True)

        self.pkl_path = self.artifact_path("results_pickle")
        self.json_path = self.artifact_path("results_json")
        self.csv_path = self.artifact_path("portfolio_returns_csv")
        self.test_sample_path = self.artifact_path("test_sample_csv")
        self.models_dir = self.artifact_path("models_dir")

    def _resolve_artifact_path(self, artifact_name: str) -> Path:
        path = Path(artifact_name)
        if path.is_absolute():
            return path
        return self.output_path / path

    def artifact_path(self, artifact_field: str) -> Path:
        """Return the resolved path for an ArtifactNames field."""
        return self._resolve_artifact_path(getattr(self.artifacts, artifact_field))

    def model_path(self, model_name: str) -> Path:
        """Return the configured save path for a trained Keras model."""
        return self.models_dir / f"{model_name}.keras"

    def _load_results_from_path(
            self,
            path: Path,
            loader: Callable[[Any], Any],
            binary: bool = False
    ) -> List[Dict[str, Any]]:
        if not path.exists() or path.stat().st_size == 0:
            return []
        try:
            with open(path, "rb" if binary else "r") as f:
                results = loader(f)
            return results if isinstance(results, list) else []
        except Exception:
            return []

    def load_existing_results(self) -> List[Dict[str, Any]]:
        """Load existing result metadata from pickle, falling back to JSON."""
        pickle_results = self._load_results_from_path(self.pkl_path, pickle.load, binary=True)
        if pickle_results:
            return pickle_results
        return self._load_results_from_path(self.json_path, json.load)

    def load_existing_model_names(self) -> set:
        """Load model_name values from existing result artifacts."""
        return {
            result["model_name"]
            for result in self.load_existing_results()
            if isinstance(result, dict) and result.get("model_name")
        }

    def has_existing_results(self) -> bool:
        """Return True if any configured result artifact already contains data."""
        if self.load_existing_results():
            return True
        return self.csv_path.exists() and self.csv_path.stat().st_size > 0

    def initialize_files(self, resume: Union[ResumeMode, bool, str]):
        """
        Initialize output files based on resume mode.

        Args:
            resume: ResumeMode or legacy bool. True preserves/upserts, False resets.
        """
        resume_mode = ResumeMode.coerce(resume)

        if resume_mode == ResumeMode.OVERWRITE:
            print("✓ Fresh start mode: Resetting output files")
            self.pkl_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.pkl_path, 'wb') as f:
                pickle.dump([], f)
            if self.csv_path.exists():
                os.remove(self.csv_path)
            if self.json_path.exists():
                os.remove(self.json_path)
            return

        existing_count = len(self.load_existing_results())

        if resume_mode == ResumeMode.FAIL_IF_EXISTS:
            if self.has_existing_results():
                raise FileExistsError(
                    f"Configured results already exist in {self.output_path}. "
                    "Use resume=True/update_existing, resume=False/overwrite, "
                    "or ResumeMode.SKIP_EXISTING to continue."
                )
            print("✓ Fail-if-exists mode: No existing results found")
            return

        print(f"✓ Resume mode ({resume_mode.value}): Found {existing_count} existing model(s)")
        if resume_mode == ResumeMode.SKIP_EXISTING:
            print("  Existing model_name entries will be skipped; new models will be added")
        else:
            print(f"  Models will be updated/added as training proceeds")

    def save_test_sample(self, test_sample: pd.DataFrame) -> Path:
        """Save the configured test-sample CSV artifact and return its path."""
        self.test_sample_path.parent.mkdir(parents=True, exist_ok=True)
        test_sample.to_csv(self.test_sample_path, index=False)
        return self.test_sample_path

    def save_result_to_pickle(self, result: Dict[str, Any]):
        """
        Save a single result to the pickle file.

        Args:
            result: Result dictionary to save
        """
        if not self.config.save_pickle:
            return

        current_results_list = []

        if self.pkl_path.exists() and self.pkl_path.stat().st_size > 0:
            try:
                with open(self.pkl_path, 'rb') as f:
                    current_results_list = pickle.load(f)
            except Exception as e:
                print(f"⚠ Could not load existing pickle: {e}")
                current_results_list = []

        model_name = result.get('model_name')
        found = False
        for i, existing_result in enumerate(current_results_list):
            if existing_result.get('model_name') == model_name:
                current_results_list[i] = result
                found = True
                print(f"  ↻ Updated existing entry for {model_name}")
                break

        if not found:
            current_results_list.append(result)
            print(f"  + Added new entry for {model_name}")

        self.pkl_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.pkl_path, 'wb') as f:
            pickle.dump(current_results_list, f)

        del current_results_list
        gc.collect()

    def save_result_to_json(self):
        """Update JSON file from pickle file."""
        if not self.config.save_json:
            return

        try:
            with open(self.pkl_path, 'rb') as f:
                results_list = pickle.load(f)

            json_safe_results = []
            for r in results_list:
                r_copy = {k: v for k, v in r.items() if k not in ['decile_returns', 'ts_returns']}
                json_safe_results.append(r_copy)

            self.json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.json_path, 'w') as f:
                json.dump(json_safe_results, f, indent=2)

            del results_list
            del json_safe_results
            gc.collect()

        except Exception as e:
            print(f"⚠ Could not update JSON: {e}")

    def save_decile_returns_to_csv(
            self,
            decile_returns: pd.DataFrame,
            model_identifier: str
    ):
        """
        Save decile returns to CSV file.

        Args:
            decile_returns: DataFrame with decile returns
            model_identifier: Identifier for the model
        """
        if not self.config.save_csv:
            return

        new_data = {}
        for col in decile_returns.columns:
            col_name = f"{model_identifier}_{col}"
            new_data[col_name] = decile_returns[col].values

        new_df = pd.DataFrame(new_data, index=decile_returns.index)
        new_df.index.name = 'date'

        if self.csv_path.exists():
            try:
                existing_df = pd.read_csv(self.csv_path, index_col='date', parse_dates=True)

                cols_to_remove = [c for c in existing_df.columns
                                  if c.startswith(f"{model_identifier}_")]
                if cols_to_remove:
                    existing_df = existing_df.drop(columns=cols_to_remove)
                    print(f"  ↻ Replacing existing CSV columns for {model_identifier}")

                combined_df = existing_df.join(new_df, how='outer')
                self.csv_path.parent.mkdir(parents=True, exist_ok=True)
                combined_df.to_csv(self.csv_path)

                del existing_df
                del combined_df
            except Exception as e:
                print(f"⚠ Error updating CSV, overwriting: {e}")
                self.csv_path.parent.mkdir(parents=True, exist_ok=True)
                new_df.to_csv(self.csv_path)
        else:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            new_df.to_csv(self.csv_path)

        del new_df
        gc.collect()

    def save_ts_returns_to_csv(
            self,
            ts_returns: pd.DataFrame,
            model_identifier: str
    ):
        """
        Save time-series strategy returns to CSV file.

        Args:
            ts_returns: DataFrame with time-series strategy returns
            model_identifier: Identifier for the model
        """
        if not self.config.save_csv:
            return

        new_data = {}
        for col in ts_returns.columns:
            col_name = f"{model_identifier}_{col}"
            new_data[col_name] = ts_returns[col].values

        new_df = pd.DataFrame(new_data, index=ts_returns.index)
        new_df.index.name = 'date'

        if self.csv_path.exists():
            try:
                existing_df = pd.read_csv(self.csv_path, index_col='date', parse_dates=True)

                cols_to_remove = [c for c in existing_df.columns
                                  if c.startswith(f"{model_identifier}_")]
                if cols_to_remove:
                    existing_df = existing_df.drop(columns=cols_to_remove)
                    print(f"  ↻ Replacing existing CSV columns for {model_identifier}")

                combined_df = existing_df.join(new_df, how='outer')
                self.csv_path.parent.mkdir(parents=True, exist_ok=True)
                combined_df.to_csv(self.csv_path)

                del existing_df
                del combined_df
            except Exception as e:
                print(f"⚠ Error updating CSV, overwriting: {e}")
                self.csv_path.parent.mkdir(parents=True, exist_ok=True)
                new_df.to_csv(self.csv_path)
        else:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            new_df.to_csv(self.csv_path)

        del new_df
        gc.collect()

    def load_results(self) -> List[Dict[str, Any]]:
        """Load all results from pickle file."""
        if self.pkl_path.exists():
            with open(self.pkl_path, 'rb') as f:
                return pickle.load(f)
        return []


# ============================================================================
# DATA SPLITTING
# ============================================================================

@dataclass
class DataSplitResult:
    """Materialized train/validation/test split arrays and saveable metadata."""
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    test_dates: Optional[np.ndarray]
    test_asset_ids: Optional[np.ndarray] = None
    test_sample: Optional[pd.DataFrame] = None
    train_dates: Optional[List[Any]] = None
    val_dates: Optional[List[Any]] = None
    test_date_values: Optional[List[Any]] = None

    def as_tuple(self) -> Tuple[np.ndarray, ...]:
        return (
            self.X_train,
            self.y_train,
            self.X_val,
            self.y_val,
            self.X_test,
            self.y_test,
            self.test_dates,
        )


class DataSplitter:
    """Prepare model arrays from DataFrames according to split/missing-data config."""

    def __init__(self, config: ScalingLawConfig):
        self.config = config

    def prepare(
            self,
            df: pd.DataFrame,
            feature_cols: List[str],
            target_col: str = 'xret',
            date_col: str = 'date',
            asset_id_col: Optional[str] = None
    ) -> DataSplitResult:
        """Prepare train/validation/test splits from a DataFrame."""
        mode = self._resolve_mode()
        if mode == SplitMode.PRE_SPLIT:
            return self._prepare_pre_split(self.config.split.pre_split)

        model_cols = self._validate_columns(df, feature_cols, target_col, date_col, asset_id_col)
        required_model_cols = list(dict.fromkeys(list(feature_cols) + [target_col, date_col]))
        position_col = self._internal_position_col(df)
        model_data = df[model_cols].copy()
        model_data[position_col] = np.arange(len(df))
        model_data = self._apply_missing_data(
            model_data,
            required_model_cols,
            feature_cols,
            target_col,
            date_col,
        )
        model_data = model_data.sort_values(date_col)

        if mode == SplitMode.DATE_CUTOFFS:
            train_data, val_data, test_data, split_dates = self._split_by_date_cutoffs(
                model_data, date_col
            )
        elif mode == SplitMode.DATE_PROPORTIONS:
            train_data, val_data, test_data, split_dates = self._split_by_date_proportions(
                model_data, date_col
            )
        elif mode == SplitMode.MASKS:
            train_data, val_data, test_data, split_dates = self._split_by_masks(
                df, model_data, position_col, date_col
            )
        else:
            raise ValueError(f"Unsupported split mode: {mode.value}")

        result = self._build_result(
            original_df=df,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            feature_cols=feature_cols,
            target_col=target_col,
            date_col=date_col,
            asset_id_col=asset_id_col,
            position_col=position_col,
            split_dates=split_dates,
        )

        del train_data, val_data, test_data, model_data
        gc.collect()

        return result

    def _resolve_mode(self) -> SplitMode:
        split_config = self.config.split
        mode = split_config.mode
        if mode != SplitMode.AUTO:
            return mode
        if split_config.pre_split is not None:
            return SplitMode.PRE_SPLIT
        if split_config.has_masks():
            return SplitMode.MASKS
        if isinstance(split_config.test_size, str):
            return SplitMode.DATE_CUTOFFS
        return SplitMode.DATE_PROPORTIONS

    @staticmethod
    def _validate_columns(
            df: pd.DataFrame,
            feature_cols: List[str],
            target_col: str,
            date_col: str,
            asset_id_col: Optional[str] = None
    ) -> List[str]:
        requested_cols = list(feature_cols) + [target_col, date_col]
        if asset_id_col is not None:
            requested_cols.append(asset_id_col)
        missing_cols = [col for col in requested_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                "DataFrame is missing required column(s): "
                f"{', '.join(dict.fromkeys(missing_cols))}"
            )
        return list(dict.fromkeys(requested_cols))

    @staticmethod
    def _internal_position_col(df: pd.DataFrame) -> str:
        base_name = "__scaling_law_original_position__"
        position_col = base_name
        suffix = 1
        while position_col in df.columns:
            position_col = f"{base_name}_{suffix}"
            suffix += 1
        return position_col

    def _apply_missing_data(
            self,
            model_data: pd.DataFrame,
            model_cols: List[str],
            feature_cols: List[str],
            target_col: str,
            date_col: str
    ) -> pd.DataFrame:
        policy = self.config.missing_data.policy
        feature_cols_unique = list(dict.fromkeys(feature_cols))

        if policy == MissingDataPolicy.DROP_ANY:
            return model_data.dropna(subset=model_cols)

        if policy == MissingDataPolicy.DROP_TARGET_ONLY:
            return model_data.dropna(subset=[target_col, date_col])

        if policy == MissingDataPolicy.ERROR:
            missing_counts = model_data[model_cols].isna().sum()
            missing_counts = missing_counts[missing_counts > 0]
            if not missing_counts.empty:
                counts_text = ", ".join(
                    f"{col}={int(count)}" for col, count in missing_counts.items()
                )
                raise ValueError(
                    "MissingDataConfig(policy='error') found missing values in model "
                    f"columns: {counts_text}"
                )
            return model_data

        if policy == MissingDataPolicy.IMPUTE_MEAN:
            cleaned = model_data.dropna(subset=[target_col, date_col]).copy()
            try:
                feature_means = cleaned[feature_cols_unique].mean(numeric_only=False)
            except TypeError as exc:
                raise ValueError(
                    "MissingDataConfig(policy='impute_mean') requires numeric feature "
                    "columns so feature means can be computed"
                ) from exc

            all_missing_features = [
                col for col in feature_cols_unique if pd.isna(feature_means[col])
            ]
            if all_missing_features:
                raise ValueError(
                    "MissingDataConfig(policy='impute_mean') cannot impute feature "
                    "column(s) with all values missing: "
                    f"{', '.join(all_missing_features)}"
                )

            cleaned.loc[:, feature_cols_unique] = cleaned[feature_cols_unique].fillna(feature_means)
            return cleaned

        raise ValueError(f"Unsupported missing-data policy: {policy.value}")

    @staticmethod
    def _unique_dates(model_data: pd.DataFrame, date_col: str) -> List[Any]:
        return sorted(model_data[date_col].unique())

    def _split_by_date_cutoffs(
            self,
            model_data: pd.DataFrame,
            date_col: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, List[Any]]]:
        split_config = self.config.split
        try:
            test_cutoff = pd.Timestamp(split_config.test_size)
            val_cutoff = pd.Timestamp(split_config.val_size)
        except Exception as exc:
            raise ValueError(
                "date_cutoffs split mode requires parseable date-like test_size "
                "and val_size values"
            ) from exc

        if val_cutoff >= test_cutoff:
            raise ValueError(
                "date_cutoffs split mode requires val_size cutoff to be before "
                f"test_size cutoff; got val_size={split_config.val_size!r}, "
                f"test_size={split_config.test_size!r}"
            )

        unique_dates = self._unique_dates(model_data, date_col)
        train_dates = [d for d in unique_dates if pd.Timestamp(d) < val_cutoff]
        val_dates = [
            d for d in unique_dates
            if val_cutoff <= pd.Timestamp(d) < test_cutoff
        ]
        test_dates = [d for d in unique_dates if pd.Timestamp(d) >= test_cutoff]

        return self._split_by_date_lists(
            model_data, date_col, train_dates, val_dates, test_dates
        )

    def _split_by_date_proportions(
            self,
            model_data: pd.DataFrame,
            date_col: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, List[Any]]]:
        split_config = self.config.split
        try:
            test_size = float(split_config.test_size)
            val_size = float(split_config.val_size)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "date_proportions split mode requires numeric test_size and val_size"
            ) from exc

        if not 0 < test_size < 1:
            raise ValueError(f"test_size must be between 0 and 1, got {split_config.test_size!r}")
        if not 0 < val_size < 1:
            raise ValueError(f"val_size must be between 0 and 1, got {split_config.val_size!r}")

        unique_dates = self._unique_dates(model_data, date_col)
        n_dates = len(unique_dates)
        train_end_idx = int(n_dates * (1 - test_size))
        train_dates = unique_dates[:train_end_idx]
        test_dates = unique_dates[train_end_idx:]

        train_val_split_idx = int(len(train_dates) * (1 - val_size))
        val_dates = train_dates[train_val_split_idx:]
        train_dates = train_dates[:train_val_split_idx]

        return self._split_by_date_lists(
            model_data, date_col, train_dates, val_dates, test_dates
        )

    def _split_by_date_lists(
            self,
            model_data: pd.DataFrame,
            date_col: str,
            train_dates: List[Any],
            val_dates: List[Any],
            test_dates: List[Any]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, List[Any]]]:
        train_data = model_data[model_data[date_col].isin(train_dates)]
        val_data = model_data[model_data[date_col].isin(val_dates)]
        test_data = model_data[model_data[date_col].isin(test_dates)]
        return train_data, val_data, test_data, {
            "train": train_dates,
            "val": val_dates,
            "test": test_dates,
        }

    def _split_by_masks(
            self,
            original_df: pd.DataFrame,
            model_data: pd.DataFrame,
            position_col: str,
            date_col: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, List[Any]]]:
        split_config = self.config.split
        if not all(
                mask is not None
                for mask in (split_config.train_mask, split_config.val_mask, split_config.test_mask)
        ):
            raise ValueError(
                "masks split mode requires train_mask, val_mask, and test_mask "
                "aligned to the original DataFrame index"
            )

        train_mask = self._mask_to_array(split_config.train_mask, "train_mask", original_df.index)
        val_mask = self._mask_to_array(split_config.val_mask, "val_mask", original_df.index)
        test_mask = self._mask_to_array(split_config.test_mask, "test_mask", original_df.index)

        overlap = (train_mask & val_mask) | (train_mask & test_mask) | (val_mask & test_mask)
        if np.any(overlap):
            raise ValueError(
                f"Split masks overlap on {int(np.sum(overlap)):,} original DataFrame row(s)"
            )

        original_positions = model_data[position_col].to_numpy(dtype=int)
        train_data = model_data.loc[train_mask[original_positions]]
        val_data = model_data.loc[val_mask[original_positions]]
        test_data = model_data.loc[test_mask[original_positions]]

        return train_data, val_data, test_data, {
            "train": self._unique_dates(train_data, date_col),
            "val": self._unique_dates(val_data, date_col),
            "test": self._unique_dates(test_data, date_col),
        }

    @staticmethod
    def _mask_to_array(mask: Any, name: str, df_index: pd.Index) -> np.ndarray:
        if isinstance(mask, pd.Series):
            if mask.index.equals(df_index):
                mask_values = mask.to_numpy()
            else:
                if not df_index.is_unique:
                    raise ValueError(
                        f"{name} must have exactly the original DataFrame index when "
                        "the DataFrame index contains duplicate labels"
                    )
                aligned = mask.reindex(df_index)
                if aligned.isna().any():
                    raise ValueError(
                        f"{name} is missing labels from the original DataFrame index"
                    )
                mask_values = aligned.to_numpy()
        else:
            mask_values = np.asarray(mask)
            if mask_values.ndim != 1:
                raise ValueError(f"{name} must be a one-dimensional boolean mask")
            if len(mask_values) != len(df_index):
                raise ValueError(
                    f"{name} length mismatch: expected {len(df_index):,} values aligned "
                    f"to the original DataFrame, got {len(mask_values):,}"
                )

        if len(mask_values) != len(df_index):
            raise ValueError(
                f"{name} length mismatch: expected {len(df_index):,} values aligned "
                f"to the original DataFrame, got {len(mask_values):,}"
            )
        return np.asarray(mask_values, dtype=bool)

    def _build_result(
            self,
            original_df: pd.DataFrame,
            train_data: pd.DataFrame,
            val_data: pd.DataFrame,
            test_data: pd.DataFrame,
            feature_cols: List[str],
            target_col: str,
            date_col: str,
            asset_id_col: Optional[str],
            position_col: str,
            split_dates: Dict[str, List[Any]],
    ) -> DataSplitResult:
        self._validate_non_empty_splits(train_data, val_data, test_data)

        X_train = train_data[feature_cols].values.astype(np.float32)
        y_train = train_data[target_col].values.astype(np.float32)
        X_val = val_data[feature_cols].values.astype(np.float32)
        y_val = val_data[target_col].values.astype(np.float32)
        X_test = test_data[feature_cols].values.astype(np.float32)
        y_test = test_data[target_col].values.astype(np.float32)
        test_dates_array = test_data[date_col].values
        test_asset_ids = (
            test_data[asset_id_col].values
            if asset_id_col is not None
            else None
        )

        save_cols = [
            c for c in [
                date_col,
                target_col,
                asset_id_col,
                'id',
                'permno',
                'market_equity',
                'lme',
                'excntry',
            ]
            if c in original_df.columns
        ]
        save_cols = list(dict.fromkeys(save_cols))
        test_positions = test_data[position_col].to_numpy(dtype=int)
        test_sample = original_df.iloc[test_positions][save_cols].copy()

        result = DataSplitResult(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            test_dates=test_dates_array,
            test_asset_ids=test_asset_ids,
            test_sample=test_sample,
            train_dates=split_dates["train"],
            val_dates=split_dates["val"],
            test_date_values=split_dates["test"],
        )
        self._validate_result_arrays(result)
        return result

    def _prepare_pre_split(
            self,
            pre_split: Optional[PreSplitData]
    ) -> DataSplitResult:
        if pre_split is None:
            raise ValueError("pre_split split mode requires SplitConfig.pre_split")

        result = DataSplitResult(
            X_train=pre_split.X_train,
            y_train=pre_split.y_train,
            X_val=pre_split.X_val,
            y_val=pre_split.y_val,
            X_test=pre_split.X_test,
            y_test=pre_split.y_test,
            test_dates=pre_split.test_dates,
            test_asset_ids=pre_split.test_asset_ids,
            test_sample=pre_split.test_sample,
        )
        self._validate_result_arrays(result)
        return result

    @staticmethod
    def _validate_non_empty_splits(
            train_data: pd.DataFrame,
            val_data: pd.DataFrame,
            test_data: pd.DataFrame
    ):
        empty_splits = [
            split_name
            for split_name, split_data in (
                ("train", train_data),
                ("val", val_data),
                ("test", test_data),
            )
            if len(split_data) == 0
        ]
        if empty_splits:
            raise ValueError(
                "Data split produced empty split(s): "
                f"{', '.join(empty_splits)}. Check split_config cutoffs, "
                "proportions, masks, and missing-data policy."
            )

    @staticmethod
    def _validate_result_arrays(result: DataSplitResult):
        split_pairs = (
            ("train", result.X_train, result.y_train),
            ("val", result.X_val, result.y_val),
            ("test", result.X_test, result.y_test),
        )
        for split_name, X, y in split_pairs:
            if len(X) != len(y):
                raise ValueError(
                    f"{split_name} split has mismatched X/y lengths: "
                    f"X_{split_name}={len(X):,}, y_{split_name}={len(y):,}"
                )
            if len(X) == 0:
                raise ValueError(f"{split_name} split is empty")

        if result.test_dates is not None and len(result.test_dates) != len(result.y_test):
            raise ValueError(
                "test_dates length mismatch: "
                f"test_dates={len(result.test_dates):,}, y_test={len(result.y_test):,}"
            )
        if result.test_asset_ids is not None and len(result.test_asset_ids) != len(result.y_test):
            raise ValueError(
                "test_asset_ids length mismatch: "
                f"test_asset_ids={len(result.test_asset_ids):,}, y_test={len(result.y_test):,}"
            )


# ============================================================================
# PORTFOLIO ANALYZER
# ============================================================================

class PortfolioAnalyzer:
    """Analyzer for computing portfolio statistics from model predictions."""

    @staticmethod
    def _panel_strategy_weights(group: pd.DataFrame, strategy_name: str) -> pd.Series:
        """Build per-asset panel portfolio weights for optional turnover/cost accounting."""
        if 'asset_id' not in group.columns:
            return pd.Series(dtype=float)

        weights = pd.Series(0.0, index=group.index)

        if strategy_name == 'Forecast_Weighted':
            predictions = group['prediction'].values
            sum_abs_pred = np.abs(predictions).sum()
            if sum_abs_pred > 0:
                weights.loc[group.index] = predictions / sum_abs_pred
        else:
            leg_map = {
                'LS_10': ([10], [1]),
                'LS_30': ([8, 9, 10], [1, 2, 3]),
                'LS_50': ([6, 7, 8, 9, 10], [1, 2, 3, 4, 5]),
            }
            long_deciles, short_deciles = leg_map[strategy_name]

            def assign_decile_leg(deciles: List[int], sign: float):
                available_deciles = [
                    decile for decile in deciles
                    if np.any(group['decile'].values == decile)
                ]
                if not available_deciles:
                    return
                decile_weight = sign / len(available_deciles)
                for decile in available_deciles:
                    decile_mask = group['decile'] == decile
                    n_assets = int(decile_mask.sum())
                    if n_assets > 0:
                        weights.loc[decile_mask] = decile_weight / n_assets

            assign_decile_leg(long_deciles, 1.0)
            assign_decile_leg(short_deciles, -1.0)

        asset_weights = pd.Series(weights.values, index=group['asset_id'].values)
        return asset_weights.groupby(level=0).sum()

    @staticmethod
    def _compute_panel_trading_costs(
            test_df: pd.DataFrame,
            trading_config: TradingConfig
    ) -> Dict[str, Dict[str, pd.Series]]:
        """Compute optional panel turnover/cost series without changing default behavior."""
        cost_results: Dict[str, Dict[str, pd.Series]] = {}
        strategies = ['LS_10', 'LS_30', 'LS_50', 'Forecast_Weighted']

        for strategy_name in strategies:
            gross_returns = {}
            turnovers = {}
            transaction_costs = {}
            net_returns = {}
            previous_weights = pd.Series(dtype=float)

            for date_value, group in test_df.sort_values('date').groupby('date', sort=True):
                weights = PortfolioAnalyzer._panel_strategy_weights(group, strategy_name)
                gross_return = float(
                    np.dot(
                        weights.reindex(group['asset_id'].values).fillna(0.0).values,
                        group['actual_return'].values,
                    )
                )

                all_assets = weights.index.union(previous_weights.index)
                turnover = float(
                    np.abs(
                        weights.reindex(all_assets, fill_value=0.0)
                        - previous_weights.reindex(all_assets, fill_value=0.0)
                    ).sum()
                )
                transaction_cost = float(trading_config.transaction_cost_rate * turnover)

                gross_returns[date_value] = gross_return
                turnovers[date_value] = turnover
                transaction_costs[date_value] = transaction_cost
                net_returns[date_value] = gross_return - transaction_cost
                previous_weights = weights

            cost_results[strategy_name] = {
                'gross_return': pd.Series(gross_returns),
                'turnover': pd.Series(turnovers),
                'transaction_cost': pd.Series(transaction_costs),
                'net_return': pd.Series(net_returns),
            }

        return cost_results

    @staticmethod
    def analyze_predictions(
            test_dates: np.ndarray,
            predictions: np.ndarray,
            actual_returns: np.ndarray,
            annualization_periods: int = 12,
            asset_ids: Optional[np.ndarray] = None,
            trading_config: Optional[TradingConfig] = None
    ) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """
        Analyze model predictions and compute portfolio statistics (panel data).

        Args:
            test_dates: Array of test dates
            predictions: Model predictions
            actual_returns: Actual returns
            annualization_periods: Number of periods per year for annualized stats
            asset_ids: Optional asset identifiers aligned with test observations
            trading_config: Optional trading cost/constraint config for future panel costs

        Returns:
            Tuple of (portfolio_stats dict, decile_returns DataFrame)
        """
        trading_config = trading_config or TradingConfig()
        annualization_periods = int(annualization_periods)
        if annualization_periods <= 0:
            raise ValueError(
                f"annualization_periods must be positive, got {annualization_periods}"
            )
        if asset_ids is not None and len(asset_ids) != len(actual_returns):
            raise ValueError(
                "asset_ids length mismatch: "
                f"asset_ids={len(asset_ids):,}, actual_returns={len(actual_returns):,}"
            )

        test_df = pd.DataFrame({
            'date': test_dates,
            'prediction': predictions.copy(),
            'actual_return': actual_returns.copy()
        })
        if asset_ids is not None:
            test_df['asset_id'] = np.asarray(asset_ids).copy()

        def assign_deciles(predictions: pd.Series):
            n_stocks = len(predictions)
            if n_stocks < 10:
                n_quantiles = min(n_stocks, 10)
                deciles = pd.qcut(
                    predictions, q=n_quantiles,
                    labels=False, duplicates='drop'
                ) + 1
            else:
                deciles = pd.qcut(
                    predictions, q=10,
                    labels=False, duplicates='drop'
                ) + 1
            return pd.Series(deciles, index=predictions.index)

        test_df['decile'] = (
            test_df.groupby('date')['prediction']
            .transform(assign_deciles)
        )

        def calc_forecast_weighted(group):
            predictions = group['prediction'].values
            actual_returns = group['actual_return'].values
            sum_abs_pred = np.abs(predictions).sum()
            if sum_abs_pred > 0:
                weights = predictions / sum_abs_pred
            else:
                weights = np.zeros(len(predictions))
            return (weights * actual_returns).sum()

        forecast_weighted_returns = test_df.groupby('date').apply(calc_forecast_weighted)
        panel_costs_applied = (
            asset_ids is not None
            and trading_config.transaction_cost_rate > 0
        )
        panel_costs = (
            PortfolioAnalyzer._compute_panel_trading_costs(test_df, trading_config)
            if panel_costs_applied
            else None
        )

        decile_returns = test_df.groupby(['date', 'decile'])['actual_return'].mean().unstack(fill_value=np.nan)

        del test_df
        gc.collect()

        for i in range(1, 11):
            if i not in decile_returns.columns:
                decile_returns[i] = np.nan

        decile_returns = decile_returns[[i for i in range(1, 11)]]
        decile_returns.columns = [f'D{i}' for i in range(1, 11)]
        decile_returns['Forecast_Weighted'] = forecast_weighted_returns

        del forecast_weighted_returns
        gc.collect()

        # Long-short portfolios
        top50 = decile_returns[['D6', 'D7', 'D8', 'D9', 'D10']].mean(axis=1)
        bottom50 = decile_returns[['D1', 'D2', 'D3', 'D4', 'D5']].mean(axis=1)
        decile_returns['LS_50'] = top50 - bottom50

        top30 = decile_returns[['D8', 'D9', 'D10']].mean(axis=1)
        bottom30 = decile_returns[['D1', 'D2', 'D3']].mean(axis=1)
        decile_returns['LS_30'] = top30 - bottom30

        decile_returns['LS_10'] = decile_returns['D10'] - decile_returns['D1']

        if panel_costs is not None:
            for strategy_name, cost_series in panel_costs.items():
                gross_col = f'{strategy_name}_Gross'
                turnover_col = f'{strategy_name}_Turnover'
                cost_col = f'{strategy_name}_Transaction_Cost'
                decile_returns[gross_col] = cost_series['gross_return'].reindex(decile_returns.index)
                decile_returns[turnover_col] = cost_series['turnover'].reindex(decile_returns.index)
                decile_returns[cost_col] = cost_series['transaction_cost'].reindex(decile_returns.index)
                decile_returns[strategy_name] = cost_series['net_return'].reindex(decile_returns.index)

        del top50, bottom50, top30, bottom30
        gc.collect()

        # Calculate statistics
        ann_factor = np.sqrt(annualization_periods)
        ann_periods = annualization_periods

        ls_stats = {}
        for ls_name, ls_label in [
            ('LS_10', '10% Breakpoint'),
            ('LS_30', '30% Breakpoint'),
            ('LS_50', '50% Breakpoint'),
            ('Forecast_Weighted', 'Forecast Weighted')
        ]:
            returns = decile_returns[ls_name].dropna()
            mean_ret = returns.mean() * ann_periods
            std_ret = returns.std() * ann_factor
            sharpe = (mean_ret / std_ret) if std_ret > 0 else 0

            ls_stats[ls_name] = {
                'mean': float(mean_ret),
                'std': float(std_ret),
                'sharpe': float(sharpe),
                'label': ls_label
            }

        ls_stats['metadata'] = {
            'annualization_periods': int(annualization_periods),
            'asset_ids_provided': bool(asset_ids is not None),
            'trading_config': ScalingLawConfig._serialize_value(trading_config),
            'panel_trading_costs_applied': bool(panel_costs_applied),
            'panel_constraints_applied': False,
        }
        if panel_costs is not None:
            ls_stats['metadata']['panel_trading_costs'] = {
                strategy_name: {
                    'avg_turnover': float(cost_series['turnover'].mean()),
                    'total_turnover': float(cost_series['turnover'].sum()),
                    'avg_transaction_cost': float(cost_series['transaction_cost'].mean()),
                    'total_transaction_cost': float(cost_series['transaction_cost'].sum()),
                }
                for strategy_name, cost_series in panel_costs.items()
            }

        return ls_stats, decile_returns

    @staticmethod
    def _copy_ts_strategy_config(strategy_config: Optional[TSStrategyConfig]) -> TSStrategyConfig:
        base_config = strategy_config or TSStrategyConfig()
        return TSStrategyConfig(**{
            config_field.name: getattr(base_config, config_field.name)
            for config_field in fields(TSStrategyConfig)
        })

    @staticmethod
    def _apply_trading_constraints(
            weights: pd.Series,
            trading_config: TradingConfig
    ) -> pd.Series:
        constrained = weights.copy()
        if trading_config.long_only or not trading_config.allow_short:
            constrained = constrained.clip(lower=0.0)
        if trading_config.leverage_cap is not None:
            lower = 0.0 if (trading_config.long_only or not trading_config.allow_short) else -trading_config.leverage_cap
            constrained = constrained.clip(lower=lower, upper=trading_config.leverage_cap)
        return constrained

    @staticmethod
    def analyze_predictions_ts(
            test_dates: np.ndarray,
            predictions: np.ndarray,
            actual_returns: np.ndarray,
            kappa: Optional[float] = None,
            min_periods: Optional[int] = None,
            winsorize_weights: Optional[bool] = None,
            weight_floor: Optional[float] = None,
            weight_cap: Optional[float] = None,
            train_returns: Optional[np.ndarray] = None,
            strategy_config: Optional[TSStrategyConfig] = None,
            trading_config: Optional[TradingConfig] = None,
            annualization_periods: int = 12,
            signal_lag: Optional[int] = None,
            standardize_signal: Optional[bool] = None
    ) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """
        Analyze model predictions for time-series (single asset) strategy.

        Weights are centered at 1 (buy-and-hold baseline):
            π_t = 1 + κ * (μ_hat_{t+1} - r̄_t^{hist})

        Args:
            test_dates: Array of test dates
            predictions: Model predictions (forecasted returns)
            actual_returns: Actual returns
            kappa: Sensitivity to forecast deviations (default=1.0)
                - Higher κ = more aggressive deviations from buy-and-hold
                - κ=0 gives pure buy-and-hold
            min_periods: Minimum periods before computing historical mean
            winsorize_weights: Whether to cap extreme weights
            weight_floor: Minimum weight (default=-1.0, i.e. max 100% short)
            weight_cap: Maximum weight (default=3.0, i.e. max 200% leverage)
            train_returns: Training returns for warm-starting the historical mean
            strategy_config: TS strategy configuration
            trading_config: Trading cost/constraint configuration
            annualization_periods: Number of periods per year for annualized stats
            signal_lag: Optional override for strategy_config.signal_lag
            standardize_signal: Optional override for strategy_config.standardize_signal

        Returns:
            Tuple of (portfolio_stats dict, ts_returns DataFrame)
        """
        strategy_config = PortfolioAnalyzer._copy_ts_strategy_config(strategy_config)
        overrides = {
            'kappa': kappa,
            'min_periods': min_periods,
            'winsorize_weights': winsorize_weights,
            'weight_floor': weight_floor,
            'weight_cap': weight_cap,
            'signal_lag': signal_lag,
            'standardize_signal': standardize_signal,
        }
        for key, value in overrides.items():
            if value is not None:
                setattr(strategy_config, key, value)
        strategy_config.__post_init__()

        trading_config = trading_config or TradingConfig()
        trading_config.__post_init__()
        annualization_periods = int(annualization_periods)
        if annualization_periods <= 0:
            raise ValueError(
                f"annualization_periods must be positive, got {annualization_periods}"
            )

        # Create DataFrame with dates
        ts_df = pd.DataFrame({
            'date': test_dates,
            'prediction': predictions.copy(),
            'actual_return': actual_returns.copy()
        })

        ts_df = ts_df.sort_values('date').reset_index(drop=True)

        # Initialize with training data if provided
        if train_returns is not None:
            initial_sum = np.sum(train_returns)
            initial_sum_sq = np.sum(train_returns ** 2)
            initial_count = len(train_returns)
        else:
            initial_sum = 0.0
            initial_sum_sq = 0.0
            initial_count = 0

        # Compute expanding mean and std of actual returns
        hist_mean = np.zeros(len(ts_df))
        hist_std = np.zeros(len(ts_df))

        cumsum = initial_sum
        cumsum_sq = initial_sum_sq
        count = initial_count

        for i in range(len(ts_df)):
            if count >= strategy_config.min_periods:
                mean_val = cumsum / count
                var_val = (cumsum_sq / count) - mean_val ** 2
                std_val = np.sqrt(max(var_val, 1e-10))  # Floor to avoid div by zero
                hist_mean[i] = mean_val
                hist_std[i] = std_val
            else:
                hist_mean[i] = np.nan
                hist_std[i] = np.nan

            # Update running sums with current return
            ret = ts_df.loc[i, 'actual_return']
            cumsum += ret
            cumsum_sq += ret ** 2
            count += 1

        ts_df['hist_mean'] = hist_mean
        ts_df['hist_std'] = hist_std

        if strategy_config.standardize_signal:
            ts_df['z_score'] = (ts_df['prediction'] - ts_df['hist_mean']) / ts_df['hist_std']
        else:
            ts_df['z_score'] = ts_df['prediction'] - ts_df['hist_mean']

        # Weight = 1 + κ * z_score
        ts_df['weight'] = 1.0 + strategy_config.kappa * ts_df['z_score']

        # Winsorize
        if strategy_config.winsorize_weights:
            ts_df['weight'] = ts_df['weight'].clip(
                strategy_config.weight_floor,
                strategy_config.weight_cap,
            )
        ts_df['weight'] = PortfolioAnalyzer._apply_trading_constraints(
            ts_df['weight'],
            trading_config,
        )

        # Strategy return
        ts_df['weight_used'] = ts_df['weight'].shift(strategy_config.signal_lag)
        ts_df['gross_strategy_return'] = ts_df['weight_used'] * ts_df['actual_return']

        # Drop NaN
        ts_df = ts_df.dropna(subset=['gross_strategy_return']).reset_index(drop=True)
        ts_df['turnover'] = ts_df['weight_used'].diff().abs()
        if len(ts_df) > 0:
            ts_df.loc[0, 'turnover'] = abs(ts_df.loc[0, 'weight_used'])
        ts_df['turnover'] = ts_df['turnover'].fillna(0.0)
        ts_df['transaction_cost'] = trading_config.transaction_cost_rate * ts_df['turnover']
        ts_df['strategy_return'] = ts_df['gross_strategy_return'] - ts_df['transaction_cost']

        # Output DataFrame
        ts_returns = pd.DataFrame({
            'actual_return': ts_df['actual_return'].values,
            'prediction': ts_df['prediction'].values,
            'hist_mean': ts_df['hist_mean'].values,
            'hist_std': ts_df['hist_std'].values,
            'z_score': ts_df['z_score'].values,
            'weight': ts_df['weight_used'].values,
            'turnover': ts_df['turnover'].values,
            'transaction_cost': ts_df['transaction_cost'].values,
            'gross_strategy_return': ts_df['gross_strategy_return'].values,
            'strategy_return': ts_df['strategy_return'].values
        }, index=pd.to_datetime(ts_df['date'].values))

        # Stats
        ann_factor = np.sqrt(annualization_periods)
        ann_periods = annualization_periods

        strategy_rets = ts_returns['strategy_return']
        gross_strategy_rets = ts_returns['gross_strategy_return']
        actual_rets = ts_returns['actual_return']

        strategy_mean = strategy_rets.mean() * ann_periods
        strategy_std = strategy_rets.std() * ann_factor
        strategy_sharpe = (strategy_mean / strategy_std) if strategy_std > 0 else 0

        gross_strategy_mean = gross_strategy_rets.mean() * ann_periods
        gross_strategy_std = gross_strategy_rets.std() * ann_factor
        gross_strategy_sharpe = (
            gross_strategy_mean / gross_strategy_std
        ) if gross_strategy_std > 0 else 0

        bh_mean = actual_rets.mean() * ann_periods
        bh_std = actual_rets.std() * ann_factor
        bh_sharpe = (bh_mean / bh_std) if bh_std > 0 else 0

        hit_rate = (strategy_rets > 0).mean()
        avg_weight = ts_returns['weight'].mean()
        std_weight = ts_returns['weight'].std()
        avg_z = ts_returns['z_score'].mean()
        std_z = ts_returns['z_score'].std()
        avg_turnover = ts_returns['turnover'].mean()
        total_turnover = ts_returns['turnover'].sum()
        avg_transaction_cost = ts_returns['transaction_cost'].mean()
        total_transaction_cost = ts_returns['transaction_cost'].sum()

        ic = ts_returns['prediction'].shift(strategy_config.signal_lag).corr(ts_returns['actual_return'])
        signal_corr = ts_returns['z_score'].shift(strategy_config.signal_lag).corr(ts_returns['actual_return'])

        ts_stats = {
            'strategy': {
                'mean': float(strategy_mean),
                'std': float(strategy_std),
                'sharpe': float(strategy_sharpe),
                'label': 'Forecast Strategy'
            },
            'gross_strategy': {
                'mean': float(gross_strategy_mean),
                'std': float(gross_strategy_std),
                'sharpe': float(gross_strategy_sharpe),
                'label': 'Forecast Strategy (Gross)'
            },
            'buy_hold': {
                'mean': float(bh_mean),
                'std': float(bh_std),
                'sharpe': float(bh_sharpe),
                'label': 'Buy & Hold'
            },
            'hit_rate': float(hit_rate),
            'avg_weight': float(avg_weight),
            'std_weight': float(std_weight),
            'avg_z_score': float(avg_z),
            'std_z_score': float(std_z),
            'avg_turnover': float(avg_turnover),
            'total_turnover': float(total_turnover),
            'avg_transaction_cost': float(avg_transaction_cost),
            'total_transaction_cost': float(total_transaction_cost),
            'transaction_cost_rate': float(trading_config.transaction_cost_rate),
            'kappa': float(strategy_config.kappa),
            'min_periods': int(strategy_config.min_periods),
            'signal_lag': int(strategy_config.signal_lag),
            'standardize_signal': bool(strategy_config.standardize_signal),
            'signal_return_corr': float(signal_corr) if not np.isnan(signal_corr) else 0.0,
            'information_coefficient': float(ic) if not np.isnan(ic) else 0.0,
            'n_periods': int(len(strategy_rets)),
            'metadata': {
                'annualization_periods': int(annualization_periods),
                'trading_config': ScalingLawConfig._serialize_value(trading_config),
                'ts_strategy_config': ScalingLawConfig._serialize_value(strategy_config),
            },
        }

        del ts_df
        gc.collect()

        return ts_stats, ts_returns

    @staticmethod
    def print_portfolio_stats(portfolio_stats: Dict[str, Any]):
        """Print formatted portfolio statistics (panel mode)."""
        print(f"\n{'=' * 95}")
        print("LONG-SHORT PORTFOLIO ANALYSIS")
        print('=' * 95)
        print(f"{'Metric':<15} {'10% Breakpoint':>18} {'30% Breakpoint':>18} "
              f"{'50% Breakpoint':>18} {'Forecast Weighted':>18}")
        print('─' * 95)
        print(f"{'Ann. Mean':<15} {portfolio_stats['LS_10']['mean']:>17.4f} "
              f"{portfolio_stats['LS_30']['mean']:>17.4f} "
              f"{portfolio_stats['LS_50']['mean']:>17.4f} "
              f"{portfolio_stats['Forecast_Weighted']['mean']:>17.4f}")
        print(f"{'Ann. Std Dev':<15} {portfolio_stats['LS_10']['std']:>17.4f} "
              f"{portfolio_stats['LS_30']['std']:>17.4f} "
              f"{portfolio_stats['LS_50']['std']:>17.4f} "
              f"{portfolio_stats['Forecast_Weighted']['std']:>17.4f}")
        print(f"{'Ann. Sharpe':<15} {portfolio_stats['LS_10']['sharpe']:>17.4f} "
              f"{portfolio_stats['LS_30']['sharpe']:>17.4f} "
              f"{portfolio_stats['LS_50']['sharpe']:>17.4f} "
              f"{portfolio_stats['Forecast_Weighted']['sharpe']:>17.4f}")
        print('=' * 95)

    @staticmethod
    def print_ts_portfolio_stats(ts_stats: Dict[str, Any]):
        """Print formatted portfolio statistics (time-series mode)."""
        print(f"\n{'=' * 70}")
        print("TIME-SERIES STRATEGY ANALYSIS")
        print('=' * 70)
        print(f"{'Metric':<20} {'Forecast Strategy':>22} {'Buy & Hold':>22}")
        print('─' * 70)
        print(f"{'Ann. Mean':<20} {ts_stats['strategy']['mean']:>21.4f} "
              f"{ts_stats['buy_hold']['mean']:>21.4f}")
        print(f"{'Ann. Std Dev':<20} {ts_stats['strategy']['std']:>21.4f} "
              f"{ts_stats['buy_hold']['std']:>21.4f}")
        print(f"{'Ann. Sharpe':<20} {ts_stats['strategy']['sharpe']:>21.4f} "
              f"{ts_stats['buy_hold']['sharpe']:>21.4f}")
        print('─' * 70)
        print(f"{'Hit Rate':<20} {ts_stats['hit_rate']:>21.4f}")
        print(f"{'Avg Weight':<20} {ts_stats['avg_weight']:>21.4f}")
        print(f"{'N Periods':<20} {ts_stats['n_periods']:>21d}")
        print('=' * 70)


# ============================================================================
# SCALING LAW PLOTTING
# ============================================================================
class ScalingLawPlotter:
    """Plotter class for creating scaling law visualizations."""

    def __init__(
            self,
            results_path: Union[str, Path],
            artifacts: Optional[Union[ArtifactNames, Dict[str, Any]]] = None
    ):
        """
        Initialize the plotter.

        Args:
            results_path: Path to the results directory
            artifacts: Optional artifact-name configuration
        """
        self.results_path = Path(results_path)
        if artifacts is None:
            self.artifacts = ArtifactNames()
        elif isinstance(artifacts, ArtifactNames):
            self.artifacts = artifacts
        elif isinstance(artifacts, dict):
            self.artifacts = ArtifactNames(**artifacts)
        else:
            raise TypeError(
                f"artifacts must be ArtifactNames, a dict, or None; got {type(artifacts).__name__}"
            )

    def _artifact_path(self, artifact_field: str) -> Path:
        path = Path(getattr(self.artifacts, artifact_field))
        if path.is_absolute():
            return path
        return self.results_path / path

    def _load_results(self) -> List[Dict[str, Any]]:
        """Load results from pickle or JSON file. Convert MSE to RMSE*100 and R² to percentage."""
        try:
            with open(self._artifact_path("results_pickle"), 'rb') as f:
                results = pickle.load(f)
        except Exception:
            with open(self._artifact_path("results_json"), 'r') as f:
                results = json.load(f)

        # Convert MSE to RMSE*100 for all loss metrics
        # Convert R² to percentage
        for r in results:
            # Convert top-level loss metrics
            if 'test_loss' in r:
                original = r['test_loss']
                r['test_loss'] = np.sqrt(r['test_loss']) * 100

            # Convert R² to percentage
            if 'test_r2' in r:
                r['test_r2'] = r['test_r2'] * 100
            if 'val_r2' in r:
                r['val_r2'] = r['val_r2'] * 100

            # Convert training curve losses
            if 'training_curve' in r:
                curve = r['training_curve']
                if 'train_loss' in curve:
                    curve['train_loss'] = [np.sqrt(v) * 100 for v in curve['train_loss']]
                if 'val_loss' in curve:
                    original_final = curve['val_loss'][-1]
                    curve['val_loss'] = [np.sqrt(v) * 100 for v in curve['val_loss']]

        return results

    def _fit_scaling_law_varpro(
            self,
            x_data: np.ndarray,
            y_data: np.ndarray,
            is_sharpe: bool = False,
            use_compute_weighting: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Fit scaling law using Variable Projection (Separable Non-Linear Least Squares).

        For LOSS: Uses LOG-SCALE residuals to handle data spanning multiple orders of magnitude.
        For SHARPE: Uses linear residuals (since Sharpe can be negative).
        """
        if len(x_data) <= 3:
            return None

        try:
            scale_factor = np.median(x_data)
            x_scaled = x_data / scale_factor

            if use_compute_weighting:
                log_x = np.log(x_data)
                log_x_shifted = log_x - log_x.min() + 1
                weights = log_x_shifted / np.sum(log_x_shifted)
            else:
                weights = np.ones_like(x_data) / len(x_data)

            if is_sharpe:
                # Original linear-residual approach for Sharpe (can be negative)
                sqrt_w = np.sqrt(weights)
                col_ones = np.ones_like(x_scaled)

                def get_linear_params(b):
                    try:
                        with np.errstate(over='raise', invalid='raise'):
                            col_power = np.power(x_scaled, b)
                    except FloatingPointError:
                        return None, np.inf

                    A = np.column_stack([col_ones, col_power])
                    A_w = A * sqrt_w[:, np.newaxis]
                    y_w = y_data * sqrt_w

                    theta, residuals, rank, s = np.linalg.lstsq(A_w, y_w, rcond=None)

                    if residuals.size == 0:
                        y_pred = A @ theta
                        resid_sum = np.sum(weights * (y_data - y_pred) ** 2)
                    else:
                        resid_sum = residuals[0]

                    return theta, resid_sum

                def objective(b):
                    _, rss = get_linear_params(b)
                    return rss

                res = minimize_scalar(
                    objective,
                    bounds=(-10.0, -0.001),
                    method='bounded'
                )

                if not res.success:
                    return None

                best_b = res.x
                theta, rss = get_linear_params(best_b)
                if theta is None:
                    return None

                intercept_opt, slope_opt = theta
                real_slope = slope_opt * np.power(scale_factor, -best_b)

                L_inf = intercept_opt
                a = -real_slope
                b = best_b

            else:
                # LOG-SCALE residuals for Loss - fixes the asymptote issue
                def objective(b):
                    col_power = np.power(x_scaled, b)

                    def inner_objective(params):
                        L_inf, a_scaled = params
                        if L_inf <= 0 or a_scaled <= 0:
                            return 1e10

                        y_pred = L_inf + a_scaled * col_power
                        if np.any(y_pred <= 0):
                            return 1e10

                        # KEY FIX: Minimize log residuals (relative error)
                        log_residuals = np.log(y_data) - np.log(y_pred)
                        return np.sum(weights * log_residuals ** 2)

                    # Initial guess from OLS
                    A = np.column_stack([np.ones_like(x_scaled), col_power])
                    theta_init = np.linalg.lstsq(A, y_data, rcond=None)[0]

                    # Ensure valid initial guess
                    if theta_init[0] <= 0:
                        theta_init[0] = np.min(y_data) * 0.9
                    if theta_init[1] <= 0:
                        theta_init[1] = 1.0

                    # Optimize inner problem
                    res_inner = minimize(inner_objective, theta_init, method='L-BFGS-B',
                                         bounds=[(1e-10, np.min(y_data)), (1e-50, 1e50)])

                    return res_inner.fun

                res = minimize_scalar(
                    objective,
                    bounds=(-10.0, -0.001),
                    method='bounded'
                )

                if not res.success:
                    return None

                best_b = res.x

                # Recover final parameters
                col_power = np.power(x_scaled, best_b)

                def inner_objective(params):
                    L_inf, a_scaled = params
                    if L_inf <= 0 or a_scaled <= 0:
                        return 1e10
                    y_pred = L_inf + a_scaled * col_power
                    if np.any(y_pred <= 0):
                        return 1e10
                    log_residuals = np.log(y_data) - np.log(y_pred)
                    return np.sum(weights * log_residuals ** 2)

                A = np.column_stack([np.ones_like(x_scaled), col_power])
                theta_init = np.linalg.lstsq(A, y_data, rcond=None)[0]
                if theta_init[0] <= 0:
                    theta_init[0] = np.min(y_data) * 0.9
                if theta_init[1] <= 0:
                    theta_init[1] = 1.0

                res_inner = minimize(inner_objective, theta_init, method='L-BFGS-B',
                                     bounds=[(1e-10, np.min(y_data)), (1e-50, 1e50)])

                L_inf = res_inner.x[0]
                a = res_inner.x[1] * np.power(scale_factor, -best_b)
                b = best_b

            try:
                C0 = np.power(1.0 / abs(a), 1.0 / b)
            except (ValueError, ZeroDivisionError, FloatingPointError):
                return None

            if not np.isfinite(C0) or C0 <= 0:
                return None

            exponent = int(np.floor(np.log10(abs(C0))))
            mantissa = C0 / (10 ** exponent)

            if is_sharpe:
                y_pred = L_inf - a * np.power(x_data, b)
            else:
                y_pred = L_inf + a * np.power(x_data, b)

            ss_res = np.sum((y_data - y_pred) ** 2)
            ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)

            return {
                'L_inf': L_inf,
                'C0': C0,
                'mantissa': mantissa,
                'exponent': exponent,
                'b': b,
                'r_squared': r_squared,
                'a': a
            }

        except Exception as e:
            print(f"  Fitting failed: {e}")
            return None

    def plot_scaling_curves(
            self,
            loss_type: str = 'val_loss',
            x_axis: str = 'compute',
            figsize: Tuple[int, int] = (14, 9),
            title: Optional[str] = None,
            save_name: str = 'scaling_curves.png',
            dpi: int = 300,
            show_final_points: bool = True,
            fit_scaling_law: bool = True,
            use_compute_weighting: bool = False
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Create scaling law plot showing training curves for each model."""
        results = self._load_results()
        print(f"\nLoaded {len(results)} models")

        fig, ax = plt.subplots(figsize=figsize, facecolor='white')

        all_params = [r['actual_params'] for r in results]
        norm = plt.matplotlib.colors.LogNorm(vmin=min(all_params), vmax=max(all_params))
        cmap = plt.cm.viridis

        final_x_values = []
        final_y_values = []

        for result in results:
            params = result['actual_params']
            curve = result['training_curve']

            if x_axis == 'compute':
                x_data = curve['cumulative_pf_days']
                x_label = 'Compute (PetaFLOP-days)'
            else:
                x_data = curve['epochs']
                x_label = 'Epoch'

            y_data = curve[loss_type]
            color = cmap(norm(params))

            ax.plot(x_data, y_data, color=color, linewidth=3.5, alpha=0.8)

            if show_final_points:
                ax.scatter(x_data[-1], y_data[-1], color=color, s=100,
                           edgecolors='black', linewidth=1.5, zorder=5)

            final_x_values.append(x_data[-1])
            final_y_values.append(y_data[-1])

        if fit_scaling_law and len(final_x_values) > 3:
            self._fit_and_plot_scaling_law(
                ax, np.array(final_x_values), np.array(final_y_values),
                x_var='C' if x_axis == 'compute' else 'E',
                use_compute_weighting=use_compute_weighting
            )

        ax.set_xscale('log')
        ax.set_yscale('log')
        # Keep a small positive floor on the log-loss axis without clipping
        # RMSE values that were rescaled from MSE into percentage units.
        ax.set_ylim(bottom=1e-1)
        ax.set_xlabel(x_label, fontsize=16, fontweight='bold')

        loss_names = {'train_loss': 'Training Loss', 'val_loss': 'Validation Loss'}
        ax.set_ylabel(loss_names[loss_type], fontsize=16, fontweight='bold')

        if title is None:
            title = f'Neural Network Scaling Curves - {loss_names[loss_type]}'
        ax.set_title(title, fontsize=18, fontweight='bold', pad=20)

        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(labelsize=12)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label('Parameters', rotation=270, labelpad=25, fontsize=14, fontweight='bold')

        if fit_scaling_law:
            ax.legend(fontsize=13, loc='upper right', framealpha=0.9)

        plt.tight_layout()

        save_path = self.results_path / save_name
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"✓ Plot saved to: {save_path}")
        plt.show()

        return fig, ax

    def plot_final_performance(
            self,
            metric: str = 'test_loss',
            x_axis: str = 'compute',
            fit_curve: bool = True,
            figsize: Tuple[int, int] = (12, 8),
            title: Optional[str] = None,
            save_name: str = 'test_performance.png',
            dpi: int = 300,
            use_compute_weighting: bool = False
    ) -> Tuple[Optional[plt.Figure], Optional[plt.Axes]]:
        """Plot final test performance vs model size/compute."""
        results = self._load_results()
        print(f"\nLoaded {len(results)} models")

        # Filter to results that contain the requested metric
        results = [r for r in results if metric in r]
        if len(results) == 0:
            print(f"✗ No results contain metric '{metric}'")
            return None, None

        params = np.array([r['actual_params'] for r in results])
        metric_values = np.array([r[metric] for r in results])

        if x_axis == 'compute':
            x_data = np.array([r['pf_days'] for r in results])
            x_label = 'Compute (PetaFLOP-days)'
            x_var = 'C'
        else:
            x_data = params
            x_label = 'Parameters'
            x_var = 'N'

        fig, ax = plt.subplots(figsize=figsize, facecolor='white')

        scatter = ax.scatter(x_data, metric_values, c=params, cmap='viridis',
                             s=200, alpha=0.7, edgecolors='black', linewidth=1.5,
                             norm=plt.matplotlib.colors.LogNorm())

        cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
        cbar.set_label('Parameters', rotation=270, labelpad=25, fontsize=14, fontweight='bold')

        if fit_curve and len(x_data) > 3 and metric == 'test_loss':
            self._fit_and_plot_scaling_law(
                ax, x_data, metric_values, x_var=x_var,
                use_compute_weighting=use_compute_weighting
            )
        elif fit_curve and len(x_data) > 3 and metric in ('test_r2', 'val_r2'):
            self._fit_and_plot_scaling_law(
                ax, x_data, metric_values, x_var=x_var,
                use_compute_weighting=use_compute_weighting,
                increasing=True, label_symbol='R^2(c)'
            )

        ax.set_xscale('log')
        if metric == 'test_loss':
            ax.set_yscale('log')

        ax.set_xlabel(x_label, fontsize=16, fontweight='bold')

        metric_labels = {'test_loss': 'Test Loss (RMSE in Percent)', 'test_r2': 'Test R² (%)', 'val_r2': 'Validation R² (%)'}
        ax.set_ylabel(metric_labels[metric], fontsize=16, fontweight='bold')

        if title is None:
            title = f'{metric_labels[metric]} vs Model Size'
        ax.set_title(title, fontsize=18, fontweight='bold', pad=20)

        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(labelsize=12)

        if fit_curve:
            ax.legend(fontsize=13, loc='best', framealpha=0.9)

        plt.tight_layout()

        save_path = self.results_path / save_name
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"✓ Plot saved to: {save_path}")
        plt.show()

        return fig, ax

    def plot_sharpe_ratio_scaling(
            self,
            breakpoint: str = '50',
            x_axis: str = 'compute',
            fit_curve: bool = True,
            figsize: Tuple[int, int] = (12, 8),
            title: Optional[str] = None,
            save_name: Optional[str] = None,
            dpi: int = 300,
            use_compute_weighting: bool = False
    ) -> Tuple[Optional[plt.Figure], Optional[plt.Axes]]:
        """Plot Sharpe ratio vs model size/compute."""
        results = self._load_results()
        print(f"\nLoaded {len(results)} models")

        breakpoint_key = f'LS_{breakpoint}'
        breakpoint_label = f'{breakpoint}% Breakpoint'

        params_list, sharpe_list, compute_list = [], [], []

        for r in results:
            if 'portfolio_stats' in r and breakpoint_key in r['portfolio_stats']:
                params_list.append(r['actual_params'])
                sharpe_list.append(r['portfolio_stats'][breakpoint_key]['sharpe'])
                compute_list.append(r['pf_days'])

        if len(params_list) == 0:
            print("✗ No portfolio statistics found!")
            return None, None

        params = np.array(params_list)
        sharpe_values = np.array(sharpe_list)

        if x_axis == 'compute':
            x_data = np.array(compute_list)
            x_label = 'Compute (PetaFLOP-days)'
            x_var = 'C'
        else:
            x_data = params
            x_label = 'Parameters'
            x_var = 'N'

        fig, ax = plt.subplots(figsize=figsize, facecolor='white')

        scatter = ax.scatter(x_data, sharpe_values, c=params, cmap='viridis',
                             s=200, alpha=0.7, edgecolors='black', linewidth=1.5,
                             norm=plt.matplotlib.colors.LogNorm())

        cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
        cbar.set_label('Parameters', rotation=270, labelpad=25, fontsize=14, fontweight='bold')

        if fit_curve and len(x_data) > 3:
            self._fit_and_plot_scaling_law(
                ax, x_data, sharpe_values, x_var=x_var,
                use_compute_weighting=use_compute_weighting,
                increasing=True, label_symbol='SR(c)'
            )

        ax.set_xscale('log')
        ax.set_xlabel(x_label, fontsize=16, fontweight='bold')
        ax.set_ylabel('Annualized Sharpe Ratio', fontsize=16, fontweight='bold')

        if title is None:
            title = f'Long-Short Portfolio Sharpe Ratio ({breakpoint_label})'
        ax.set_title(title, fontsize=18, fontweight='bold', pad=20)

        ax.grid(True, alpha=0.3, linestyle='--')
        ax.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        ax.tick_params(labelsize=12)

        if fit_curve:
            ax.legend(fontsize=13, loc='best', framealpha=0.9)

        plt.tight_layout()

        if save_name is None:
            save_name = f'sharpe_ratio_LS{breakpoint}_vs_{x_axis}.png'
        save_path = self.results_path / save_name
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"✓ Plot saved to: {save_path}")
        plt.show()

        return fig, ax

    def create_all_plots(self, dpi: int = 300, use_compute_weighting: bool = False):
        """Create comprehensive set of scaling law plots."""
        print("\n" + "=" * 80)
        print("CREATING SCALING LAW VISUALIZATIONS")
        print("=" * 80)

        print("\n1. Validation Loss Training Curves vs Compute")
        self.plot_scaling_curves(
            loss_type='val_loss',
            x_axis='compute',
            save_name='scaling_curves_val_compute.png',
            dpi=dpi,
            use_compute_weighting=use_compute_weighting
        )

        print("\n2. Test Loss vs Compute")
        self.plot_final_performance(
            metric='test_loss',
            x_axis='compute',
            save_name='test_loss_vs_compute.png',
            dpi=dpi,
            use_compute_weighting=use_compute_weighting
        )

        print("\n3. Validation R² vs Compute")
        self.plot_final_performance(
            metric='val_r2',
            x_axis='compute',
            save_name='val_r2_vs_compute.png',
            dpi=dpi,
            use_compute_weighting=use_compute_weighting
        )

        print("\n4. Test R² vs Compute")
        self.plot_final_performance(
            metric='test_r2',
            x_axis='compute',
            save_name='test_r2_vs_compute.png',
            dpi=dpi,
            use_compute_weighting=use_compute_weighting
        )

        print("\n5. Sharpe Ratio LS50 vs Compute")
        self.plot_sharpe_ratio_scaling(
            breakpoint='50',
            x_axis='compute',
            dpi=dpi,
            use_compute_weighting=use_compute_weighting
        )

        print("\n6. Sharpe Ratio LS30 vs Compute")
        self.plot_sharpe_ratio_scaling(
            breakpoint='30',
            x_axis='compute',
            dpi=dpi,
            use_compute_weighting=use_compute_weighting
        )

        print("\n7. Sharpe Ratio LS10 vs Compute")
        self.plot_sharpe_ratio_scaling(
            breakpoint='10',
            x_axis='compute',
            dpi=dpi,
            use_compute_weighting=use_compute_weighting
        )

        print("\n" + "=" * 80)
        print("ALL PLOTS CREATED SUCCESSFULLY")
        print("=" * 80)

    def _fit_and_plot_scaling_law(
            self,
            ax: plt.Axes,
            x_data: np.ndarray,
            y_data: np.ndarray,
            x_var: str = 'C',
            use_compute_weighting: bool = False,
            increasing: bool = False,
            label_symbol: str = 'L(c)'
    ):
        """
        Fit and plot a scaling law curve using Variable Projection.

        Args:
            ax: Matplotlib axes to plot on
            x_data: X-axis values (compute or parameters)
            y_data: Y-axis metric values
            x_var: Variable name for x-axis in LaTeX equation ('C' or 'N')
            use_compute_weighting: Weight larger-compute models more heavily
            increasing: If False, fits Y = Y_inf + a*(C/C0)^b (decreasing, for loss).
                        If True, fits Y = Y_inf - a*(C/C0)^b (increasing, for Sharpe/R²).
            label_symbol: Symbol in the plot legend equation (e.g. 'L', 'SR', 'R^2')
        """
        fit_result = self._fit_scaling_law_varpro(
            x_data, y_data,
            is_sharpe=increasing,
            use_compute_weighting=use_compute_weighting
        )

        if fit_result is not None:
            Y_inf = fit_result['L_inf']
            a = fit_result['a']
            b = fit_result['b']
            mantissa = fit_result['mantissa']
            exponent = fit_result['exponent']
            r_squared = fit_result['r_squared']

            x_smooth = np.logspace(np.log10(x_data.min()),
                                   np.log10(x_data.max()), 100)

            sign_str = '-' if increasing else '+'
            if increasing:
                y_smooth = Y_inf - a * np.power(x_smooth, b)
            else:
                y_smooth = Y_inf + a * np.power(x_smooth, b)

            label = (
                f'${label_symbol} = {Y_inf:.3f} {sign_str} '
                f'\\left(\\frac{{{x_var}}}{{{mantissa:.1f} \\times 10^{{{exponent}}}}}\\right)^{{{b:.2f}}}$, '
                f'$R^2 = {r_squared * 100:.1f}\\%$')
            ax.plot(x_smooth, y_smooth, 'r--', linewidth=3, alpha=0.9, label=label, zorder=10)

            latex_eq = (
                f"${Y_inf:.4f} {sign_str} "
                f"\\left(\\frac{{{x_var}}}{{{mantissa:.1f} \\times 10^{{{exponent}}}}}\\right)^{{{b:.4f}}}$")
            print(f"✓ Scaling law fit (R²={r_squared:.4f}):")
            print(f"  {latex_eq}")
        else:
            print("✗ Could not fit scaling law")

    def plot_ts_sharpe_ratio_scaling(
            self,
            x_axis: str = 'compute',
            fit_curve: bool = True,
            figsize: Tuple[int, int] = (12, 8),
            title: Optional[str] = None,
            save_name: Optional[str] = None,
            dpi: int = 300,
            use_compute_weighting: bool = False
    ) -> Tuple[Optional[plt.Figure], Optional[plt.Axes]]:
        """
        Plot time-series strategy Sharpe ratio vs model size/compute.

        Args:
            x_axis: X-axis variable ('compute' or 'params')
            fit_curve: Whether to fit a scaling law curve
            figsize: Figure size
            title: Plot title
            save_name: Filename for saving
            dpi: Resolution
            use_compute_weighting: Whether to weight by compute during fitting

        Returns:
            Tuple of (figure, axes) or (None, None) if no data
        """
        results = self._load_results()
        print(f"\nLoaded {len(results)} models")

        params_list, strategy_sharpe_list, bh_sharpe_list, compute_list = [], [], [], []

        for r in results:
            if 'portfolio_stats' in r and 'strategy' in r['portfolio_stats']:
                params_list.append(r['actual_params'])
                strategy_sharpe_list.append(r['portfolio_stats']['strategy']['sharpe'])
                bh_sharpe_list.append(r['portfolio_stats']['buy_hold']['sharpe'])
                compute_list.append(r['pf_days'])

        if len(params_list) == 0:
            print("✗ No time-series portfolio statistics found!")
            return None, None

        params = np.array(params_list)
        strategy_sharpe = np.array(strategy_sharpe_list)
        bh_sharpe = np.array(bh_sharpe_list)

        if x_axis == 'compute':
            x_data = np.array(compute_list)
            x_label = 'Compute (PetaFLOP-days)'
            x_var = 'C'
        else:
            x_data = params
            x_label = 'Parameters'
            x_var = 'N'

        fig, ax = plt.subplots(figsize=figsize, facecolor='white')

        # Plot strategy Sharpe ratios
        scatter = ax.scatter(x_data, strategy_sharpe, c=params, cmap='viridis',
                             s=200, alpha=0.7, edgecolors='black', linewidth=1.5,
                             norm=plt.matplotlib.colors.LogNorm())

        cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
        cbar.ax.tick_params(labelsize=16)
        cbar.set_label('Parameters', rotation=270, labelpad=25, fontsize=14, fontweight='bold')

        # Plot buy-and-hold benchmark as horizontal line
        bh_mean = np.mean(bh_sharpe)
        ax.axhline(y=bh_mean, color='black', linestyle='-', linewidth=2, alpha=0.5,
                   label=f'Buy & Hold (SR={bh_mean:.3f})')

        if fit_curve and len(x_data) > 3:
            self._fit_and_plot_scaling_law(
                ax, x_data, strategy_sharpe, x_var=x_var,
                use_compute_weighting=use_compute_weighting,
                increasing=True, label_symbol='SR(c)'
            )

        ax.set_xscale('log')
        ax.set_xlabel(x_label, fontsize=16, fontweight='bold')
        ax.set_ylabel('Annualized Sharpe Ratio', fontsize=16, fontweight='bold')

        if title is None:
            title = 'Time-Series Strategy Sharpe Ratio vs Compute'
        ax.set_title(title, fontsize=18, fontweight='bold', pad=20)

        ax.grid(True, alpha=0.3, linestyle='--')
        ax.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        ax.tick_params(labelsize=16)

        ax.legend(fontsize=16, loc='best', framealpha=0.9)

        plt.tight_layout()

        if save_name is None:
            save_name = f'sharpe_ratio_ts_vs_{x_axis}.png'
        save_path = self.results_path / save_name
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"✓ Plot saved to: {save_path}")
        plt.show()

        return fig, ax


# ============================================================================
# MAIN SCALING LAW EXPERIMENT CLASS
# ============================================================================

class ScalingLawExperiment:
    """
    Main class for running neural network scaling law experiments.

    This class orchestrates the entire scaling law experiment, including
    data preparation, model training, result saving, and visualization.

    Example:
        >>> config = ScalingLawConfig(
        ...     param_sizes=['1K', '10K', '100K'],
        ...     epochs=500,
        ...     output_dir='./output/'
        ... )
        >>> experiment = ScalingLawExperiment(config)
        >>> results = experiment.run(X_train, y_train, X_val, y_val, X_test, y_test)
    """

    def __init__(self, config: Optional[ScalingLawConfig] = None):
        """
        Initialize the scaling law experiment.

        Args:
            config: ScalingLawConfig instance (uses defaults if None)
        """
        self.config = config or ScalingLawConfig()
        self.model_builder = ModelBuilder(self.config)
        self.results_manager = ResultsManager(self.config.output_dir, self.config)
        self.data_splitter = DataSplitter(self.config)
        self._last_split_result: Optional[DataSplitResult] = None

        # Configure TensorFlow
        self._configure_tensorflow()

    def _configure_tensorflow(self):
        """Configure TensorFlow settings based on configuration."""
        # GPU configuration
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✓ Found {len(gpus)} GPU(s) - Memory growth enabled")
            except RuntimeError as e:
                print(f"GPU configuration error: {e}")
        else:
            print("⚠ No GPU found - using CPU only")

        # Determinism
        if self.config.enable_determinism:
            os.environ['TF_DETERMINISTIC_OPS'] = '1'
            os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
            os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
            try:
                tf.config.experimental.enable_op_determinism()
                print("✓ Deterministic operations enabled")
            except Exception as e:
                print(f"⚠ Could not enable determinism: {e}")

        # Mixed precision
        from tensorflow.keras import mixed_precision
        precision_policy = self.config.compute.resolve_mixed_precision_policy()
        try:
            mixed_precision.set_global_policy(precision_policy)
        except ValueError as exc:
            if self.config.precision == 8 and not self.config.mixed_precision_policy:
                raise ValueError(
                    "precision=8 maps to the Keras 'mixed_float8' policy, but this "
                    "TensorFlow/Keras runtime does not appear to support float8 "
                    "global policies. Use precision=16, 32, or 64, or upgrade to a "
                    "runtime with float8 policy support."
                ) from exc
            raise

        active_policy = mixed_precision.global_policy()
        precision_label = (
            "policy override"
            if self.config.mixed_precision_policy
            else f"{self.config.precision}-bit"
        )
        print(
            f"✓ Precision: {precision_label} "
            f"(policy={active_policy.name}, compute dtype={active_policy.compute_dtype})"
        )

    @staticmethod
    def parse_size(s: Union[str, int]) -> int:
        """Convert size string to integer."""
        if isinstance(s, int):
            return s
        s = str(s).upper()
        if 'B' in s:
            return int(float(s.replace('B', '')) * 1_000_000_000)
        if 'M' in s:
            return int(float(s.replace('M', '')) * 1_000_000)
        elif 'K' in s:
            return int(float(s.replace('K', '')) * 1_000)
        return int(s)

    @staticmethod
    def format_params(n: int) -> str:
        """Format parameter count as human-readable string."""
        if n < 1_000:
            return str(n)
        if n < 1_000_000:
            k = n / 1_000
            return f"{k:g}K"
        if n < 1_000_000_000:
            m = n / 1_000_000
            return f"{m:g}M"
        b = n / 1_000_000_000
        return f"{b:g}B"
    @staticmethod
    def make_param_sizes(
            jump: int = 100,
            min_size: int = 0,
            max_size: int = 1_000_000_000
    ) -> List[str]:
        """Generate a list of parameter sizes with geometric spacing."""
        sizes = []

        # 1 - 1K: steps of 25
        if min_size < 1_000:
            sizes += list(range(max(min_size, 0), min(1_000, max_size) + 1, jump))

        # 1K - 10K: steps of jump*10
        if max_size >= 1_000:
            sizes += list(range(1_000, min(10_000, max_size) + 1, jump*10))

        # 10K - 100K: steps of jump * 100
        if max_size >= 10_000:
            sizes += list(range(10_000, min(100_000, max_size) + 1, jump * 100))

        # 100K - 1M: steps of jump * 1000
        if max_size >= 100_000:
            sizes += list(range(100_000, min(1_000_000, max_size) + 1, jump * 1_000))

        # 1M - 10M: steps of jump * 10000
        if max_size >= 1_000_000:
            sizes += list(range(1_000_000, min(10_000_000, max_size) + 1, jump * 10_000))

        # 10M - 100M: steps of jump * 100000
        if max_size >= 10_000_000:
            sizes += list(range(10_000_000, min(100_000_000, max_size) + 1, jump * 100_000))

        # 100M - 1B: steps of jump * 1000000
        if max_size >= 100_000_000:
            sizes += list(range(100_000_000, min(1_000_000_000, max_size) + 1, jump * 1_000_000))

        sizes = [s for s in sizes if min_size <= s <= max_size]
        sizes = list(dict.fromkeys(sizes))
        return [ScalingLawExperiment.format_params(n) for n in sizes]

    @staticmethod
    def _solve_ar1_normal_equations(
            N: int,
            s_x: float,
            s_y: float,
            s_xx: float,
            s_xy: float
    ) -> Tuple[float, float]:
        """
        Solve the 2×2 OLS normal equations for an AR(1) model.

        Given sufficient statistics accumulated over N lag-response pairs
        (x_t, y_t) where x_t = y_{t-1}, returns OLS estimates (c, phi) of
        the model  y_t = c + phi * x_t.

        Suitable for O(1)-per-step incremental updates: the caller maintains
        the five sufficient statistics and calls this function after each new
        observation, avoiding an O(n²) full-refit loop.

        Returns:
            (c, phi): intercept and AR coefficient
        """
        denom = N * s_xx - s_x * s_x
        if N == 0 or abs(denom) < 1e-12:
            return (s_y / N if N > 0 else 0.0), 0.0
        phi = (N * s_xy - s_x * s_y) / denom
        c   = (s_y - phi * s_x) / N
        return float(c), float(phi)

    @staticmethod
    def _fit_ar1_ols(y: np.ndarray) -> Tuple[float, float]:
        """
        Fit an AR(1) model by OLS on the array y.

        Estimates  y[t] = c + phi * y[t-1]  over all consecutive pairs in y.
        For panel data y is the chronologically-sorted target vector, so the
        fit reflects the aggregate serial dependence in the sequence (including
        cross-sectional adjacency within each period).  The result serves as a
        fixed-coefficient benchmark for OOS R² calculations.

        Returns:
            (c, phi): intercept and AR coefficient
        """
        n = len(y)
        if n < 2:
            return float(np.mean(y)) if n == 1 else 0.0, 0.0
        N    = n - 1
        s_x  = float(np.sum(y[:-1]))
        s_y  = float(np.sum(y[1:]))
        s_xx = float(np.dot(y[:-1], y[:-1]))
        s_xy = float(np.dot(y[:-1], y[1:]))
        return ScalingLawExperiment._solve_ar1_normal_equations(N, s_x, s_y, s_xx, s_xy)

    @staticmethod
    def _benchmark_display_name(benchmark: Union[BenchmarkConfig, str, Callable[..., Any]]) -> str:
        benchmark_mode = benchmark.mode if isinstance(benchmark, BenchmarkConfig) else benchmark
        return ScalingLawConfig._serialize_value(benchmark_mode)

    @staticmethod
    def _benchmark_predictions_for_split(
            benchmark_mode: str,
            history: np.ndarray,
            target: np.ndarray
    ) -> np.ndarray:
        """Resolve a built-in benchmark series using the historical legacy formulas."""
        history = np.asarray(history, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)

        if benchmark_mode == "historical_mean_updating":
            predictions = np.zeros(len(target), dtype=np.float64)
            cumsum = np.sum(history)
            count = len(history)
            for i in range(len(target)):
                predictions[i] = cumsum / count
                cumsum += target[i]
                count += 1
            return predictions

        if benchmark_mode == "ar1":
            c_ar1, phi_ar1 = ScalingLawExperiment._fit_ar1_ols(history)
            predictions = np.zeros(len(target), dtype=np.float64)
            if len(target) == 0:
                return predictions
            predictions[0] = c_ar1 + phi_ar1 * history[-1]
            for i in range(1, len(target)):
                predictions[i] = c_ar1 + phi_ar1 * target[i - 1]
            return predictions

        if benchmark_mode == "ar1_updating":
            predictions = np.zeros(len(target), dtype=np.float64)
            _N = len(history) - 1
            _sx = float(np.sum(history[:-1]))
            _sy = float(np.sum(history[1:]))
            _sxx = float(np.dot(history[:-1], history[:-1]))
            _sxy = float(np.dot(history[:-1], history[1:]))
            c_ar1, phi_ar1 = ScalingLawExperiment._solve_ar1_normal_equations(
                _N, _sx, _sy, _sxx, _sxy
            )
            y_prev = history[-1]
            for i in range(len(target)):
                predictions[i] = c_ar1 + phi_ar1 * y_prev
                _sx += y_prev
                _sy += target[i]
                _sxx += y_prev * y_prev
                _sxy += y_prev * target[i]
                _N += 1
                c_ar1, phi_ar1 = ScalingLawExperiment._solve_ar1_normal_equations(
                    _N, _sx, _sy, _sxx, _sxy
                )
                y_prev = target[i]
            return predictions

        return np.full(len(target), np.mean(history), dtype=np.float64)

    @staticmethod
    def _coerce_benchmark_array(
            value: Any,
            expected_len: int,
            label: str
    ) -> np.ndarray:
        predictions = np.asarray(value, dtype=np.float64).reshape(-1)
        if len(predictions) != expected_len:
            raise ValueError(
                f"Callable benchmark returned {len(predictions):,} {label} predictions; "
                f"expected {expected_len:,}"
            )
        return predictions

    @staticmethod
    def _first_present(mapping: Dict[str, Any], keys: Tuple[str, ...]) -> Any:
        for key in keys:
            if key in mapping:
                return mapping[key]
        return None

    @staticmethod
    def _parse_combined_benchmark_result(
            result: Any,
            val_len: int,
            test_len: int
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if isinstance(result, dict):
            val_value = ScalingLawExperiment._first_present(
                result,
                ("val", "validation", "val_pred", "validation_pred", "val_predictions", "validation_predictions")
            )
            test_value = ScalingLawExperiment._first_present(
                result,
                ("test", "test_pred", "test_predictions")
            )
            if val_value is not None and test_value is not None:
                return (
                    ScalingLawExperiment._coerce_benchmark_array(val_value, val_len, "validation"),
                    ScalingLawExperiment._coerce_benchmark_array(test_value, test_len, "test"),
                )
            return None

        if isinstance(result, (list, tuple)) and len(result) == 2:
            return (
                ScalingLawExperiment._coerce_benchmark_array(result[0], val_len, "validation"),
                ScalingLawExperiment._coerce_benchmark_array(result[1], test_len, "test"),
            )

        return None

    @staticmethod
    def _call_custom_benchmark_for_split(
            benchmark_callable: Callable[..., Any],
            split: str,
            history: np.ndarray,
            target: np.ndarray,
            y_train: np.ndarray,
            y_val: np.ndarray,
            y_test: np.ndarray
    ) -> np.ndarray:
        call_attempts = (
            (
                (),
                {
                    "split": split,
                    "history": history.copy(),
                    "target": target.copy(),
                    "y_train": y_train.copy(),
                    "y_val": y_val.copy(),
                    "y_test": y_test.copy(),
                },
            ),
            ((split, history.copy(), target.copy()), {}),
            ((history.copy(), target.copy()), {}),
        )

        last_error = None
        for args, kwargs in call_attempts:
            try:
                result = benchmark_callable(*args, **kwargs)
                if isinstance(result, dict):
                    split_value = ScalingLawExperiment._first_present(
                        result,
                        (
                            split,
                            f"{split}_pred",
                            f"{split}_predictions",
                            "pred",
                            "prediction",
                            "predictions",
                        ),
                    )
                    if split_value is not None:
                        result = split_value
                return ScalingLawExperiment._coerce_benchmark_array(
                    result,
                    len(target),
                    "validation" if split == "val" else "test",
                )
            except TypeError as exc:
                last_error = exc
                continue

        raise TypeError(
            "Callable benchmark must accept either keyword context "
            "(split, history, target, y_train, y_val, y_test), "
            "(split, history, target), or (history, target)"
        ) from last_error

    @staticmethod
    def _resolve_benchmark_predictions(
            benchmark: Union[BenchmarkConfig, str, Callable[..., Any]],
            y_train: np.ndarray,
            y_val: np.ndarray,
            y_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        benchmark_config = (
            benchmark if isinstance(benchmark, BenchmarkConfig)
            else BenchmarkConfig(mode=benchmark)
        )
        benchmark_mode = benchmark_config.mode
        historical_returns = np.concatenate([y_train, y_val])

        if isinstance(benchmark_mode, str):
            val_benchmark = ScalingLawExperiment._benchmark_predictions_for_split(
                benchmark_mode, y_train, y_val
            )
            test_benchmark = ScalingLawExperiment._benchmark_predictions_for_split(
                benchmark_mode, historical_returns, y_test
            )
            return val_benchmark, test_benchmark

        combined_attempts = (
            (
                (),
                {
                    "y_train": y_train.copy(),
                    "y_val": y_val.copy(),
                    "y_test": y_test.copy(),
                    "validation_history": y_train.copy(),
                    "test_history": historical_returns.copy(),
                },
            ),
            ((y_train.copy(), y_val.copy(), y_test.copy()), {}),
        )

        for args, kwargs in combined_attempts:
            try:
                combined_result = benchmark_mode(*args, **kwargs)
                parsed = ScalingLawExperiment._parse_combined_benchmark_result(
                    combined_result,
                    len(y_val),
                    len(y_test),
                )
                if parsed is not None:
                    return parsed
            except TypeError:
                pass

        val_benchmark = ScalingLawExperiment._call_custom_benchmark_for_split(
            benchmark_mode,
            "val",
            y_train,
            y_val,
            y_train,
            y_val,
            y_test,
        )
        test_benchmark = ScalingLawExperiment._call_custom_benchmark_for_split(
            benchmark_mode,
            "test",
            historical_returns,
            y_test,
            y_train,
            y_val,
            y_test,
        )
        return val_benchmark, test_benchmark

    def train_single_model(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: np.ndarray,
            y_val: np.ndarray,
            X_test: np.ndarray,
            y_test: np.ndarray,
            target_params: int,
            test_dates: Optional[np.ndarray] = None,
            epochs: Optional[int] = None,
            model_name: str = "model",
            portfolio: str = "panel",
            kappa: Optional[float] = None,
            benchmark: Optional[Union[str, Callable[..., Any], BenchmarkConfig]] = None,
            asset_ids: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Train a single model and return comprehensive results.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            X_test: Test features
            y_test: Test targets
            target_params: Target number of parameters
            test_dates: Optional test dates for portfolio analysis
            epochs: Number of epochs (uses config default if None)
            model_name: Name identifier for the model
            portfolio: Portfolio analysis mode ('panel' or 'ts')
            kappa: Risk-scaling constant for time-series strategy (default=1.0)
            benchmark: Benchmark model used in the denominator of OOS R².
                'historical_mean'          – fixed mean of the training set (default).
                'historical_mean_updating' – expanding mean that incorporates each
                                             newly observed test return before
                                             forecasting the next period.
                'ar1'                      – AR(1) estimated once on the training
                                             (or training+validation) set; coefficients
                                             are held fixed throughout the test period.
                'ar1_updating'             – AR(1) initially estimated on training
                                             (or training+validation) data, then
                                             re-estimated on an expanding window as
                                             each test return is observed.  Coefficient
                                             updates are O(1) per step via incremental
                                             normal-equation solving.

        Returns:
            Dictionary with training results and statistics
        """
        if epochs is None:
            epochs = self.config.get_epochs(target_params)
        effective_benchmark = (
            self.config.benchmark
            if benchmark is None
            else benchmark if isinstance(benchmark, BenchmarkConfig)
            else BenchmarkConfig(mode=benchmark)
        )
        ts_strategy_config = TSStrategyConfig(**{
            config_field.name: getattr(self.config.ts_strategy, config_field.name)
            for config_field in fields(TSStrategyConfig)
        })
        if kappa is not None:
            ts_strategy_config.kappa = float(kappa)
            ts_strategy_config.__post_init__()
        annualization_periods = self.config.annualization.periods
        trading_config = self.config.trading
        effective_portfolio_config = PortfolioConfig(
            mode=portfolio,
            asset_id_col=self.config.portfolio.asset_id_col,
        )
        portfolio_mode = effective_portfolio_config.mode.value

        print(f"\n{'=' * 80}")
        print(f"MODEL: {model_name} | Target: {target_params:,} parameters")
        print('=' * 80)

        if self.config.debug_memory:
            MemoryManager.print_memory_usage("START of train_single_model")

        # Aggressive cleanup before building
        MemoryManager.aggressive_cleanup()

        if self.config.debug_memory:
            MemoryManager.print_memory_usage("AFTER initial cleanup")

        # Build model
        model, normalizer, actual_params, architecture = self.model_builder.build_model(
            X_train.shape[1], target_params
        )

        if normalizer is not None:
            normalizer.adapt(X_train)

        print(f"Architecture: {architecture}")
        print(f"Actual parameters: {actual_params:,}")
        print(f"Normalization: {self.config.normalization.value}")

        flops_per_epoch = float(self.config.compute.estimate_flops_per_epoch(
            actual_params=actual_params,
            train_samples=len(X_train),
            input_dim=X_train.shape[1],
            architecture=architecture,
            model=model
        ))

        # Compile model
        if self.config.clip_norm is not None:
            optimizer = Adam(learning_rate=self.config.learning_rate,
                             clipnorm=self.config.clip_norm)
        else:
            optimizer = Adam(learning_rate=self.config.learning_rate)

        model.compile(
            loss='mean_squared_error',
            optimizer=optimizer,
            metrics=[R2PercentMetric()]
        )

        # Setup callbacks
        callbacks = [SingleLineProgressCallback()]

        if self.config.lr_scheduler_enabled:
            patience = self.config.get_lr_scheduler_patience(epochs)
            lr_scheduler = ReduceLROnPlateau(
                monitor='val_loss',
                factor=self.config.lr_scheduler_factor,
                patience=patience,
                min_lr=self.config.lr_scheduler_min_lr,
                verbose=1
            )
            callbacks.append(lr_scheduler)

        live_plot = None
        if self.config.show_live_plots:
            live_plot = LivePlotCallback()
            callbacks.append(live_plot)

        if self.config.debug_memory:
            MemoryManager.print_memory_usage("BEFORE training")

        # Train
        start_time = time.time()
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=self.config.train_batch_size,
            validation_batch_size=self.config.validation_batch_size,
            verbose=0,
            callbacks=callbacks
        )
        train_time = time.time() - start_time

        if self.config.debug_memory:
            MemoryManager.print_memory_usage("AFTER training")

        # Compute metrics
        cumulative_flops = [(i + 1) * flops_per_epoch for i in range(epochs)]
        cumulative_pf_days = [f / 8.64e19 for f in cumulative_flops]

        train_loss = float(history.history['loss'][-1])
        val_loss = float(history.history['val_loss'][-1])
        train_loss_history = [float(x) for x in history.history['loss']]
        val_loss_history = [float(x) for x in history.history['val_loss']]

        test_pred = model.predict(
            X_test,
            batch_size=self.config.prediction_batch_size,
            verbose=0
        ).flatten()
        test_mse = float(np.mean((y_test - test_pred) ** 2))

        val_pred = model.predict(
            X_val,
            batch_size=self.config.prediction_batch_size,
            verbose=0
        ).flatten()

        # Prediction diagnostics
        pred_std = float(np.std(test_pred))
        pred_mean = float(np.mean(test_pred))
        actual_std = float(np.std(y_test))
        actual_mean = float(np.mean(y_test))

        print(f"\nPrediction Statistics:")
        print(f"  Predictions - Mean: {pred_mean:.6f}, Std: {pred_std:.6f}")
        print(f"  Actuals     - Mean: {actual_mean:.6f}, Std: {actual_std:.6f}")
        print(f"  Std Ratio (pred/actual): {pred_std / actual_std:.4f}")

        historical_returns = np.concatenate([y_train, y_val])
        val_expanding_means, expanding_means = self._resolve_benchmark_predictions(
            effective_benchmark,
            y_train,
            y_val,
            y_test,
        )

        val_ss_res = np.sum((y_val - val_pred) ** 2)
        val_ss_tot = np.sum((y_val - val_expanding_means) ** 2)
        val_r2 = float(1 - (val_ss_res / val_ss_tot))

        ss_res = np.sum((y_test - test_pred) ** 2)
        ss_tot = np.sum((y_test - expanding_means) ** 2)
        test_r2 = float(1 - (ss_res / ss_tot))

        if live_plot is not None:
            live_plot.cleanup()
            del live_plot

        del callbacks
        MemoryManager.aggressive_cleanup()

        if self.config.debug_memory:
            MemoryManager.print_memory_usage("AFTER model deletion")

        # Portfolio analysis
        portfolio_stats = None
        decile_returns_df = None
        ts_returns_df = None

        if test_dates is not None:
            if portfolio_mode == "panel":
                portfolio_stats, decile_returns_df = PortfolioAnalyzer.analyze_predictions(
                    test_dates,
                    test_pred,
                    y_test,
                    annualization_periods=annualization_periods,
                    asset_ids=asset_ids,
                    trading_config=trading_config,
                )
                PortfolioAnalyzer.print_portfolio_stats(portfolio_stats)
            elif portfolio_mode == "ts":
                train_val_returns = np.concatenate([y_train, y_val])
                portfolio_stats, ts_returns_df = PortfolioAnalyzer.analyze_predictions_ts(
                    test_dates, test_pred, y_test,
                    strategy_config=ts_strategy_config,
                    trading_config=trading_config,
                    annualization_periods=annualization_periods,
                    train_returns=train_val_returns,
                )
                PortfolioAnalyzer.print_ts_portfolio_stats(portfolio_stats)

        total_flops = cumulative_flops[-1]
        total_pf_days = cumulative_pf_days[-1]

        print(f"\nResults: Train={train_loss:.6f} | Val={val_loss:.6f} (R²={val_r2:.4f}) | "
              f"Test={test_mse:.6f} (R²={test_r2:.4f})")
        print(f"Time: {train_time:.1f}s | Compute: {total_pf_days:.2e} PF-days")

        # Build results dictionary
        results_dict = {
            'model_name': model_name,
            'target_params': int(target_params),
            'actual_params': int(actual_params),
            'architecture': architecture,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'test_loss': test_mse,
            'val_r2': val_r2,
            'test_r2': test_r2,
            'train_time': float(train_time),
            'total_flops': float(total_flops),
            'pf_days': float(total_pf_days),
            'epochs': int(epochs),
            'batch_size': int(self.config.batch_size),
            'learning_rate': float(self.config.learning_rate),
            'flops_per_epoch': float(flops_per_epoch),
            'normalization': self.config.normalization.value,
            'architecture_mode': self.config.architecture_mode.value,
            'portfolio_mode': portfolio_mode,
            'benchmark': self._benchmark_display_name(effective_benchmark),
            'benchmark_config': {
                'mode': self._benchmark_display_name(effective_benchmark),
            },
            'annualization_periods': int(annualization_periods),
            'portfolio_config': ScalingLawConfig._serialize_value(effective_portfolio_config),
            'trading_config': ScalingLawConfig._serialize_value(trading_config),
            'ts_strategy_config': ScalingLawConfig._serialize_value(ts_strategy_config),
            'training_curve': {
                'epochs': list(range(1, epochs + 1)),
                'train_loss': train_loss_history,
                'val_loss': val_loss_history,
                'cumulative_flops': [float(x) for x in cumulative_flops],
                'cumulative_pf_days': [float(x) for x in cumulative_pf_days]
            }
        }

        if portfolio_stats is not None:
            results_dict['portfolio_stats'] = portfolio_stats
            if portfolio_mode == "panel" and decile_returns_df is not None:
                results_dict['decile_returns'] = decile_returns_df
            elif portfolio_mode == "ts" and ts_returns_df is not None:
                results_dict['ts_returns'] = ts_returns_df

        # Optionally save the trained model to disk and record its path
        if self.config.save_models:
            try:
                model_path = self.results_manager.model_path(model_name)
                model_path.parent.mkdir(parents=True, exist_ok=True)
                model.save(str(model_path))

                results_dict["model_save"] = str(model_path)
                print(f"✓ Saved model to {model_path}")
            except Exception as e:
                print(f"⚠ Could not save model {model_name}: {e}")

        del test_pred
        del val_pred
        del historical_returns
        del expanding_means
        del val_expanding_means
        del model
        del normalizer
        del history
        del optimizer

        MemoryManager.aggressive_cleanup()

        if self.config.debug_memory:
            MemoryManager.print_memory_usage("END of train_single_model")

        return results_dict

    @staticmethod
    def _format_date_range(dates: Optional[List[Any]]) -> str:
        if not dates:
            return "n/a"
        return f"{dates[0]} to {dates[-1]}"

    @classmethod
    def _print_data_split_summary(cls, split_result: DataSplitResult):
        print(
            f"Train: {len(split_result.X_train):,} | "
            f"Val: {len(split_result.X_val):,} | "
            f"Test: {len(split_result.X_test):,}"
        )

        if all(
                dates is not None
                for dates in (
                    split_result.train_dates,
                    split_result.val_dates,
                    split_result.test_date_values,
                )
        ):
            print(
                f"Train: {cls._format_date_range(split_result.train_dates)} | "
                f"Val: {cls._format_date_range(split_result.val_dates)} | "
                f"Test: {cls._format_date_range(split_result.test_date_values)}"
            )

    def prepare_data_splits(
            self,
            df: pd.DataFrame,
            feature_cols: List[str],
            target_col: str = 'xret',
            date_col: str = 'date',
            asset_id_col: Optional[str] = None
    ) -> Tuple[np.ndarray, ...]:
        """
        Prepare train/validation/test splits from a DataFrame.

        Uses time-based splitting to avoid look-ahead bias.
        """
        if asset_id_col is None:
            asset_id_col = self.config.portfolio.asset_id_col

        split_result = self.data_splitter.prepare(
            df,
            feature_cols,
            target_col,
            date_col,
            asset_id_col=asset_id_col,
        )
        self._last_split_result = split_result

        if split_result.test_sample is not None:
            test_sample_path = self.results_manager.save_test_sample(split_result.test_sample)
            print(f"Saved test sample: {len(split_result.X_test):,} obs → {test_sample_path}")

        self._print_data_split_summary(split_result)
        return split_result.as_tuple()

    @staticmethod
    def _validate_run_inputs(
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: np.ndarray,
            y_val: np.ndarray,
            X_test: np.ndarray,
            y_test: np.ndarray,
            test_dates: Optional[np.ndarray],
            asset_ids: Optional[np.ndarray] = None
    ):
        split_pairs = (
            ("train", X_train, y_train),
            ("val", X_val, y_val),
            ("test", X_test, y_test),
        )
        for split_name, X, y in split_pairs:
            if X is None or y is None:
                raise ValueError(f"run() requires X_{split_name} and y_{split_name}")
            if len(X) != len(y):
                raise ValueError(
                    f"run() received mismatched X/y lengths for {split_name}: "
                    f"X_{split_name}={len(X):,}, y_{split_name}={len(y):,}"
                )
            if len(X) == 0:
                raise ValueError(
                    f"run() received an empty {split_name} split; train, val, and "
                    "test sets must all be non-empty"
                )

        if test_dates is not None and len(test_dates) != len(y_test):
            raise ValueError(
                "run() received test_dates with the wrong length: "
                f"test_dates={len(test_dates):,}, y_test={len(y_test):,}"
            )
        if asset_ids is not None and len(asset_ids) != len(y_test):
            raise ValueError(
                "run() received asset_ids with the wrong length: "
                f"asset_ids={len(asset_ids):,}, y_test={len(y_test):,}"
            )

    def run(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: np.ndarray,
            y_val: np.ndarray,
            X_test: np.ndarray,
            y_test: np.ndarray,
            test_dates: Optional[np.ndarray] = None,
            portfolio: Optional[Union[str, PortfolioMode]] = None,
            kappa: Optional[float] = None,
            benchmark: Optional[Union[str, Callable[..., Any], BenchmarkConfig]] = None,
            asset_ids: Optional[np.ndarray] = None
    ) -> List[Dict[str, Any]]:
        """
        Run the complete scaling law experiment.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            X_test: Test features
            y_test: Test targets
            test_dates: Optional test dates for portfolio analysis
            portfolio: Portfolio analysis mode ('panel' or 'ts')
                - 'panel': Cross-sectional decile portfolio analysis (default)
                - 'ts': Time-series strategy with π_t = κ * μ_hat_{t+1}
            kappa: Risk-scaling constant for time-series strategy (default=1.0)
            benchmark: Benchmark for OOS R².
                'historical_mean'          – fixed mean of training+validation set (default).
                'historical_mean_updating' – expanding mean that updates within the test period.
                'ar1'                      – AR(1) estimated once on training+validation data,
                                             coefficients frozen throughout the test period.
                'ar1_updating'             – AR(1) initially estimated on training+validation
                                             data, then re-estimated on an expanding window
                                             as each test return is observed.

        Returns:
            List of result dictionaries for each model
        """
        self._validate_run_inputs(
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            test_dates,
            asset_ids=asset_ids,
        )

        portfolio_config = (
            self.config.portfolio
            if portfolio is None
            else PortfolioConfig(mode=portfolio)
        )
        portfolio_mode = portfolio_config.mode.value
        effective_benchmark = (
            self.config.benchmark
            if benchmark is None
            else benchmark if isinstance(benchmark, BenchmarkConfig)
            else BenchmarkConfig(mode=benchmark)
        )
        effective_kappa = self.config.ts_strategy.kappa if kappa is None else float(kappa)

        # Validate portfolio mode
        if portfolio_mode not in ["panel", "ts"]:
            raise ValueError(f"portfolio must be 'panel' or 'ts', got '{portfolio_mode}'")

        if isinstance(effective_benchmark.mode, str) and effective_benchmark.mode not in VALID_BENCHMARK_MODES:
            raise ValueError(
                f"benchmark must be one of {sorted(VALID_BENCHMARK_MODES)}, "
                f"got '{effective_benchmark.mode}'"
            )

        resume_mode = ResumeMode.coerce(self.config.resume)

        # Parse and filter parameter sizes
        param_sizes_int = [self.parse_size(s) for s in self.config.param_sizes]

        if self.config.stop_at_size is not None:
            stop_at_int = self.parse_size(self.config.stop_at_size)
            param_sizes_int = [p for p in param_sizes_int if p <= stop_at_int]

        if self.config.start_at_size is not None:
            start_at_int = self.parse_size(self.config.start_at_size)
            param_sizes_int = [p for p in param_sizes_int if p >= start_at_int]

        print(f"\n{'=' * 80}")
        print("SCALING LAWS EXPERIMENT CONFIGURATION")
        print('=' * 80)
        print(f"Model sizes to test: {len(param_sizes_int)}")
        if param_sizes_int:
            print(f"Range: {param_sizes_int[0]:,} to {param_sizes_int[-1]:,} parameters")
        print(f"Epochs: {'Variable by size' if callable(self.config.epochs) else self.config.epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"Normalization: {self.config.normalization.value}")
        print(f"Architecture mode: {self.config.architecture_mode.value}")
        print(f"Portfolio mode: {portfolio_mode}")
        if portfolio_mode == "ts":
            print(f"Kappa (risk scaling): {effective_kappa}")
        print(f"Output directory: {self.config.output_dir}")
        print(f"Resume mode: {resume_mode.value}")
        print(f"Debug memory: {self.config.debug_memory}")
        print('=' * 80)

        # Initialize results files
        self.results_manager.initialize_files(resume_mode)
        existing_model_names = set()
        if resume_mode == ResumeMode.SKIP_EXISTING:
            existing_model_names = self.results_manager.load_existing_model_names()

        # Set random seeds
        np.random.seed(self.config.random_state)
        tf.random.set_seed(self.config.random_state)

        experiment_start = time.time()

        if self.config.debug_memory:
            MemoryManager.print_memory_usage("BEFORE training loop")

        for i, size in enumerate(param_sizes_int):
            model_epochs = self.config.get_epochs(size)
            model_name_str = f"model_{size}"
            if self.config.run_name:
                model_name_str += f"_{self.config.run_name}"

            print(f"\n{'#' * 80}")
            print(f"PROGRESS: [{i + 1}/{len(param_sizes_int)}] - "
                  f"{(i + 1) / len(param_sizes_int) * 100:.1f}% Complete")
            print(f"Model Size: {size:,} params | Epochs: {model_epochs}")
            print(f"{'#' * 80}")

            if resume_mode == ResumeMode.SKIP_EXISTING and model_name_str in existing_model_names:
                print(f"↷ Skipping {model_name_str}: existing result found")
                continue

            if self.config.debug_memory:
                MemoryManager.print_memory_usage(f"START model {i + 1}")

            try:
                result = self.train_single_model(
                    X_train, y_train, X_val, y_val, X_test, y_test,
                    target_params=size,
                    test_dates=test_dates,
                    epochs=model_epochs,
                    model_name=model_name_str,
                    portfolio=portfolio_mode,
                    kappa=kappa,
                    benchmark=effective_benchmark,
                    asset_ids=asset_ids,
                )

                # Handle portfolio returns based on mode
                if portfolio_mode == "panel" and 'decile_returns' in result:
                    decile_returns = result.pop('decile_returns')
                    self.results_manager.save_decile_returns_to_csv(
                        decile_returns, result['model_name']
                    )
                    del decile_returns
                    gc.collect()
                elif portfolio_mode == "ts" and 'ts_returns' in result:
                    ts_returns = result.pop('ts_returns')
                    self.results_manager.save_ts_returns_to_csv(
                        ts_returns, result['model_name']
                    )
                    del ts_returns
                    gc.collect()

                # Save results
                self.results_manager.save_result_to_pickle(result)
                self.results_manager.save_result_to_json()

                print(f"✓ Results saved to {self.config.output_dir}")

                del result
                MemoryManager.aggressive_cleanup()

                if self.config.debug_memory:
                    MemoryManager.print_memory_usage(f"END model {i + 1} (after cleanup)")

            except Exception as e:
                print(f"✗ ERROR training {size} parameter model: {e}")
                import traceback
                traceback.print_exc()
                MemoryManager.aggressive_cleanup()
                continue

        total_time = time.time() - experiment_start
        print(f"\n{'=' * 80}")
        print(f"EXPERIMENT COMPLETE")
        print(f"Total time: {total_time / 3600:.2f} hours")
        print(f"Results saved to: {self.config.output_dir}")
        print('=' * 80)

        return self.results_manager.load_results()

    def run_from_dataframe(
            self,
            df: pd.DataFrame,
            feature_cols: List[str],
            target_col: str = 'xret',
            date_col: str = 'date',
            portfolio: Optional[Union[str, PortfolioMode]] = None,
            kappa: Optional[float] = None,
            benchmark: Optional[Union[str, Callable[..., Any], BenchmarkConfig]] = None,
            asset_id_col: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Run the scaling law experiment directly from a DataFrame.

        This is a convenience method that handles data splitting internally.

        Args:
            df: Input DataFrame with features, target, and dates
            feature_cols: List of feature column names
            target_col: Target column name
            date_col: Date column name
            portfolio: Portfolio analysis mode ('panel' or 'ts')
            kappa: Risk-scaling constant for time-series strategy (default=1.0)
            benchmark: Benchmark for OOS R².
                'historical_mean'          – fixed mean of training+validation set (default).
                'historical_mean_updating' – expanding mean that updates within the test period.
                'ar1'                      – AR(1) estimated once on training+validation data,
                                             coefficients frozen throughout the test period.
                'ar1_updating'             – AR(1) initially estimated on training+validation
                                             data, then re-estimated on an expanding window
                                             as each test return is observed.

        Returns:
            List of result dictionaries for each model
        """
        print("\nPreparing data...")
        X_train, y_train, X_val, y_val, X_test, y_test, test_dates = self.prepare_data_splits(
            df,
            feature_cols,
            target_col,
            date_col,
            asset_id_col=asset_id_col,
        )
        test_asset_ids = (
            self._last_split_result.test_asset_ids
            if self._last_split_result is not None
            else None
        )

        return self.run(X_train, y_train, X_val, y_val, X_test, y_test, test_dates,
                        portfolio=portfolio, kappa=kappa, benchmark=benchmark,
                        asset_ids=test_asset_ids)

    def _print_results_table(self):
        """Print a summary table of val RMSE, test RMSE, and LS50 Sharpe for all trained models."""
        try:
            with open(self.results_manager.pkl_path, 'rb') as f:
                results = pickle.load(f)
        except Exception:
            try:
                with open(self.results_manager.json_path, 'r') as f:
                    results = json.load(f)
            except Exception as e:
                print(f"⚠ Could not load results for summary table: {e}")
                return

        results = sorted(results, key=lambda r: r.get('actual_params', 0))

        col_model  = 22
        col_metric = 14

        header = (f"{'Model':<{col_model}} {'Val RMSE':>{col_metric}} "
                  f"{'Test RMSE':>{col_metric}} {'Sharpe LS50':>{col_metric}}")
        sep = '─' * len(header)

        print(f"\n{'=' * len(header)}")
        print("SCALING LAW RESULTS SUMMARY")
        print(f"{'=' * len(header)}")
        print(header)
        print(sep)

        for r in results:
            model_name  = r.get('model_name', 'N/A')
            val_loss    = r.get('val_loss')
            test_loss   = r.get('test_loss')
            port_stats  = r.get('portfolio_stats')
            port_mode   = r.get('portfolio_mode', 'panel')

            val_rmse_str  = f"{np.sqrt(val_loss)  * 100:.4f}" if val_loss  is not None else 'N/A'
            test_rmse_str = f"{np.sqrt(test_loss) * 100:.4f}" if test_loss is not None else 'N/A'

            if port_stats is not None:
                if port_mode == 'panel' and 'LS_50' in port_stats:
                    sharpe_str = f"{port_stats['LS_50']['sharpe']:.4f}"
                elif port_mode == 'ts' and 'strategy' in port_stats:
                    sharpe_str = f"{port_stats['strategy']['sharpe']:.4f}"
                else:
                    sharpe_str = 'N/A'
            else:
                sharpe_str = 'N/A'

            print(f"{model_name:<{col_model}} {val_rmse_str:>{col_metric}} "
                  f"{test_rmse_str:>{col_metric}} {sharpe_str:>{col_metric}}")

        print(f"{'=' * len(header)}\n")

    def create_plots(self, dpi: int = 300):
        """Create all scaling law visualizations."""
        plotter = ScalingLawPlotter(self.config.output_dir, self.config.output.artifacts)
        plotter.create_all_plots(dpi=dpi)
        self._print_results_table()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_system_info():
    """Print system and TensorFlow information."""
    print("=" * 80)
    print("NEURAL NETWORK SCALING LAWS FOR FINANCE")
    print("=" * 80)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU devices: {tf.config.list_physical_devices('GPU')}")
    print("=" * 80)
    print()
