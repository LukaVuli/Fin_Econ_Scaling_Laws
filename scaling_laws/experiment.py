"""End-to-end neural-network scaling-law experiment."""

from __future__ import annotations

import gc
import json
import os
import pickle
import time
from dataclasses import fields
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from .callbacks import (
    FuzzyStopCallback,
    LivePlotCallback,
    R2PercentMetric,
    SingleLineProgressCallback,
)
from .config import (
    BenchmarkConfig,
    PortfolioConfig,
    ScalingLawConfig,
    TSStrategyConfig,
)
from .data_splitter import DataSplitResult, DataSplitter
from .enums import PortfolioMode, ResumeMode, VALID_BENCHMARK_MODES
from .model_builder import ModelBuilder
from .plotting import ScalingLawPlotter
from .portfolio import PortfolioAnalyzer
from .results import ResultsManager
from .utils.format import format_params, parse_size
from .utils.memory import MemoryManager


class ScalingLawExperiment:
    """
    Main class for running neural network scaling law experiments.

    This class orchestrates the entire scaling law experiment, including
    data preparation, model training, result saving, and visualization.

    Example:
        >>> config = ScalingLawConfig(
        ...     param_sizes=['1K', '10K', '100K'],
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
        self.results_manager = ResultsManager(self.config.output.output_dir, self.config)
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
        if self.config.compute.enable_determinism:
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
            if self.config.compute.precision == 8 and not self.config.compute.mixed_precision_policy:
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
            if self.config.compute.mixed_precision_policy
            else f"{self.config.compute.precision}-bit"
        )
        print(
            f"✓ Precision: {precision_label} "
            f"(policy={active_policy.name}, compute dtype={active_policy.compute_dtype})"
        )

    @staticmethod
    def make_param_sizes(min_size: int = 1_000, max_size: int = 1_000_000, num: int = 8) -> List[str]:
        """Generate a geometrically-spaced list of parameter sizes."""
        grid = np.geomspace(min_size, max_size, num=num)
        grid = np.unique(np.round(grid).astype(int))
        return [format_params(int(n)) for n in grid]

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

    def _save_training_history_plot(
            self,
            history_dict: Dict[str, List[float]],
            target_params: int,
            model_name: str,
            show_r2: bool,
    ) -> Optional[Path]:
        """Render the training-history plot for one model and save it to disk.

        Mirrors the styling of LivePlotCallback (log-scale loss, twin R²
        axis when available) but builds a fresh, off-screen Figure so the
        save works regardless of whether the live window was shown. The
        PNG lands next to where the model would be saved, at
        ``Models/<model_name>_training.png``.
        """
        try:
            from matplotlib.figure import Figure
        except Exception as exc:
            print(f"⚠ Could not import matplotlib to save training plot: {exc}")
            return None

        losses = [
            float(v) if v is not None and np.isfinite(v) and v > 0 else np.nan
            for v in history_dict.get('loss', [])
        ]
        val_losses = [
            float(v) if v is not None and np.isfinite(v) and v > 0 else np.nan
            for v in history_dict.get('val_loss', [])
        ]
        if not losses:
            return None
        epochs_list = list(range(len(losses)))

        r2_values = history_dict.get('r2_percent')
        val_r2_values = history_dict.get('val_r2_percent')
        plot_r2 = bool(show_r2 and r2_values is not None and val_r2_values is not None)
        if plot_r2:
            r2_values = [
                float(v) if v is not None and np.isfinite(float(v)) else np.nan
                for v in r2_values
            ]
            val_r2_values = [
                float(v) if v is not None and np.isfinite(float(v)) else np.nan
                for v in val_r2_values
            ]

        n_params = int(target_params)
        if n_params >= 1_000_000:
            value, suffix = n_params / 1_000_000, "M"
            size_text = f"{value:.2f}".rstrip("0").rstrip(".") + suffix
        elif n_params >= 1_000:
            value, suffix = n_params / 1_000, "K"
            size_text = f"{value:.2f}".rstrip("0").rstrip(".") + suffix
        else:
            size_text = str(n_params)
        title_text = f"Training Progress — {size_text} parameters"

        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        train_line, = ax.plot(epochs_list, losses, 'b-',
                              label='Training Loss', linewidth=2)
        val_line, = ax.plot(epochs_list, val_losses, 'r-',
                            label='Validation Loss', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (RMSE in Percent)')
        ax.set_title(title_text)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        finite_y = [v for v in (*losses, *val_losses)
                    if v is not None and np.isfinite(v) and v > 0]
        if finite_y:
            ymin = min(finite_y)
            ymax = max(finite_y)
            if ymin == ymax:
                ymin, ymax = ymin / 2.0, ymax * 2.0
            ax.set_ylim(ymin / 1.2, ymax * 1.2)
        if epochs_list:
            ax.set_xlim(-0.5, max(epochs_list[-1], 1) + 0.5)

        if plot_r2:
            ax_r2 = ax.twinx()
            train_r2_line, = ax_r2.plot(epochs_list, r2_values, 'b--',
                                        label='Training R²', linewidth=2)
            val_r2_line, = ax_r2.plot(epochs_list, val_r2_values, 'r--',
                                      label='Validation R²', linewidth=2)
            ax_r2.set_ylabel('R² (%)')
            finite_r2 = [v for v in (*r2_values, *val_r2_values)
                         if v is not None and np.isfinite(v)]
            if finite_r2:
                r2min = min(finite_r2)
                r2max = max(finite_r2)
                if r2min == r2max:
                    r2min, r2max = r2min - 1.0, r2max + 1.0
                pad = max((r2max - r2min) * 0.1, 0.1)
                ax_r2.set_ylim(r2min - pad, r2max + pad)
            ax.legend(
                [train_line, val_line, train_r2_line, val_r2_line],
                ['Training Loss', 'Validation Loss',
                 'Training R²', 'Validation R²'],
                loc='upper left',
            )
        else:
            ax.legend(loc='upper left')

        try:
            models_dir = self.results_manager.models_dir
            models_dir.mkdir(parents=True, exist_ok=True)
            save_path = models_dir / f"{model_name}_training.png"
            fig.savefig(str(save_path), dpi=100, bbox_inches='tight')
            return save_path
        except Exception as exc:
            print(f"⚠ Could not save training plot for {model_name}: {exc}")
            return None
        finally:
            fig.clear()

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

        if self.config.runtime.debug_memory:
            MemoryManager.print_memory_usage("START of train_single_model")

        # Aggressive cleanup before building
        MemoryManager.aggressive_cleanup()

        if self.config.runtime.debug_memory:
            MemoryManager.print_memory_usage("AFTER initial cleanup")

        # Build model
        model, normalizer, actual_params, architecture = self.model_builder.build_model(
            X_train.shape[1], target_params
        )

        if normalizer is not None:
            normalizer.adapt(X_train)

        print(f"Architecture: {architecture}")
        print(f"Actual parameters: {actual_params:,}")

        flops_per_epoch = float(self.config.compute.estimate_flops_per_epoch(
            actual_params=actual_params,
            train_samples=len(X_train),
            input_dim=X_train.shape[1],
            architecture=architecture,
            model=model
        ))

        # Compile model
        if self.config.training.clip_norm is not None:
            optimizer = Adam(learning_rate=self.config.training.learning_rate,
                             clipnorm=self.config.training.clip_norm)
        else:
            optimizer = Adam(learning_rate=self.config.training.learning_rate)

        model.compile(
            loss='mean_squared_error',
            optimizer=optimizer,
            metrics=[R2PercentMetric()]
        )

        # Setup callbacks
        callbacks = [SingleLineProgressCallback()]

        if self.config.scheduler.lr_scheduler_enabled:
            patience = self.config.get_lr_scheduler_patience(epochs)
            lr_scheduler = ReduceLROnPlateau(
                monitor='val_loss',
                factor=self.config.scheduler.lr_scheduler_factor,
                patience=patience,
                min_lr=self.config.scheduler.lr_scheduler_min_lr,
                verbose=1
            )
            callbacks.append(lr_scheduler)

        live_plot = None
        if self.config.runtime.show_live_plots:
            live_plot = LivePlotCallback(
                show_r2=self.config.runtime.show_live_r2,
                target_params=target_params,
            )
            callbacks.append(live_plot)

        # Fuzzy stop: optionally let training continue past `epochs` and
        # terminate at the next confirmed local optimum of a smoothed
        # validation metric. The model.fit budget is widened to
        # `epochs + max_extra_epochs`; the callback halts within that.
        fuzzy_stop = self.config.fuzzy_stop.resolve(epochs)
        fuzzy_stop_callback: Optional[FuzzyStopCallback] = None
        fit_epochs = epochs
        if fuzzy_stop.enabled:
            fuzzy_stop_callback = FuzzyStopCallback(
                scheduled_epochs=epochs,
                monitor=fuzzy_stop.monitor,
                mode=fuzzy_stop.mode,
                smoothing_window=fuzzy_stop.smoothing_window,
                patience=fuzzy_stop.patience,
                max_extra_epochs=fuzzy_stop.max_extra_epochs,
                restore_best_weights=fuzzy_stop.restore_best_weights,
            )
            callbacks.append(fuzzy_stop_callback)
            fit_epochs = epochs + fuzzy_stop.max_extra_epochs
            print(
                f"Training floor: {epochs} epochs (always trained in full). "
                f"Fuzzy stop active only in epochs {epochs + 1}-{fit_epochs}: "
                f"monitor={fuzzy_stop.monitor} ({fuzzy_stop.mode}), "
                f"window={fuzzy_stop.smoothing_window}, "
                f"patience={fuzzy_stop.patience}, "
                f"max_extra_epochs={fuzzy_stop.max_extra_epochs}"
            )

        if self.config.runtime.debug_memory:
            MemoryManager.print_memory_usage("BEFORE training")

        # Train
        start_time = time.time()
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=fit_epochs,
            batch_size=self.config.training.train_batch_size,
            validation_batch_size=self.config.training.validation_batch_size,
            verbose=0,
            callbacks=callbacks
        )
        train_time = time.time() - start_time

        # Persist the training-history plot to disk so it can be inspected
        # later — independent of whether the live window was shown.
        self._save_training_history_plot(
            history.history,
            target_params=target_params,
            model_name=model_name,
            show_r2=self.config.runtime.show_live_r2,
        )

        if self.config.runtime.debug_memory:
            MemoryManager.print_memory_usage("AFTER training")

        # Compute metrics. Use the actual epoch count from history because
        # fuzzy stop (or any future early-termination path) can make the
        # realized run shorter than `fit_epochs`.
        actual_epochs = len(history.history['loss'])
        cumulative_flops = [(i + 1) * flops_per_epoch for i in range(actual_epochs)]
        cumulative_pf_days = [f / 8.64e19 for f in cumulative_flops]

        train_loss = float(history.history['loss'][-1])
        val_loss = float(history.history['val_loss'][-1])
        train_loss_history = [float(x) for x in history.history['loss']]
        val_loss_history = [float(x) for x in history.history['val_loss']]

        test_pred = model.predict(
            X_test,
            batch_size=self.config.training.prediction_batch_size,
            verbose=0
        ).flatten()
        test_mse = float(np.mean((y_test - test_pred) ** 2))

        val_pred = model.predict(
            X_val,
            batch_size=self.config.training.prediction_batch_size,
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

        if self.config.runtime.debug_memory:
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

        print("\n" + "-" * 80)
        print("MODEL RESULTS")
        print("-" * 80)
        print("Loss Metrics")
        print(f"  Train Loss:       {train_loss:.6f}")
        print(f"  Validation Loss:  {val_loss:.6f}")
        print(f"  Test Loss:        {test_mse:.6f}")
        print("R2 Metrics")
        print(f"  Validation R2:    {val_r2:.4f}")
        print(f"  Test R2:          {test_r2:.4f}")
        print("Runtime and Compute")
        print(f"  Training Time:    {train_time:.1f}s")
        print(f"  Total Compute:    {total_pf_days:.2e} PF-days")
        print("-" * 80)

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
            'epochs': int(actual_epochs),
            'scheduled_epochs': int(epochs),
            'batch_size': int(self.config.training.train_batch_size),
            'learning_rate': float(self.config.training.learning_rate),
            'flops_per_epoch': float(flops_per_epoch),
            'normalization': self.config.architecture.normalization.value,
            'architecture_mode': self.config.architecture.architecture_mode.value,
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
                'epochs': list(range(1, actual_epochs + 1)),
                'train_loss': train_loss_history,
                'val_loss': val_loss_history,
                'cumulative_flops': [float(x) for x in cumulative_flops],
                'cumulative_pf_days': [float(x) for x in cumulative_pf_days]
            }
        }

        if fuzzy_stop_callback is not None:
            results_dict['fuzzy_stop'] = {
                'enabled': True,
                'monitor': fuzzy_stop.monitor,
                'mode': fuzzy_stop.mode,
                'smoothing_window': fuzzy_stop.smoothing_window,
                'patience': fuzzy_stop.patience,
                'max_extra_epochs': fuzzy_stop.max_extra_epochs,
                'triggered': bool(fuzzy_stop_callback.triggered),
                'stop_reason': fuzzy_stop_callback.stop_reason,
                'restored_to_epoch': fuzzy_stop_callback.restored_to_epoch,
                'kept_current_epoch': bool(fuzzy_stop_callback.kept_current_epoch),
            }
        else:
            results_dict['fuzzy_stop'] = {'enabled': False}

        if portfolio_stats is not None:
            results_dict['portfolio_stats'] = portfolio_stats
            if portfolio_mode == "panel" and decile_returns_df is not None:
                results_dict['decile_returns'] = decile_returns_df
            elif portfolio_mode == "ts" and ts_returns_df is not None:
                results_dict['ts_returns'] = ts_returns_df

        # Optionally save the trained model to disk and record its path
        if self.config.output.save_models:
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

        if self.config.runtime.debug_memory:
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
                raise ValueError(f"run_from_arrays() requires X_{split_name} and y_{split_name}")
            if len(X) != len(y):
                raise ValueError(
                    f"run_from_arrays() received mismatched X/y lengths for {split_name}: "
                    f"X_{split_name}={len(X):,}, y_{split_name}={len(y):,}"
                )
            if len(X) == 0:
                raise ValueError(
                    f"run_from_arrays() received an empty {split_name} split; train, val, and "
                    "test sets must all be non-empty"
                )

        if test_dates is not None and len(test_dates) != len(y_test):
            raise ValueError(
                "run_from_arrays() received test_dates with the wrong length: "
                f"test_dates={len(test_dates):,}, y_test={len(y_test):,}"
            )
        if asset_ids is not None and len(asset_ids) != len(y_test):
            raise ValueError(
                "run_from_arrays() received asset_ids with the wrong length: "
                f"asset_ids={len(asset_ids):,}, y_test={len(y_test):,}"
            )

    def run_from_arrays(
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

        resume_mode = ResumeMode.coerce(self.config.runtime.resume)

        # Parse and filter parameter sizes
        param_sizes_int = [parse_size(s) for s in self.config.param_sizes]

        if self.config.stop_at_size is not None:
            stop_at_int = parse_size(self.config.stop_at_size)
            param_sizes_int = [p for p in param_sizes_int if p <= stop_at_int]

        if self.config.start_at_size is not None:
            start_at_int = parse_size(self.config.start_at_size)
            param_sizes_int = [p for p in param_sizes_int if p >= start_at_int]

        print(f"\n{'=' * 80}")
        print("SCALING LAWS EXPERIMENT CONFIGURATION")
        print('=' * 80)
        print(f"Model sizes to test: {len(param_sizes_int)}")
        if param_sizes_int:
            print(f"Range: {param_sizes_int[0]:,} to {param_sizes_int[-1]:,} parameters")
        print(f"Epochs: {'Variable by size' if callable(self.config.training.epochs) else self.config.training.epochs}")
        print(f"Batch size: {self.config.training.train_batch_size}")
        print(f"Learning rate: {self.config.training.learning_rate}")
        print(f"Architecture mode: {self.config.architecture.architecture_mode.value}")
        print(f"Portfolio mode: {portfolio_mode}")
        if portfolio_mode == "ts":
            print(f"Kappa (risk scaling): {effective_kappa}")
        print(f"Output directory: {self.config.output.output_dir}")
        print(f"Resume mode: {resume_mode.value}")
        print(f"Debug memory: {self.config.runtime.debug_memory}")
        print('=' * 80)

        # Initialize results files
        self.results_manager.initialize_files(resume_mode)
        existing_model_names = set()
        if resume_mode == ResumeMode.SKIP_EXISTING:
            existing_model_names = self.results_manager.load_existing_model_names()

        # Set random seeds
        np.random.seed(self.config.runtime.random_state)
        tf.random.set_seed(self.config.runtime.random_state)

        experiment_start = time.time()

        if self.config.runtime.debug_memory:
            MemoryManager.print_memory_usage("BEFORE training loop")

        for i, size in enumerate(param_sizes_int):
            model_epochs = self.config.get_epochs(size)
            model_name_str = f"model_{size}"
            if self.config.runtime.run_name:
                model_name_str += f"_{self.config.runtime.run_name}"

            print(f"\n{'#' * 80}")
            print(f"PROGRESS: [{i + 1}/{len(param_sizes_int)}] - "
                  f"{(i + 1) / len(param_sizes_int) * 100:.1f}% Complete")
            print(f"Model Size: {size:,} params | Epochs: {model_epochs}")
            print(f"{'#' * 80}")

            if resume_mode == ResumeMode.SKIP_EXISTING and model_name_str in existing_model_names:
                print(f"↷ Skipping {model_name_str}: existing result found")
                continue

            if self.config.runtime.debug_memory:
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

                print(f"Output Directory: {self.config.output.output_dir}")

                del result
                MemoryManager.aggressive_cleanup()

                if self.config.runtime.debug_memory:
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
        print(f"Results saved to: {self.config.output.output_dir}")
        print('=' * 80)

        return self.results_manager.load_results()

    def run(
            self,
            df: pd.DataFrame,
            X: List[str],
            y: str = 'xret',
            date_col: str = 'date',
            portfolio: Optional[Union[str, PortfolioMode]] = None,
            kappa: Optional[float] = None,
            benchmark: Optional[Union[str, Callable[..., Any], BenchmarkConfig]] = None,
            asset_id_col: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Run the scaling law experiment directly from a DataFrame.

        This is the canonical entrypoint: load your data into a DataFrame,
        specify the feature columns (``X``) and target column (``y``), and let
        the experiment handle splitting, training, and evaluation across the
        configured parameter sizes.

        For workflows where you have already split your data into NumPy arrays
        (e.g. from a custom pipeline), use ``run_from_arrays(...)`` instead.

        Args:
            df: Input DataFrame with features, target, and dates.
            X: List of feature column names.
            y: Target column name.
            date_col: Date column name.
            portfolio: Portfolio analysis mode ('panel' or 'ts').
            kappa: Risk-scaling constant for time-series strategy (default=1.0).
            benchmark: Benchmark for OOS R².
                'historical_mean'          – fixed mean of training+validation set (default).
                'historical_mean_updating' – expanding mean that updates within the test period.
                'ar1'                      – AR(1) estimated once on training+validation data,
                                             coefficients frozen throughout the test period.
                'ar1_updating'             – AR(1) initially estimated on training+validation
                                             data, then re-estimated on an expanding window
                                             as each test return is observed.
            asset_id_col: Optional asset-identifier column for panel data.

        Returns:
            List of result dictionaries for each model.
        """
        print("\nPreparing data...")
        X_train, y_train, X_val, y_val, X_test, y_test, test_dates = self.prepare_data_splits(
            df,
            X,
            y,
            date_col,
            asset_id_col=asset_id_col,
        )
        test_asset_ids = (
            self._last_split_result.test_asset_ids
            if self._last_split_result is not None
            else None
        )

        return self.run_from_arrays(X_train, y_train, X_val, y_val, X_test, y_test, test_dates,
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

    def create_plots(self, dpi: int = 300, include_ls_breakpoints: bool = False):
        """Create all scaling law visualizations.

        Set `include_ls_breakpoints=True` to also produce the LS_50, LS_30,
        and LS_10 long-short breakpoint Sharpe plots in addition to the
        default forecast-weighted Sharpe plot.
        """
        plotter = ScalingLawPlotter(self.config.output.output_dir, self.config.output.artifacts)
        plotter.create_all_plots(dpi=dpi, include_ls_breakpoints=include_ls_breakpoints)
        self._print_results_table()
