"""Custom Keras callbacks used during training."""

from __future__ import annotations

import sys
import time
from typing import Any, ClassVar, Dict, List, Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback


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
    """Callback that displays training progress in real-time with a live plot.

    Uses a single persistent Tk window for the whole process: the first
    training builds it, every subsequent training reuses the same window and
    wipes its history so the new run starts from scratch in-place. No new
    windows ever spawn.

    Implementation note: this callback embeds a matplotlib Figure inside a
    self-managed Tk window via FigureCanvasTkAgg and drives Tk's event loop
    directly. It deliberately does not use pyplot for the live window, because
    pyplot's behavior depends on the active matplotlib backend — and on macOS
    in PyCharm the default backend (MacOSX, or worse PyCharm's snapshot-only
    'module://backend_interagg') will not propagate canvas updates from a
    script context. Tk + FigureCanvasTkAgg bypasses backend negotiation
    entirely and works the same whether you Run, Debug, or paste into the
    PyCharm Python console.
    """

    # Singleton window state shared across all instances so we reuse the
    # same Tk window for every training instead of opening a new one.
    _root: ClassVar[Any] = None
    _fig: ClassVar[Any] = None
    _ax: ClassVar[Any] = None
    _ax_r2: ClassVar[Any] = None
    _canvas: ClassVar[Any] = None
    _train_line: ClassVar[Any] = None
    _val_line: ClassVar[Any] = None
    _train_r2_line: ClassVar[Any] = None
    _val_r2_line: ClassVar[Any] = None
    _window_has_r2: ClassVar[bool] = False

    def __init__(self, show_r2: bool = True, target_params: Optional[int] = None):
        super().__init__()
        self.show_r2 = show_r2
        self.target_params = target_params
        self.losses: List[float] = []
        self.val_losses: List[float] = []
        self.r2_values: List[float] = []
        self.val_r2_values: List[float] = []
        self.epochs_list: List[int] = []

    @staticmethod
    def _format_params(n: int) -> str:
        """Compact human-readable parameter count, e.g. 5K, 1.5K, 1M.

        Strips trailing zeros so round targets like 5_000 / 1_000_000
        render as '5K' / '1M' (matching what the user typed in
        param_sizes), while non-round values like 1500 still render
        accurately as '1.5K'.
        """
        if n >= 1_000_000:
            value, suffix = n / 1_000_000, "M"
        elif n >= 1_000:
            value, suffix = n / 1_000, "K"
        else:
            return str(int(n))
        text = f"{value:.2f}".rstrip("0").rstrip(".")
        return f"{text}{suffix}"

    @classmethod
    def _build_window(cls, show_r2: bool):
        # Lazy imports so a missing tkinter on a headless box only fails
        # when show_live_plots is actually requested.
        import tkinter as tk
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        cls._root = tk.Tk()
        cls._root.title("Training Progress")
        cls._root.protocol("WM_DELETE_WINDOW", cls._on_window_close)
        cls._window_has_r2 = show_r2

        # Use a bare Figure (not plt.figure) so we never touch pyplot's
        # global state for the live window.
        cls._fig = Figure(figsize=(10, 6))
        cls._ax = cls._fig.add_subplot(111)
        cls._train_line, = cls._ax.plot(
            [], [], 'b-', label='Training Loss', linewidth=2
        )
        cls._val_line, = cls._ax.plot(
            [], [], 'r-', label='Validation Loss', linewidth=2
        )
        cls._ax.set_xlabel('Epoch')
        cls._ax.set_ylabel('Loss (RMSE in Percent)')
        cls._ax.grid(True, alpha=0.3)
        cls._ax.set_yscale('log')

        if show_r2:
            cls._ax_r2 = cls._ax.twinx()
            cls._train_r2_line, = cls._ax_r2.plot(
                [], [], 'b--', label='Training R²', linewidth=2
            )
            cls._val_r2_line, = cls._ax_r2.plot(
                [], [], 'r--', label='Validation R²', linewidth=2
            )
            cls._ax_r2.set_ylabel('R² (%)')
            handles = [
                cls._train_line, cls._val_line,
                cls._train_r2_line, cls._val_r2_line,
            ]
            labels = [
                'Training Loss', 'Validation Loss',
                'Training R²', 'Validation R²',
            ]
            cls._ax.legend(handles, labels, loc='upper left')
        else:
            cls._ax_r2 = None
            cls._train_r2_line = None
            cls._val_r2_line = None
            cls._ax.legend(loc='upper left')

        cls._canvas = FigureCanvasTkAgg(cls._fig, master=cls._root)
        cls._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        cls._canvas.draw()

        # Drain Tk's event queue so the window actually appears now,
        # before TF starts hammering the main thread with compute.
        cls._root.update()

    @classmethod
    def _reset_state(cls):
        cls._root = None
        cls._fig = None
        cls._ax = None
        cls._ax_r2 = None
        cls._canvas = None
        cls._train_line = None
        cls._val_line = None
        cls._train_r2_line = None
        cls._val_r2_line = None
        cls._window_has_r2 = False

    @classmethod
    def _on_window_close(cls):
        try:
            if cls._root is not None:
                cls._root.destroy()
        except Exception:
            pass
        cls._reset_state()

    def on_train_begin(self, logs=None):
        # Wipe history so the new model's plot starts from scratch.
        self.losses = []
        self.val_losses = []
        self.r2_values = []
        self.val_r2_values = []
        self.epochs_list = []

        # Prefer the user-specified target size (e.g. 5_000 → "5K") so
        # the title matches what was set in param_sizes. Fall back to
        # the realized parameter count if no target was passed in.
        if self.target_params is not None:
            n_params = int(self.target_params)
        else:
            try:
                n_params = int(self.model.count_params())
            except Exception:
                n_params = 0
        title_text = (
            f"Training Progress — {self._format_params(n_params)} parameters"
        )

        cls = type(self)
        if cls._root is None:
            cls._build_window(self.show_r2)

        # Reset the existing window's axes and lines in-place.
        cls._root.title(title_text)
        cls._ax.set_title(title_text)
        cls._train_line.set_data([], [])
        cls._val_line.set_data([], [])
        if cls._ax_r2 is not None and cls._train_r2_line is not None:
            cls._train_r2_line.set_data([], [])
            cls._val_r2_line.set_data([], [])
        cls._canvas.draw()
        cls._root.update()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        if loss is None or val_loss is None:
            return

        cls = type(self)
        if cls._root is None or cls._canvas is None:
            return

        loss = float(loss)
        val_loss = float(val_loss)

        self.epochs_list.append(epoch)
        # Mask non-finite or non-positive values with NaN so a single
        # bad epoch shows up as a gap, not as a vanished line.
        self.losses.append(loss if np.isfinite(loss) and loss > 0 else np.nan)
        self.val_losses.append(val_loss if np.isfinite(val_loss) and val_loss > 0 else np.nan)

        cls._train_line.set_data(self.epochs_list, self.losses)
        cls._val_line.set_data(self.epochs_list, self.val_losses)

        # Compute axis limits explicitly. autoscale_view on a log axis
        # with a small number of points often refuses to move off the
        # default (1, 10) range, which would render the curves
        # off-screen for losses below 1.
        finite_y = [v for v in (*self.losses, *self.val_losses)
                    if v is not None and np.isfinite(v) and v > 0]
        if finite_y:
            ymin = min(finite_y)
            ymax = max(finite_y)
            if ymin == ymax:
                ymin, ymax = ymin / 2.0, ymax * 2.0
            cls._ax.set_ylim(ymin / 1.2, ymax * 1.2)
        if self.epochs_list:
            xmax = max(self.epochs_list)
            cls._ax.set_xlim(-0.5, max(xmax, 1) + 0.5)

        if cls._ax_r2 is not None and cls._train_r2_line is not None:
            r2 = logs.get('r2_percent')
            val_r2 = logs.get('val_r2_percent')
            r2_val = float(r2) if r2 is not None else np.nan
            val_r2_val = float(val_r2) if val_r2 is not None else np.nan
            if not np.isfinite(r2_val):
                r2_val = np.nan
            if not np.isfinite(val_r2_val):
                val_r2_val = np.nan

            self.r2_values.append(r2_val)
            self.val_r2_values.append(val_r2_val)

            cls._train_r2_line.set_data(self.epochs_list, self.r2_values)
            cls._val_r2_line.set_data(self.epochs_list, self.val_r2_values)

            finite_r2 = [v for v in (*self.r2_values, *self.val_r2_values)
                         if v is not None and np.isfinite(v)]
            if finite_r2:
                r2min = min(finite_r2)
                r2max = max(finite_r2)
                if r2min == r2max:
                    r2min, r2max = r2min - 1.0, r2max + 1.0
                pad = max((r2max - r2min) * 0.1, 0.1)
                cls._ax_r2.set_ylim(r2min - pad, r2max + pad)

        cls._canvas.draw()
        # Pump Tk's event loop so the OS actually paints the new
        # frame between epochs.
        cls._root.update()

    def on_train_end(self, logs=None):
        # Window stays open; next on_train_begin will repaint in place.
        return

    def cleanup(self):
        """No-op: the window is intentionally persistent across runs.

        Kept for backwards compatibility with the per-model call site
        that historically tore the window down after each fit. Use
        ``_teardown`` to actually close the singleton window.
        """
        return

    @classmethod
    def _teardown(cls):
        """Explicitly tear down the persistent window."""
        if cls._root is not None:
            try:
                cls._root.destroy()
            except Exception:
                pass
        cls._reset_state()


class SingleLineProgressCallback(Callback):
    """Callback that prints compact training progress with timing."""

    def __init__(self):
        super().__init__()
        self.total_epochs = 0
        self.start_time = 0
        self.epoch_times: List[float] = []
        self.progress_lines = 4
        self.has_progress_block = False

    def on_train_begin(self, logs=None):
        self.total_epochs = self.params['epochs']
        self.start_time = time.time()
        self.epoch_times = []
        self.has_progress_block = False

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

        bar_length = 20
        filled = int(bar_length * current / total)
        bar = '█' * filled + '░' * (bar_length - filled)

        percent = 100 * current / total
        train_loss = np.sqrt(max(logs.get('loss', 0), 0)) * 100
        val_loss = np.sqrt(max(logs.get('val_loss', 0), 0)) * 100
        train_r2 = logs.get('r2_percent', 0)
        val_r2 = logs.get('val_r2_percent', 0)
        line = (
            f"[{bar}] Epoch: {current:>{len(str(total))}}/{total} ({percent:.1f}%) | "
            f"Runtime: {elapsed_str} and ETA: {eta_str} | "
            f"Train Loss: {train_loss:.2f}% (R2: {train_r2:.2f}%) | "
            f"Val Loss: {val_loss:.2f}% (R2: {val_r2:.2f}%)"
        )

        pad = max(0, getattr(self, '_last_line_len', 0) - len(line))
        sys.stdout.write("\r" + line + (" " * pad))
        sys.stdout.flush()
        self._last_line_len = len(line)
        self.has_progress_block = True

    def on_train_end(self, logs=None):
        if self.has_progress_block:
            sys.stdout.write("\n")
            sys.stdout.flush()
        self.epoch_times = []
        self.has_progress_block = False

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


class FuzzyStopCallback(Callback):
    """Extend training past ``scheduled_epochs`` and stop at a local optimum.

    The scheduled epoch count is treated as a hard *minimum* — training will
    not terminate before it. Raw monitor values are recorded for the whole
    run and converted to a causal median over the last ``smoothing_window``
    values; the smoothed-best value, the raw value at that epoch, and the
    weights at that epoch are tracked **only from the scheduled epoch
    onwards**. So the local optimum we lock on to is one that occurred in
    the extension window, not somewhere in the warm-up phase. Training
    halts once either:

    * ``patience`` epochs have elapsed since the last smoothed-best update
      within the extension window — the smoothed metric has demonstrably
      walked off its optimum, so we are past a local extremum; or
    * the extra-epoch budget is exhausted — a hard cap so noisy runs that
      never confirm a stop terminate in bounded time.

    On stop, the current epoch's raw monitor value is compared against the
    raw value at the smoothed-best epoch. If the current model is actually
    better on the raw metric, its weights are kept; otherwise weights are
    restored to the smoothed-best epoch (when ``restore_best_weights`` is
    True). The callback exposes ``triggered``, ``stop_reason``,
    ``actual_epochs``, ``restored_to_epoch``, and ``kept_current_epoch``
    for downstream logging.
    """

    _STOP_LOCAL_OPT: ClassVar[str] = "local_optimum_confirmed"
    _STOP_MAX_EXTRA: ClassVar[str] = "max_extra_epochs"

    def __init__(
            self,
            scheduled_epochs: int,
            monitor: str,
            mode: str,
            smoothing_window: int,
            patience: int,
            max_extra_epochs: int,
            restore_best_weights: bool = True,
    ):
        super().__init__()
        mode_normalized = str(mode).lower()
        if mode_normalized not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        self.scheduled_epochs = max(1, int(scheduled_epochs))
        self.monitor = str(monitor)
        self.mode = mode_normalized
        self.smoothing_window = max(1, int(smoothing_window))
        self.patience = max(1, int(patience))
        self.max_extra_epochs = max(0, int(max_extra_epochs))
        self.restore_best_weights = bool(restore_best_weights)

        self._raw_history: List[float] = []
        self._smoothed_history: List[float] = []
        self._best_value: Optional[float] = None
        self._best_raw: Optional[float] = None
        self._best_epoch: int = -1
        self._best_weights: Optional[List[np.ndarray]] = None
        self._last_raw: Optional[float] = None

        self.triggered: bool = False
        self.stop_reason: Optional[str] = None
        self.actual_epochs: int = 0
        self.restored_to_epoch: Optional[int] = None
        self.kept_current_epoch: bool = False

    def _is_better(self, candidate: float, incumbent: float) -> bool:
        if self.mode == "min":
            return candidate < incumbent
        return candidate > incumbent

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        logs = logs or {}
        if self.monitor not in logs:
            return
        raw = float(logs[self.monitor])
        if not np.isfinite(raw):
            return

        self._raw_history.append(raw)
        window = self._raw_history[-self.smoothing_window:]
        smoothed = float(np.median(window))
        self._smoothed_history.append(smoothed)
        self._last_raw = raw
        self.actual_epochs = epoch + 1

        # Phase 1: scheduled training is a hard floor — record raw history
        # so smoothing is warm at the boundary, but do not track best or
        # consider stopping.
        if epoch + 1 < self.scheduled_epochs:
            return

        # Phase 2: smoothed-best tracking starts at the scheduled boundary
        # so the local optimum we lock on to is one found in the extension
        # window, never one from the warm-up phase.
        if self._best_value is None or self._is_better(smoothed, self._best_value):
            self._best_value = smoothed
            self._best_raw = raw
            self._best_epoch = epoch
            if self.restore_best_weights:
                self._best_weights = self.model.get_weights()

        epochs_since_best = epoch - self._best_epoch
        extra_epochs_used = (epoch + 1) - self.scheduled_epochs

        if epochs_since_best >= self.patience:
            self._stop(self._STOP_LOCAL_OPT)
        elif extra_epochs_used >= self.max_extra_epochs:
            self._stop(self._STOP_MAX_EXTRA)

    def _stop(self, reason: str):
        self.triggered = True
        self.stop_reason = reason

        # Final pick: compare the current epoch's raw monitor value against
        # the raw value at the smoothed-best epoch. Keep current unless
        # best is *strictly* better — that way ties (incl. the trivial
        # case where best epoch == current epoch) avoid an unnecessary
        # set_weights call. This guards against the smoothed-best epoch
        # being a lucky-smoothing point with a mediocre raw value, and
        # against the current epoch being a noise spike past a real peak.
        current_wins = (
            self._best_raw is None
            or self._last_raw is None
            or not self._is_better(self._best_raw, self._last_raw)
        )

        if current_wins:
            self.kept_current_epoch = True
            comparison_note = (
                f"; kept current epoch (raw={self._last_raw:.6g}"
                + (f" beats best-smoothed raw={self._best_raw:.6g}"
                   if self._best_raw is not None else "")
                + ")"
            )
        elif self.restore_best_weights and self._best_weights is not None:
            self.model.set_weights(self._best_weights)
            self.restored_to_epoch = self._best_epoch + 1
            self.kept_current_epoch = False
            comparison_note = (
                f"; restored weights to epoch {self.restored_to_epoch} "
                f"(raw={self._best_raw:.6g} beats current raw={self._last_raw:.6g})"
            )
        else:
            self.kept_current_epoch = True
            comparison_note = ""

        self.model.stop_training = True
        sys.stdout.write(
            f"\nFuzzy stop [{reason}] at epoch {self.actual_epochs} "
            f"(monitor={self.monitor}, mode={self.mode}){comparison_note}\n"
        )
        sys.stdout.flush()

