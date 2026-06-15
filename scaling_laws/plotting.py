"""Scaling-law plotting utilities."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, minimize, minimize_scalar

from .config import ArtifactNames


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
            ax.legend(fontsize=13, loc='upper left', framealpha=0.9)

        plt.tight_layout()

        save_path = self.results_path / save_name
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"✓ Plot saved to: {save_path}")
        plt.show()

        return fig, ax

    def plot_sharpe_ratio_scaling(
            self,
            breakpoint: str = '50',
            portfolio: Optional[str] = None,
            x_axis: str = 'compute',
            fit_curve: bool = True,
            figsize: Tuple[int, int] = (12, 8),
            title: Optional[str] = None,
            save_name: Optional[str] = None,
            dpi: int = 300,
            use_compute_weighting: bool = False
    ) -> Tuple[Optional[plt.Figure], Optional[plt.Axes]]:
        """Plot Sharpe ratio vs model size/compute.

        If `portfolio` is provided (e.g. 'Forecast_Weighted'), it is used directly
        as the portfolio_stats key. Otherwise the long-short LS_{breakpoint} key
        is used.
        """
        results = self._load_results()
        print(f"\nLoaded {len(results)} models")

        if portfolio is not None:
            breakpoint_key = portfolio
            breakpoint_label = portfolio.replace('_', ' ')
            save_suffix = portfolio
            title_prefix = 'Portfolio Sharpe Ratio'
        else:
            breakpoint_key = f'LS_{breakpoint}'
            breakpoint_label = f'{breakpoint}% Breakpoint'
            save_suffix = f'LS{breakpoint}'
            title_prefix = 'Long-Short Portfolio Sharpe Ratio'

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
            title = f'{title_prefix} ({breakpoint_label})'
        ax.set_title(title, fontsize=18, fontweight='bold', pad=20)

        ax.grid(True, alpha=0.3, linestyle='--')
        ax.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        ax.tick_params(labelsize=12)

        if fit_curve:
            ax.legend(fontsize=13, loc='upper left', framealpha=0.9)

        plt.tight_layout()

        if save_name is None:
            save_name = f'sharpe_ratio_{save_suffix}_vs_{x_axis}.png'
        save_path = self.results_path / save_name
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"✓ Plot saved to: {save_path}")
        plt.show()

        return fig, ax

    def create_all_plots(
            self,
            dpi: int = 300,
            use_compute_weighting: bool = False,
            include_ls_breakpoints: bool = False,
    ):
        """Create comprehensive set of scaling law plots.

        By default the forecast-weighted portfolio Sharpe is plotted.
        Set `include_ls_breakpoints=True` to additionally plot the LS_50,
        LS_30, and LS_10 long-short breakpoint Sharpe ratios.
        """
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

        print("\n5. Sharpe Ratio Forecast-Weighted vs Compute")
        self.plot_sharpe_ratio_scaling(
            portfolio='Forecast_Weighted',
            x_axis='compute',
            dpi=dpi,
            use_compute_weighting=use_compute_weighting
        )

        if include_ls_breakpoints:
            print("\n6. Sharpe Ratio LS50 vs Compute")
            self.plot_sharpe_ratio_scaling(
                breakpoint='50',
                x_axis='compute',
                dpi=dpi,
                use_compute_weighting=use_compute_weighting
            )

            print("\n7. Sharpe Ratio LS30 vs Compute")
            self.plot_sharpe_ratio_scaling(
                breakpoint='30',
                x_axis='compute',
                dpi=dpi,
                use_compute_weighting=use_compute_weighting
            )

            print("\n8. Sharpe Ratio LS10 vs Compute")
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

        ax.legend(fontsize=16, loc='upper left', framealpha=0.9)

        plt.tight_layout()

        if save_name is None:
            save_name = f'sharpe_ratio_ts_vs_{x_axis}.png'
        save_path = self.results_path / save_name
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"✓ Plot saved to: {save_path}")
        plt.show()

        return fig, ax

