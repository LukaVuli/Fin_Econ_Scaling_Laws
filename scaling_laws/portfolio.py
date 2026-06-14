"""Portfolio analytics computed from model predictions."""

from __future__ import annotations

import gc
from dataclasses import fields
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import ScalingLawConfig, TradingConfig, TSStrategyConfig


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
