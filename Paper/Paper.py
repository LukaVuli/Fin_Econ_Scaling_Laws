
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path
import math

def plot_validation_curves(results, save_path='validation_loss_curves.png', dpi=300,
                           x_min=None, x_max=None, y_min=None, y_max=None):
    """Create validation loss scaling curves plot."""
    print("\n" + "=" * 80)
    print("Creating Validation Loss Curves Plot")
    print("=" * 80)

    fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')

    all_params = [r['actual_params'] for r in results]
    norm = plt.matplotlib.colors.LogNorm(vmin=min(all_params), vmax=max(all_params))
    cmap = plt.cm.viridis

    all_x_values = []
    all_y_values = []
    final_x_values = []
    final_y_values = []

    for result in results:
        params = result['actual_params']
        curve = result['training_curve']

        x_data = curve['cumulative_pf_days']
        y_data = curve['val_loss']

        all_x_values.extend(x_data)
        all_y_values.extend(y_data)

        color = cmap(norm(params))

        ax.plot(x_data, y_data, color=color, linewidth=3.5, alpha=0.8)
        ax.scatter(x_data[-1], y_data[-1], color=color, s=100,
                   edgecolors='black', linewidth=1.5, zorder=5)

        final_x_values.append(x_data[-1])
        final_y_values.append(y_data[-1])

    final_x_values = np.array(final_x_values)
    final_y_values = np.array(final_y_values)
    all_x_values = np.array(all_x_values)
    all_y_values = np.array(all_y_values)

    print(f"\nLoaded {len(results)} models")

    fit_params = None

    if len(final_x_values) > 3:
        try:
            def scaling_law(C, L_inf, a, b):
                return L_inf + a * np.power(C, b)

            L_inf_guess = np.min(final_y_values) * 0.9
            y_range = np.max(final_y_values) - np.min(final_y_values)
            a_guess = y_range * (np.max(final_x_values) ** 0.2)
            b_guess = -0.2

            bounds = ([0, 1e-10, -2.0], [np.min(final_y_values), np.inf, 0])

            popt, pcov = curve_fit(scaling_law, final_x_values, final_y_values,
                                   p0=[L_inf_guess, a_guess, b_guess],
                                   bounds=bounds, maxfev=10000)

            L_inf, a, b = popt
            C0 = np.power(1.0 / a, 1.0 / b)

            exponent = int(np.floor(np.log10(abs(C0))))
            mantissa = C0 / (10 ** exponent)

            x_smooth = np.logspace(np.log10(final_x_values.min()),
                                   np.log10(final_x_values.max()), 100)
            y_smooth = scaling_law(x_smooth, L_inf, a, b)

            y_pred = scaling_law(final_x_values, L_inf, a, b)
            ss_res = np.sum((final_y_values - y_pred) ** 2)
            ss_tot = np.sum((final_y_values - np.mean(final_y_values)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)

            fit_params = {
                'L_inf': L_inf, 'C0': C0, 'mantissa': mantissa,
                'exponent': exponent, 'b': b, 'r_squared': r_squared
            }

            ax.plot(x_smooth, y_smooth, 'r--', linewidth=3, alpha=0.9,
                    label=f'Scaling Law: $L = {L_inf:.3f} + \\left(\\frac{{C}}{{{mantissa:.1f} \\times 10^{{{exponent}}}}}\\right)^{{{b:.2f}}}$, $R^2$: {r_squared * 100:.1f}%',
                    zorder=10)

            print(
                f"\n✓ Fitted scaling law: L_inf={L_inf:.4f}, C0={mantissa:.1f}×10^{exponent}, b={b:.4f}, R²={r_squared:.4f}")

        except Exception as e:
            print(f"\n✗ Could not fit scaling law: {e}")

    # Set axis ranges
    if x_min is None:
        reasonable_idx = all_y_values < 1.0
        if reasonable_idx.any():
            x_data_filtered = all_x_values[reasonable_idx]
            x_min = x_data_filtered.min() * 0.5
        else:
            x_min = all_x_values.min() * 0.7

    if x_max is None:
        x_max = all_x_values.max() * 1.5

    ax.set_xlim(x_min, x_max)

    if y_min is None:
        converged_idx = all_x_values > (final_x_values.min() * 0.5)
        if converged_idx.any():
            y_data_filtered = all_y_values[converged_idx]
            y_min = y_data_filtered.min() * 0.85
        else:
            y_min = all_y_values.min() * 0.9

    if y_max is None:
        converged_idx = all_x_values > (final_x_values.min() * 0.5)
        if converged_idx.any():
            y_data_filtered = all_y_values[converged_idx]
            y_max = min(0.095, y_data_filtered.max() * 1.3)
        else:
            y_max = min(0.095, all_y_values.max() * 1.2)

    ax.set_ylim(y_min, y_max)

    # Formatting
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Compute (PetaFLOP-days)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Validation Loss', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.tick_params(labelsize=12)

    # Custom formatter to ensure mantissa always has 1 decimal place
    from matplotlib.ticker import FuncFormatter
    def format_with_decimal(x, pos):
        if x == 0:
            return '0'
        # Calculate mantissa and exponent
        exponent = int(np.floor(np.log10(abs(x))))
        mantissa = x / (10 ** exponent)
        # Format with exactly 1 decimal place
        return r'${:.1f} \times 10^{{{}}}$'.format(mantissa, exponent)

    ax.yaxis.set_major_formatter(FuncFormatter(format_with_decimal))
    ax.yaxis.set_minor_formatter(FuncFormatter(format_with_decimal))

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('Parameters', rotation=270, labelpad=25, fontsize=16, fontweight='bold')
    cbar.ax.tick_params(labelsize=11)

    if fit_params is not None:
        ax.legend(fontsize=16, loc='upper right', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"✓ Plot saved to: {save_path}")
    plt.show()

    return fig, ax, fit_params


def plot_test_loss(results, save_path='test_loss_vs_compute.png', dpi=300):
    """Plot final test loss vs compute."""
    print("\n" + "=" * 80)
    print("Creating Test Loss vs Compute Plot")
    print("=" * 80)

    params = np.array([r['actual_params'] for r in results])
    test_loss = np.array([r['test_loss'] for r in results])
    compute = np.array([r['pf_days'] for r in results])

    print(f"\nLoaded {len(results)} models")

    fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')

    scatter = ax.scatter(compute, test_loss, c=params, cmap='viridis',
                         s=200, alpha=0.7, edgecolors='black', linewidth=1.5,
                         norm=plt.matplotlib.colors.LogNorm())

    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label('Parameters', rotation=270, labelpad=25, fontsize=16, fontweight='bold')
    cbar.ax.tick_params(labelsize=11)

    fit_params = None

    if len(compute) > 3:
        try:
            def scaling_law(C, L_inf, a, b):
                return L_inf + a * np.power(C, b)

            L_inf_guess = np.min(test_loss) * 0.9
            y_range = np.max(test_loss) - np.min(test_loss)
            a_guess = y_range * (np.max(compute) ** 0.2)
            b_guess = -0.2

            bounds = ([0, 1e-20, -2.0], [np.min(test_loss), np.inf, 0])

            popt, pcov = curve_fit(scaling_law, compute, test_loss,
                                   p0=[L_inf_guess, a_guess, b_guess],
                                   bounds=bounds, maxfev=10000)

            L_inf, a, b = popt
            C0 = np.power(1.0 / a, 1.0 / b)

            exponent = int(np.floor(np.log10(abs(C0))))
            mantissa = C0 / (10 ** exponent)

            x_smooth = np.logspace(np.log10(compute.min()),
                                   np.log10(compute.max()), 100)
            y_smooth = scaling_law(x_smooth, L_inf, a, b)

            y_pred = scaling_law(compute, L_inf, a, b)
            ss_res = np.sum((test_loss - y_pred) ** 2)
            ss_tot = np.sum((test_loss - np.mean(test_loss)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)

            fit_params = {
                'L_inf': L_inf, 'C0': C0, 'mantissa': mantissa,
                'exponent': exponent, 'b': b, 'r_squared': r_squared
            }

            ax.plot(x_smooth, y_smooth, 'r--', linewidth=3, alpha=0.8,
                    label=f'Scaling Law: $L = {L_inf:.3f} + \\left(\\frac{{C}}{{{mantissa:.1f} \\times 10^{{{exponent}}}}}\\right)^{{{b:.2f}}}$, $R^2$: {r_squared * 100:.1f}%')

            print(
                f"\n✓ Fitted scaling law: L_inf={L_inf:.6f}, C0={mantissa:.1f}×10^{exponent}, b={b:.4f}, R²={r_squared:.4f}")

        except Exception as e:
            print(f"\n✗ Could not fit curve: {e}")

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Compute (PetaFLOP-days)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Test Loss', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.tick_params(labelsize=12)

    if fit_params is not None:
        ax.legend(fontsize=16, loc='best', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"✓ Plot saved to: {save_path}")
    plt.show()

    return fig, ax, fit_params


def merge_scaling_results(main_json_path, new_json_path, output_json_path,
                          models_to_remove=None):
    """
    Merge two scaling results JSON files, removing specified models from main and adding new ones.

    Args:
        main_json_path: Path to the main JSON with all models
        new_json_path: Path to the new JSON with replacement/additional models
        output_json_path: Path where the merged JSON will be saved
        models_to_remove: List of model names to remove from main JSON (e.g., ['model_500', 'model_1000'])

    Returns:
        Combined results list
    """
    print("\n" + "=" * 80)
    print("MERGING SCALING RESULTS")
    print("=" * 80)

    if models_to_remove is None:
        models_to_remove = []

    # Load main JSON
    print(f"\nLoading main results from: {main_json_path}")
    with open(main_json_path, 'r') as f:
        main_results = json.load(f)
    print(f"  Loaded {len(main_results)} models")

    # Load new JSON
    print(f"\nLoading new results from: {new_json_path}")
    with open(new_json_path, 'r') as f:
        new_results = json.load(f)
    print(f"  Loaded {len(new_results)} models")

    # Remove specified models from main results
    if models_to_remove:
        print(f"\nRemoving models: {models_to_remove}")
        filtered_main = [r for r in main_results if r['model_name'] not in models_to_remove]
        removed_count = len(main_results) - len(filtered_main)
        print(f"  Removed {removed_count} models")
        main_results = filtered_main

    # Add new models
    print(f"\nAdding {len(new_results)} new models:")
    for r in new_results:
        print(f"  + {r['model_name']} ({r['actual_params']:,} params)")

    # Combine
    combined_results = main_results + new_results

    # Sort by actual_params
    combined_results.sort(key=lambda x: x['actual_params'])

    print(f"\nCombined total: {len(combined_results)} models")

    # Save
    print(f"\nSaving merged results to: {output_json_path}")
    with open(output_json_path, 'w') as f:
        json.dump(combined_results, f, indent=2)

    print("✓ Merge complete!")
    print("=" * 80)

    return combined_results


def plot_sharpe_ratio(results, portfolio_csv_path, breakpoint=10,
                      save_path='sharpe_ratio_vs_compute.png', dpi=300):
    """
    Plot Sharpe ratios of long-short portfolios vs compute.

    Args:
        results: List of model results
        portfolio_csv_path: Path to CSV with portfolio returns
        breakpoint: Which LS portfolio to use (50, 30, or 10)
        save_path: Path to save the plot
        dpi: Resolution for saved plot

    Returns:
        fig, ax, fit_params (dict with SR_inf, C0, b, r_squared or None)
    """
    print("\n" + "=" * 80)
    print(f"Creating Sharpe Ratio vs Compute Plot (LS_{breakpoint})")
    print("=" * 80)

    import pandas as pd

    # Load portfolio returns
    df = pd.read_csv(portfolio_csv_path)

    # Extract data for each model
    params_list = []
    sharpe_ratios = []
    compute_list = []

    for result in results:
        param_count = result['actual_params']
        compute = result['pf_days']

        # Find the LS column for this model
        col_name = f"{param_count}_LS_{breakpoint}"

        if col_name in df.columns:
            returns = df[col_name].dropna()

            # Calculate Sharpe ratio (assuming risk-free rate = 0)
            mean_return = returns.mean()
            std_return = returns.std()

            if std_return > 0:
                sharpe_ratio = mean_return / std_return

                params_list.append(param_count)
                sharpe_ratios.append(sharpe_ratio*math.sqrt(12))
                compute_list.append(compute)

    params = np.array(params_list)
    sharpe_ratios = np.array(sharpe_ratios)
    compute = np.array(compute_list)

    print(f"\nLoaded {len(params)} models")
    print(f"Compute range: {compute.min():.2e} to {compute.max():.2e}")
    print(f"Sharpe ratio range: {sharpe_ratios.min():.4f} to {sharpe_ratios.max():.4f}")

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')

    # Plot points
    scatter = ax.scatter(compute, sharpe_ratios, c=params, cmap='viridis',
                         s=200, alpha=0.7, edgecolors='black', linewidth=1.5,
                         norm=plt.matplotlib.colors.LogNorm())

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label('Parameters', rotation=270, labelpad=25, fontsize=16, fontweight='bold')
    cbar.ax.tick_params(labelsize=11)

    # Initialize fit_params
    fit_params = None

    # Fit curve (Sharpe ratio should increase with compute and plateau)
    if len(compute) > 3:
        try:
            # Scaling law: SR = SR_inf - a * C^b (where b < 0 for diminishing returns)
            def scaling_law(C, SR_inf, a, b):
                """Sharpe approaches SR_inf as compute increases"""
                return SR_inf - a * np.power(C, b)

            # Initial guess
            SR_inf_guess = np.max(sharpe_ratios) * 1.1  # 110% of max
            sharpe_range = np.max(sharpe_ratios) - np.min(sharpe_ratios)
            a_guess = sharpe_range * (np.min(compute) ** 0.2)
            b_guess = -0.2

            print(f"\n  Initial guesses: SR_inf={SR_inf_guess:.4f}, a={a_guess:.4e}, b={b_guess:.4f}")

            # Set bounds
            # SR_inf should be greater than max observed Sharpe
            # a should be positive
            # b should be negative (diminishing returns)
            bounds = (
                [np.max(sharpe_ratios), 1e-10, -2.0],  # lower bounds
                [np.max(sharpe_ratios) * 5, np.inf, -0.01]  # upper bounds
            )

            # Fit
            popt, pcov = curve_fit(scaling_law, compute, sharpe_ratios,
                                   p0=[SR_inf_guess, a_guess, b_guess],
                                   bounds=bounds,
                                   maxfev=10000)

            SR_inf, a, b = popt

            print(f"\n  Fitted parameters: SR_inf={SR_inf:.4f}, a={a:.4e}, b={b:.4f}")

            # Check for valid parameters
            if a <= 0 or b >= 0 or np.isnan(a) or np.isnan(b) or np.isnan(SR_inf):
                raise ValueError(f"Invalid fitted parameters: SR_inf={SR_inf}, a={a}, b={b}")

            # Convert to (C/C0)^b form
            # Since a * C^b = (C/C0)^b, we have C0 = (1/a)^(1/b)
            C0 = np.power(1.0 / a, 1.0 / b)

            if not np.isfinite(C0) or C0 <= 0:
                raise ValueError("C0 not finite or positive")

            # Format C0
            exponent = int(np.floor(np.log10(abs(C0))))
            mantissa = C0 / (10 ** exponent)

            # Generate smooth curve
            x_smooth = np.logspace(np.log10(compute.min()),
                                   np.log10(compute.max()), 100)
            y_smooth = scaling_law(x_smooth, SR_inf, a, b)

            # Calculate R²
            y_pred = scaling_law(compute, SR_inf, a, b)
            ss_res = np.sum((sharpe_ratios - y_pred) ** 2)
            ss_tot = np.sum((sharpe_ratios - np.mean(sharpe_ratios)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)

            # Only use the fit if R² is reasonable
            if r_squared > 0.3:
                # Store fit parameters
                fit_params = {
                    'SR_inf': SR_inf,
                    'C0': C0,
                    'mantissa': mantissa,
                    'exponent': exponent,
                    'b': b,
                    'r_squared': r_squared
                }

                # Plot
                ax.plot(x_smooth, y_smooth, 'r--', linewidth=3, alpha=0.8,
                        label=f'Scaling Law: $SR = {SR_inf:.3f} - \\left(\\frac{{C}}{{{mantissa:.1f} \\times 10^{{{exponent}}}}}\\right)^{{{b:.2f}}}$, $R^2$: {r_squared * 100:.1f}%')

                print(f"\n✓ Fitted scaling law:")
                print(f"  SR_inf: {SR_inf:.4f}")
                print(f"  C0: {mantissa:.1f} × 10^{exponent}")
                print(f"  b: {b:.4f}")
                print(f"  R²: {r_squared:.4f}")
            else:
                print(f"\n✗ Scaling law fit poor (R²={r_squared:.4f}), not displaying")

        except Exception as e:
            print(f"\n✗ Could not fit curve: {e}")

    # Formatting
    ax.set_xscale('log')
    ax.set_xlabel('Compute (PetaFLOP-days)', fontsize=16, fontweight='bold')
    ax.set_ylabel(f'Test Sample Sharpe Ratio)', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.tick_params(labelsize=16)

    if fit_params is not None:
        ax.legend(fontsize=16, loc='best', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"✓ Plot saved to: {save_path}")
    plt.show()

    return fig, ax, fit_params


def main(results_path=None, returns_path=None, output_dir=None, dpi=300,
         x_min=None, x_max=None, y_min=None, y_max=None, breaker = 50):
    """Main function to create plots and summary table."""
    print("\n" + "=" * 80)
    print("REPRODUCING SCALING LAW PLOTS")
    print("=" * 80)

    # Load results
    if results_path is None:
        for path in ['/mnt/user-data/uploads/scaling_results.pkl',
                     '/mnt/user-data/uploads/scaling_results.json',
                     'scaling_results.pkl',
                     'scaling_results.json']:
            if Path(path).exists():
                results_path = path
                break

    if results_path is None:
        raise FileNotFoundError("Could not find scaling_results.pkl or .json")

    results_path = Path(results_path)
    print(f"\nLoading results from: {results_path}")

    if results_path.suffix == '.pkl':
        with open(results_path, 'rb') as f:
            results = pickle.load(f)
    else:
        with open(results_path, 'r') as f:
            results = json.load(f)

    if output_dir is None:
        output_dir = results_path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")

    # Print summary table
    print("\n" + "=" * 80)
    print("MODEL SUMMARY TABLE")
    print("=" * 80)

    print(f"\n{'Model':<15} {'Parameters':>12} {'Final Val Loss':>16} {'Test Loss':>12}")
    print("-" * 80)

    for result in results:
        model_name = result['model_name']
        params = result['actual_params']
        final_val_loss = result['training_curve']['val_loss'][-1]
        test_loss = result['test_loss']

        print(f"{model_name:<15} {params:>12,} {final_val_loss:>16.6f} {test_loss:>12.6f}")

    # Create plots
    fig1, ax1, val_fit_params = plot_validation_curves(
        results,
        save_path=output_dir / 'validation_loss_curves.png',
        dpi=dpi, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max
    )

    fig2, ax2, test_fit_params = plot_test_loss(
        results,
        save_path=output_dir / 'test_loss_vs_compute.png',
        dpi=dpi
    )

    # Check for portfolio CSV and create Sharpe ratio plot

    fig3, ax3, sharpe_fit_params = plot_sharpe_ratio(
        results,
        portfolio_csv_path=returns_path,
        breakpoint=breaker,  # Default to LS_10
        save_path=output_dir / 'sharpe_ratio_vs_compute.png',
        dpi=dpi
    )

    # Print scaling law equations
    print("\n" + "=" * 80)
    print("SCALING LAW EQUATIONS")
    print("=" * 80)

    if val_fit_params is not None:
        L = val_fit_params['L_inf']
        m = val_fit_params['mantissa']
        e = val_fit_params['exponent']
        b = val_fit_params['b']
        r2 = val_fit_params['r_squared']

        print(f"\nValidation Loss Scaling Law:")
        print(f"  L = {L:.4f} + (C / ({m:.1f} × 10^{e}))^{b:.4f},  R² = {r2:.4f}")
        print(f"\n  LaTeX: L = {L:.4f} + \\left(\\frac{{C}}{{{m:.1f} \\times 10^{{{e}}}}}\\right)^{{{b:.4f}}}")
    else:
        print("\nValidation Loss Scaling Law: Could not fit")

    if test_fit_params is not None:
        L = test_fit_params['L_inf']
        m = test_fit_params['mantissa']
        e = test_fit_params['exponent']
        b = test_fit_params['b']
        r2 = test_fit_params['r_squared']

        print(f"\nTest Loss Scaling Law:")
        print(f"  L = {L:.4f} + (C / ({m:.1f} × 10^{e}))^{b:.4f},  R² = {r2:.4f}")
        print(f"\n  LaTeX: L = {L:.4f} + \\left(\\frac{{C}}{{{m:.1f} \\times 10^{{{e}}}}}\\right)^{{{b:.4f}}}")
    else:
        print("\nTest Loss Scaling Law: Could not fit")

    if sharpe_fit_params is not None:
        SR = sharpe_fit_params['SR_inf']
        m = sharpe_fit_params['mantissa']
        e = sharpe_fit_params['exponent']
        b = sharpe_fit_params['b']
        r2 = sharpe_fit_params['r_squared']

        print(f"\nSharpe Ratio Scaling Law (LS_10):")
        print(f"  SR = {SR:.4f} - (C / ({m:.1f} × 10^{e}))^{b:.4f},  R² = {r2:.4f}")
        print(f"\n  LaTeX: SR = {SR:.4f} - \\left(\\frac{{C}}{{{m:.1f} \\times 10^{{{e}}}}}\\right)^{{{b:.4f}}}")

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    'Return_scaling_results.json'
    main_json = "/Users/lukavulicevic/Desktop/ScalingLaws/Output/scaling_results_merged.json"
    new_json = "/Users/lukavulicevic/Desktop/ScalingLaws/Output/scaling_results.json"
    output_json = "/Users/lukavulicevic/Desktop/ScalingLaws/Output/scaling_results_merged.json"
    merge_scaling_results(main_json, new_json, output_json,models_to_remove=['model_1000'])


    results_path = '/Users/lukavulicevic/Desktop/ScalingLaws/Output/scaling_results_merged.json'
    returns_path = '/Users/lukavulicevic/Desktop/ScalingLaws/Output/portfolio_returns_merged.csv'

    results_path = '/Users/lukavulicevic/Desktop/ScalingLaws/Output/scaling_results.json'
    returns_path = '/Users/lukavulicevic/Desktop/ScalingLaws/Output/portfolio_returns.csv'
    output_dir = '/Users/lukavulicevic/Desktop/ScalingLaws/Output/'
    main(results_path=results_path, returns_path=returns_path,output_dir=output_dir, breaker=50,x_min=1e-9, x_max=1e-3, y_min=0.0485, y_max=0.073)


