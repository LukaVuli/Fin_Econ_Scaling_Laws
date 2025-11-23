
import numpy as np
import pandas as pd
from DataDefinitions.datadefinition import dd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Normalization, BatchNormalization, Dropout, Input
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import sys
import time
import json
import pickle
from pathlib import Path
from scipy.optimize import curve_fit
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# GPU CONFIGURATION
# ============================================================================
print("=" * 80)
print("NEURAL NETWORK SCALING LAWS FOR FINANCE")
print("=" * 80)
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU devices: {tf.config.list_physical_devices('GPU')}")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✓ Found {len(gpus)} GPU(s) - Metal acceleration enabled")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("⚠ No GPU found - using CPU only")

print("=" * 80)
print()


# ============================================================================
# CALLBACKS
# ============================================================================
class LivePlotCallback(Callback):
    """Callback that displays training progress in real-time."""

    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        self.epochs = []

        plt.ion()  # Turn on interactive mode
        self.fig, self.ax = plt.subplots(figsize=(10, 6))

    def on_epoch_end(self, epoch, logs={}):
        self.epochs.append(epoch)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

        # Clear and redraw
        self.ax.clear()
        self.ax.plot(self.epochs, self.losses, 'b-', label='Training Loss', linewidth=2)
        self.ax.plot(self.epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2)
        self.ax.set_xlabel('Epoch')
        self.ax.set_ylabel('Loss (MSE)')
        self.ax.set_title('Training Progress')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        self.ax.set_yscale('log')  # Log scale

        # Update the plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def on_train_end(self, logs={}):
        plt.ioff()  # Turn off interactive mode
        plt.close(self.fig)


class SingleLineProgress(Callback):
    """Print training progress on a single updating line with timing."""

    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']
        self.start_time = time.time()
        self.epoch_times = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = epoch + 1
        total = self.epochs

        elapsed = time.time() - self.start_time

        if current > 1:
            epoch_time = elapsed / current
            self.epoch_times.append(epoch_time)
            recent_avg = sum(self.epoch_times[-10:]) / min(len(self.epoch_times), 10)
            eta = recent_avg * (total - current)
        else:
            eta = 0

        elapsed_str = self.format_time(elapsed)
        eta_str = self.format_time(eta)

        bar_length = 40
        filled = int(bar_length * current / total)
        bar = '█' * filled + '░' * (bar_length - filled)

        percent = 100 * current / total
        msg = f"\r[{bar}] {percent:.1f}% | "
        msg += f"Loss: {logs.get('loss', 0):.6f} Val: {logs.get('val_loss', 0):.6f} | "
        msg += f"{elapsed_str} < {eta_str}"

        sys.stdout.write(msg)
        sys.stdout.flush()

    def on_train_end(self, logs=None):
        print()

    @staticmethod
    def format_time(seconds):
        if seconds < 0:
            return "..."
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"


# ============================================================================
# DATA PREPARATION
# ============================================================================

def get_excess_return(df):
    """Add excess returns to dataframe."""
    df = df.copy()
    factors = dd('famafrench', item='F-F_Research_Data_Factors', start='1900-01-01', end=None)
    data_ff3 = factors.extract()
    rf_series = data_ff3['RF'] / 100

    if not np.issubdtype(df['date'].dtype, np.datetime64):
        df['date'] = pd.to_datetime(df['date'])

    df['year_month'] = df['date'].dt.to_period('M').dt.to_timestamp()
    rf_df = rf_series.reset_index()
    rf_df.columns = ['year_month', 'rf']

    if hasattr(rf_df['year_month'].dtype, 'name') and 'period' in str(rf_df['year_month'].dtype):
        rf_df['year_month'] = rf_df['year_month'].dt.to_timestamp()
    elif rf_df['year_month'].dtype == 'object':
        rf_df['year_month'] = pd.to_datetime(rf_df['year_month'])

    df = df.merge(rf_df, on='year_month', how='left')
    df['xret'] = df['ret'] - df['rf']
    df = df.drop(columns=['year_month', 'rf'])

    return df


def create_lagged_features(df, base_char_cols, n_additional_lags=0,
                           include_historical_avg=False, firm_id_col='permno'):
    """
    Create lagged features and historical averages.

    Parameters:
    -----------
    df : DataFrame
        Input dataframe with date and firm identifier
    base_char_cols : list
        List of base characteristic column names (already lagged by 1 month)
    n_additional_lags : int
        Number of additional lags to create (e.g., 2 means add t-2, t-3)
        If n_additional_lags=11, you get t-2 through t-12 (total 12 lags including t-1)
    include_historical_avg : bool
        Whether to include historical average of each characteristic
    firm_id_col : str
        Column name for firm identifier

    Returns:
    --------
    df : DataFrame
        DataFrame with new lagged features
    expanded_char_cols : list
        List of all characteristic columns (base + lags + averages)
    """

    print(f"\n{'=' * 80}")
    print("CREATING LAGGED FEATURES")
    print('=' * 80)
    print(f"Base characteristics: {len(base_char_cols)}")
    print(f"Additional lags: {n_additional_lags}")
    print(f"Include historical average: {include_historical_avg}")

    df = df.copy()
    df = df.sort_values([firm_id_col, 'date'])

    expanded_char_cols = base_char_cols.copy()

    # Create additional lags
    if n_additional_lags > 0:
        print(f"\nCreating {n_additional_lags} additional lags...")
        for lag in range(2, n_additional_lags + 2):  # Start from lag 2 (since base is lag 1)
            print(f"  Creating lag {lag}...", end='')
            lag_count = 0
            for char in base_char_cols:
                new_col_name = f"{char}_lag{lag}"
                df[new_col_name] = df.groupby(firm_id_col)[char].shift(lag - 1)
                expanded_char_cols.append(new_col_name)
                lag_count += 1
            print(f" {lag_count} features created")

    # Create historical averages
    if include_historical_avg:
        print(f"\nCreating historical averages...")
        avg_count = 0
        for char in base_char_cols:
            new_col_name = f"{char}_hist_avg"
            # Expanding mean up to (but not including) current observation
            df[new_col_name] = (df.groupby(firm_id_col)[char]
                                .transform(lambda x: x.expanding().mean().shift(1)))
            expanded_char_cols.append(new_col_name)
            avg_count += 1
        print(f"  {avg_count} historical averages created")

    total_features = len(expanded_char_cols)
    print(f"\nTotal features: {total_features}")
    print(f"  Base: {len(base_char_cols)}")
    if n_additional_lags > 0:
        print(f"  Additional lags: {len(base_char_cols) * n_additional_lags}")
    if include_historical_avg:
        print(f"  Historical averages: {len(base_char_cols)}")
    print('=' * 80)

    return df, expanded_char_cols


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def negative_ic_loss(y_true, y_pred):
    """Information Coefficient loss - maximize correlation."""
    y_true_centered = y_true - tf.reduce_mean(y_true)
    y_pred_centered = y_pred - tf.reduce_mean(y_pred)

    numerator = tf.reduce_sum(y_true_centered * y_pred_centered)
    denominator = (tf.sqrt(tf.reduce_sum(tf.square(y_true_centered))) *
                   tf.sqrt(tf.reduce_sum(tf.square(y_pred_centered))) + 1e-7)

    ic = numerator / denominator
    return -ic


# ============================================================================
# MODEL BUILDING
# ============================================================================

def build_model_with_target_params(input_dim, target_params, weight_decay=1e-4):
    """
    Build a neural network targeting approximately N parameters.
    Uses consistent architecture family scaled to hit target size.
    """

    # Define depth based on model size
    if target_params < 1_000:
        layers_template = [1.0, 0.5]
    elif target_params < 10_000:
        layers_template = [1.0, 1.0, 0.5]
    elif target_params < 100_000:
        layers_template = [1.0, 1.0, 0.5, 0.25]
    elif target_params < 1_000_000:
        layers_template = [1.0, 1.0, 1.0, 0.5, 0.5, 0.25]
    else:
        layers_template = [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.25, 0.25, 0.125]

    # Binary search for width that gives approximately target_params
    def count_params(width):
        total = input_dim * width  # First layer
        for i in range(len(layers_template) - 1):
            curr_width = int(width * layers_template[i])
            next_width = int(width * layers_template[i + 1])
            total += curr_width * next_width
        total += int(width * layers_template[-1]) * 1  # Output layer
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

    # Build architecture
    layers_config = [max(1, int(best_width * scale)) for scale in layers_template]

    # Build model
    inputs = Input(shape=(input_dim,))
    normalizer = Normalization(axis=-1)
    x = normalizer(inputs)

    scaled_weight_decay = weight_decay * np.sqrt(10_000 / target_params)

    for i, neurons in enumerate(layers_config):
        x = Dense(neurons, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(scaled_weight_decay))(x)
        x = BatchNormalization()(x)

        # Light dropout in middle layers
        if i > 1 and i < len(layers_config) - 1 and target_params > 10_000:
            x = Dropout(0.1)(x)

    outputs = Dense(1)(x)
    model = Model(inputs=inputs, outputs=outputs)

    actual_params = model.count_params()

    return model, normalizer, actual_params, layers_config


# ============================================================================
# TRAINING
# ============================================================================
def train_single_model(X_train, y_train, X_val, y_val, X_test, y_test,
                       target_params, test_dates=None, epochs=500, batch_size=8192,
                       learning_rate=0.001, model_name="model",
                       show_live_plot=True,
                       loss_func='MSE',
                       freq='M'):
    """
    Train a single model and return comprehensive results including training curves
    and long-short portfolio analysis.

    Parameters:
    -----------
    test_dates : array-like, optional
        Time period identifier for each observation in test set (e.g., year-month).
        Required for portfolio analysis. Should be same length as y_test.
    freq : str, default 'M'
        Frequency of data for Sharpe ratio annualization. 'M' for monthly, 'D' for daily.
    """

    print(f"\n{'=' * 80}")
    print(f"MODEL: {model_name} | Target: {target_params:,} parameters")
    print('=' * 80)

    # Build model
    model, normalizer, actual_params, architecture = build_model_with_target_params(
        X_train.shape[1], target_params
    )
    normalizer.adapt(X_train)

    print(f"Architecture: {architecture}")
    print(f"Actual parameters: {actual_params:,}")

    # Calculate FLOPs per epoch (approximate)
    # FLOPs = 6 * params * samples (forward + backward + update)
    flops_per_epoch = 6 * actual_params * len(X_train)

    # Compile
    optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0)
    if loss_func == 'MSE':
        model.compile(loss='mean_squared_error', optimizer=optimizer)
    elif loss_func == 'IC':
        model.compile(loss=negative_ic_loss, optimizer=optimizer)

    # Callbacks
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss',  # val_loss
        factor=0.5,
        patience=epochs/3, #100
        min_lr=0.00000000001,
        verbose=1
    )

    progress = SingleLineProgress()
    callbacks = [progress, lr_scheduler]

    if show_live_plot:
        live_plot = LivePlotCallback()
        callbacks.append(live_plot)

    # Train
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=callbacks
    )
    train_time = time.time() - start_time

    # Calculate cumulative compute at each epoch (in PF-days)
    # 1 PetaFLOP-day = 10^15 FLOPS * 86400 seconds = 8.64e19 FLOPs
    cumulative_flops = [(i + 1) * flops_per_epoch for i in range(epochs)]
    cumulative_pf_days = [f / 8.64e19 for f in cumulative_flops]

    # Evaluate final performance
    train_loss = history.history['loss'][-1]
    val_loss = history.history['val_loss'][-1]

    test_pred = model.predict(X_test, verbose=0).flatten()
    test_mse = np.mean((y_test - test_pred) ** 2)

    # CORRECTED R² CALCULATION - Use expanding window mean (chronologically consistent)
    # Start with historical mean from train + val data
    historical_returns = np.concatenate([y_train, y_val])

    # For each test observation, calculate the expanding window mean using only past data
    expanding_means = np.zeros(len(y_test))
    cumsum = np.sum(historical_returns)
    count = len(historical_returns)

    for i in range(len(y_test)):
        # Mean using all data up to (but not including) observation i
        expanding_means[i] = cumsum / count
        # Add current observation to running total for next iteration
        cumsum += y_test[i]
        count += 1

    # R² using expanding window means as baseline (what we could have predicted)
    ss_res = np.sum((y_test - test_pred) ** 2)
    ss_tot = np.sum((y_test - expanding_means) ** 2)
    test_r2 = 1 - (ss_res / ss_tot)

    # ========================================================================
    # PORTFOLIO ANALYSIS
    # ========================================================================
    portfolio_stats = None
    decile_returns_df = None

    if test_dates is not None:
        # Create dataframe with predictions, actual returns, and dates
        test_df = pd.DataFrame({
            'date': test_dates,
            'prediction': test_pred,
            'actual_return': y_test
        })
        test_df.to_clipboard()

        # For each date, rank firms by prediction and assign to deciles
        def assign_deciles(group):
            n_stocks = len(group)
            if n_stocks < 10:
                # If fewer than 10 stocks, create fewer quantiles
                n_quantiles = min(n_stocks, 10)
                group['decile'] = pd.qcut(group['prediction'], q=n_quantiles,
                                          labels=False, duplicates='drop') + 1
            else:
                group['decile'] = pd.qcut(group['prediction'], q=10,
                                          labels=False, duplicates='drop') + 1
            return group

        test_df = test_df.groupby('date', group_keys=False).apply(assign_deciles)

        # Calculate Forecast_Weighted portfolio
        def calc_forecast_weighted(group):
            predictions = group['prediction'].values
            actual_returns = group['actual_return'].values

            # Calculate weights: w_i = y_hat_i / sum(|y_hat_j|)
            sum_abs_pred = np.abs(predictions).sum()
            if sum_abs_pred > 0:
                weights = predictions / sum_abs_pred
            else:
                weights = np.zeros(len(predictions))

            # Portfolio return is weighted sum of actual returns
            portfolio_return = (weights * actual_returns).sum()
            return portfolio_return

        forecast_weighted_returns = test_df.groupby('date').apply(calc_forecast_weighted)

        # Calculate equal-weighted returns for each decile in each period
        decile_returns = test_df.groupby(['date', 'decile'])['actual_return'].mean().unstack(fill_value=np.nan)

        # Ensure all deciles 1-10 are present
        for i in range(1, 11):
            if i not in decile_returns.columns:
                decile_returns[i] = np.nan

        decile_returns = decile_returns[[i for i in range(1, 11)]]
        decile_returns.columns = [f'D{i}' for i in range(1, 11)]

        # Add Forecast_Weighted portfolio to decile_returns
        decile_returns['Forecast_Weighted'] = forecast_weighted_returns

        # Calculate long-short portfolios
        # Top 50% (deciles 6-10) minus Bottom 50% (deciles 1-5)
        top50 = decile_returns[['D6', 'D7', 'D8', 'D9', 'D10']].mean(axis=1)
        bottom50 = decile_returns[['D1', 'D2', 'D3', 'D4', 'D5']].mean(axis=1)
        LS_50 = top50 - bottom50

        # Top 30% (deciles 8-10) minus Bottom 30% (deciles 1-3)
        top30 = decile_returns[['D8', 'D9', 'D10']].mean(axis=1)
        bottom30 = decile_returns[['D1', 'D2', 'D3']].mean(axis=1)
        LS_30 = top30 - bottom30

        # Top 10% (decile 10) minus Bottom 10% (decile 1)
        LS_10 = decile_returns['D10'] - decile_returns['D1']

        # Add LS portfolios to dataframe
        decile_returns['LS_50'] = LS_50
        decile_returns['LS_30'] = LS_30
        decile_returns['LS_10'] = LS_10

        # Annualization factor based on frequency
        if freq == 'M':
            ann_factor = np.sqrt(12)
            ann_periods = 12
        elif freq == 'D':
            ann_factor = np.sqrt(252)
            ann_periods = 252
        else:
            ann_factor = 1
            ann_periods = 1

        # Calculate statistics for each LS portfolio AND Forecast_Weighted
        ls_stats = {}
        for ls_name, ls_label in [('LS_10', '10% Breakpoint'),
                                  ('LS_30', '30% Breakpoint'),
                                  ('LS_50', '50% Breakpoint'),
                                  ('Forecast_Weighted', 'Forecast Weighted')]:
            returns = decile_returns[ls_name].dropna()
            mean_ret = returns.mean() * ann_periods  # Annualized mean
            std_ret = returns.std() * ann_factor  # Annualized vol
            sharpe = (mean_ret / std_ret) if std_ret > 0 else 0

            ls_stats[ls_name] = {
                'mean': float(mean_ret),
                'std': float(std_ret),
                'sharpe': float(sharpe),
                'label': ls_label
            }

        # Print in 4-column grid format (added Forecast_Weighted)
        print(f"\n{'=' * 95}")
        print("LONG-SHORT PORTFOLIO ANALYSIS")
        print('=' * 95)
        print(f"{'Metric':<15} {'10% Breakpoint':>18} {'30% Breakpoint':>18} "
              f"{'50% Breakpoint':>18} {'Forecast Weighted':>18}")
        print('─' * 95)
        print(f"{'Ann. Mean':<15} {ls_stats['LS_10']['mean']:>17.4f} "
              f"{ls_stats['LS_30']['mean']:>17.4f} {ls_stats['LS_50']['mean']:>17.4f} "
              f"{ls_stats['Forecast_Weighted']['mean']:>17.4f}")
        print(f"{'Ann. Std Dev':<15} {ls_stats['LS_10']['std']:>17.4f} "
              f"{ls_stats['LS_30']['std']:>17.4f} {ls_stats['LS_50']['std']:>17.4f} "
              f"{ls_stats['Forecast_Weighted']['std']:>17.4f}")
        print(f"{'Ann. Sharpe':<15} {ls_stats['LS_10']['sharpe']:>17.4f} "
              f"{ls_stats['LS_30']['sharpe']:>17.4f} {ls_stats['LS_50']['sharpe']:>17.4f} "
              f"{ls_stats['Forecast_Weighted']['sharpe']:>17.4f}")
        print('=' * 95)

        portfolio_stats = ls_stats
        decile_returns_df = decile_returns

    # Total compute
    total_flops = cumulative_flops[-1]
    total_pf_days = cumulative_pf_days[-1]

    print(f"\nResults: Train={train_loss:.6f} | Val={val_loss:.6f} | "
          f"Test={test_mse:.6f} (R²={test_r2:.4f})")
    print(f"Time: {train_time:.1f}s | Compute: {total_pf_days:.2e} PF-days")

    # Clear GPU memory
    tf.keras.backend.clear_session()

    results_dict = {
        'model_name': model_name,
        'target_params': target_params,
        'actual_params': actual_params,
        'architecture': architecture,
        'train_loss': float(train_loss),
        'val_loss': float(val_loss),
        'test_loss': float(test_mse),
        'test_r2': float(test_r2),
        'train_time': float(train_time),
        'total_flops': float(total_flops),
        'pf_days': float(total_pf_days),
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'flops_per_epoch': float(flops_per_epoch),
        # Store full training curves with compute
        'training_curve': {
            'epochs': list(range(1, epochs + 1)),
            'train_loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']],
            'cumulative_flops': cumulative_flops,
            'cumulative_pf_days': cumulative_pf_days
        }
    }

    # Add portfolio results if available
    if portfolio_stats is not None:
        results_dict['portfolio_stats'] = portfolio_stats
        results_dict['decile_returns'] = decile_returns_df

    return results_dict


# ============================================================================
# MAIN SCALING EXPERIMENT
# ============================================================================

def run_scaling_experiment(df, char_cols, target_col='xret',
                           param_sizes=None,
                           stop_at_size=None,
                           test_size=0.2,
                           val_size=0.125,
                           epochs=500,
                           min_batch_size=8192,
                           learning_rate=0.001,
                           output_dir='/Users/lukavulicevic/Desktop/ScalingLaws/Output/',
                           show_live_plots=True,
                           random_state=42,
                           loss_func='MSE'):
    """
    Run complete scaling law experiment.
    epochs can be int (constant) or callable function(param_size) -> epochs
    """

    # Default parameter sizes
    if param_sizes is None:
        param_sizes = ['10', '100', '500', '1K', '2K', '5K', '10K', '50K']

    # Convert to integers
    def parse_size(s):
        if isinstance(s, int):
            return s
        s = str(s).upper()
        if 'M' in s:
            return int(float(s.replace('M', '')) * 1_000_000)
        elif 'K' in s:
            return int(float(s.replace('K', '')) * 1_000)
        else:
            return int(s)

    param_sizes_int = [parse_size(s) for s in param_sizes]

    # Determine where to stop
    if stop_at_size is not None:
        stop_at_int = parse_size(stop_at_size)
        param_sizes_int = [p for p in param_sizes_int if p <= stop_at_int]

    # Handle epochs - can be int or callable
    if callable(epochs):
        epochs_func = epochs
    else:
        epochs_func = lambda size: epochs

    print(f"\n{'=' * 80}")
    print("SCALING LAWS EXPERIMENT CONFIGURATION")
    print('=' * 80)
    print(f"Model sizes to test: {len(param_sizes_int)}")
    print(f"Range: {param_sizes_int[0]:,} to {param_sizes_int[-1]:,} parameters")
    print(f"Epochs: {'Variable by size' if callable(epochs) else epochs}")
    print(f"Loss function: {loss_func}")
    print(f"Show live plots: {show_live_plots}")
    print(f"Output directory: {output_dir}")
    print('=' * 80)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Prepare data
    print("\nPreparing data...")
    model_data = df[char_cols + [target_col, 'date']].dropna()
    model_data = model_data.sort_values('date')

    unique_dates = sorted(model_data['date'].unique())
    n_dates = len(unique_dates)

    train_end_idx = int(n_dates * (1 - test_size))
    train_dates = unique_dates[:train_end_idx]
    test_dates = unique_dates[train_end_idx:]

    train_val_split_idx = int(len(train_dates) * (1 - val_size))
    val_dates = train_dates[train_val_split_idx:]
    train_dates = train_dates[:train_val_split_idx]

    train_data = model_data[model_data['date'].isin(train_dates)]
    val_data = model_data[model_data['date'].isin(val_dates)]
    test_data = model_data[model_data['date'].isin(test_dates)]

    X_train = train_data[char_cols].values
    y_train = train_data[target_col].values
    X_val = val_data[char_cols].values
    y_val = val_data[target_col].values
    X_test = test_data[char_cols].values
    y_test = test_data[target_col].values

    # Extract test dates for portfolio analysis
    test_dates_array = test_data['date'].values

    print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    print(f"Test period: {test_dates_array.min()} to {test_dates_array.max()}")

    # Set seeds
    np.random.seed(random_state)
    tf.random.set_seed(random_state)

    # Initialize storage for portfolio returns (wide format)
    portfolio_returns_dict = {}

    # Train models
    results = []
    experiment_start = time.time()

    for i, size in enumerate(param_sizes_int):
        model_epochs = epochs_func(size)

        model_batch_size = get_scaled_batch_size(size, min_batch=min_batch_size)
        model_lr = get_scaled_learning_rate(size, model_batch_size)

        print(f"\n{'#' * 80}")
        print(f"PROGRESS: [{i + 1}/{len(param_sizes_int)}] - {(i + 1) / len(param_sizes_int) * 100:.1f}% Complete")
        print(f"Model Size: {size:,} params | Epochs: {model_epochs}")
        print(f"{'#' * 80}")

        try:
            result = train_single_model(
                X_train, y_train, X_val, y_val, X_test, y_test,
                target_params=size,
                test_dates=test_dates_array,
                epochs=model_epochs,
                batch_size=model_batch_size,
                learning_rate=model_lr,
                model_name=f"model_{size}",
                show_live_plot=show_live_plots,
                loss_func=loss_func,
                freq='M'
            )

            results.append(result)

            # Store portfolio returns in wide format
            if 'decile_returns' in result:
                decile_returns = result['decile_returns']
                actual_params = result['actual_params']

                # Add columns with prefix for this model
                for col in decile_returns.columns:
                    col_name = f"{actual_params}_{col}"
                    portfolio_returns_dict[col_name] = decile_returns[col].values

                # Store dates (only need to do this once)
                if 'date' not in portfolio_returns_dict:
                    portfolio_returns_dict['date'] = decile_returns.index.values

            # Save intermediate results after each model
            # Remove decile_returns from results before saving to JSON (too large)
            results_for_json = []
            for r in results:
                r_copy = r.copy()
                if 'decile_returns' in r_copy:
                    del r_copy['decile_returns']
                results_for_json.append(r_copy)

            with open(output_path / 'scaling_results.json', 'w') as f:
                json.dump(results_for_json, f, indent=2)

            with open(output_path / 'scaling_results.pkl', 'wb') as f:
                pickle.dump(results, f)

            # Save portfolio returns CSV
            if len(portfolio_returns_dict) > 1:  # More than just 'date'
                portfolio_df = pd.DataFrame(portfolio_returns_dict)
                portfolio_df = portfolio_df.set_index('date')
                portfolio_df.index.name = 'date'
                portfolio_df.to_csv(output_path / 'portfolio_returns.csv')
                print(f"✓ Results saved to {output_path}")
                print(f"✓ Portfolio returns: {len(portfolio_df)} periods × {len(portfolio_df.columns)} series")

        except Exception as e:
            print(f"✗ ERROR training {size} parameter model: {e}")
            import traceback
            traceback.print_exc()
            print("Continuing to next model...")
            continue

    total_time = time.time() - experiment_start
    print(f"\n{'=' * 80}")
    print(f"EXPERIMENT COMPLETE")
    print(f"Total time: {total_time / 3600:.2f} hours")
    print(f"Models trained: {len(results)}/{len(param_sizes_int)}")
    print(f"Results saved to: {output_path}")
    print('=' * 80)

    return results
# ============================================================================
# PLOTTING
# ============================================================================

def plot_scaling_curves(results_path='/Users/lukavulicevic/Desktop/ScalingLaws/Output/',
                        loss_type='val_loss',
                        x_axis='compute',
                        figsize=(14, 9),
                        title=None,
                        save_name='scaling_curves.png',
                        dpi=300,
                        show_final_points=True,
                        fit_scaling_law=True):
    """
    Create scaling law plot showing CURVES (lines) for each model.
    """

    # Load results
    results_path = Path(results_path)
    try:
        with open(results_path / 'scaling_results.pkl', 'rb') as f:
            results = pickle.load(f)
    except:
        with open(results_path / 'scaling_results.json', 'r') as f:
            results = json.load(f)

    print(f"\nLoaded {len(results)} models")

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')

    # Color map based on number of parameters
    all_params = [r['actual_params'] for r in results]
    norm = plt.matplotlib.colors.LogNorm(vmin=min(all_params), vmax=max(all_params))
    cmap = plt.cm.viridis

    # Extract final values for scaling law fit
    final_x_values = []
    final_y_values = []

    # Plot each model's training curve
    for result in results:
        params = result['actual_params']
        curve = result['training_curve']

        # Get x and y data
        if x_axis == 'compute':
            x_data = curve['cumulative_pf_days']
            x_label = 'Compute (PetaFLOP-days)'
        else:  # epochs
            x_data = curve['epochs']
            x_label = 'Epoch'

        y_data = curve[loss_type]

        # Get color based on parameter count
        color = cmap(norm(params))

        # Plot line
        ax.plot(x_data, y_data, color=color, linewidth=3.5, alpha=0.8)

        # Mark final point
        if show_final_points:
            ax.scatter(x_data[-1], y_data[-1], color=color, s=100,
                       edgecolors='black', linewidth=1.5, zorder=5)

        # Store final values for fitting
        final_x_values.append(x_data[-1])
        final_y_values.append(y_data[-1])

    # Fit scaling law to final values
    if fit_scaling_law and len(final_x_values) > 3:
        try:
            final_x_values = np.array(final_x_values)
            final_y_values = np.array(final_y_values)

            print(f"\nDiagnostic info:")
            print(f"  Final X values: min={final_x_values.min():.2e}, max={final_x_values.max():.2e}")
            print(f"  Final Y values: min={final_y_values.min():.4f}, max={final_y_values.max():.4f}")
            print(f"  Y range: {final_y_values.max() - final_y_values.min():.4f}")

            # Define the model: L = L_inf + a * C^b
            def scaling_law(C, L_inf, a, b):
                return L_inf + a * np.power(C, b)

            # Initial guess for parameters
            L_inf_guess = np.min(final_y_values) * 0.9  # 90% of minimum
            y_range = np.max(final_y_values) - np.min(final_y_values)
            a_guess = y_range * (np.max(final_x_values) ** 0.2)
            b_guess = -0.2

            print(f"  Initial guesses: L_inf={L_inf_guess:.4f}, a={a_guess:.4e}, b={b_guess:.4f}")

            # Set bounds to ensure physical solutions
            # L_inf should be between 0 and min(y)
            # a should be positive
            # b should be negative (for scaling laws)
            bounds = (
                [0, 1e-10, -2.0],  # lower bounds
                [np.min(final_y_values), np.inf, 0]  # upper bounds
            )

            # Fit the model
            popt, pcov = curve_fit(scaling_law, final_x_values, final_y_values,
                                   p0=[L_inf_guess, a_guess, b_guess],
                                   bounds=bounds,
                                   maxfev=10000)

            L_inf, a, b = popt

            print(f"  Fitted parameters: L_inf={L_inf:.4f}, a={a:.4e}, b={b:.4f}")

            # Check for valid parameters
            if a <= 0 or b == 0 or np.isnan(a) or np.isnan(b) or np.isnan(L_inf):
                raise ValueError(f"Invalid fitted parameters: L_inf={L_inf}, a={a}, b={b}")

            # Convert to (C/C0)^b form
            # Since a * C^b = (C/C0)^b, we have C0 = (1/a)^(1/b)
            C0 = np.power(1.0 / a, 1.0 / b)

            if np.isnan(C0) or np.isinf(C0):
                raise ValueError(f"C0 calculation resulted in NaN or Inf: C0={C0}")

            # Format C0 as "mantissa × 10^exponent"
            exponent = int(np.floor(np.log10(abs(C0))))
            mantissa = C0 / (10 ** exponent)

            # Generate smooth curve through final points
            x_smooth = np.logspace(np.log10(final_x_values.min()),
                                   np.log10(final_x_values.max()), 100)
            y_smooth = scaling_law(x_smooth, L_inf, a, b)

            # Calculate R²
            y_pred = scaling_law(final_x_values, L_inf, a, b)
            ss_res = np.sum((final_y_values - y_pred) ** 2)
            ss_tot = np.sum((final_y_values - np.mean(final_y_values)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)

            # Plot scaling law with the new format
            ax.plot(x_smooth, y_smooth, 'r--', linewidth=3, alpha=0.9,
                    label=f'Scaling Law: $L = {L_inf:.3f} + \\left(\\frac{{C}}{{{mantissa:.1f} \\times 10^{{{exponent}}}}}\\right)^{{{b:.2f}}}$, $R^2$: {r_squared * 100:.1f}%',
                    zorder=10)

            print(f"\n✓ Successfully fitted scaling law:")
            print(f"  L_inf (irreducible loss): {L_inf:.4f}")
            print(f"  C0 (scale parameter): {mantissa:.1f} × 10^{exponent}")
            print(f"  b (exponent): {b:.4f}")
            print(f"  R²: {r_squared:.4f}")

        except Exception as e:
            print(f"\n✗ Could not fit scaling law: {e}")
            print(f"  This may happen when:")
            print(f"  - The loss values are too similar (not enough variation)")
            print(f"  - The data doesn't follow a power law")
            print(f"  - There aren't enough data points")
            print(f"  The plot will be generated without the fitted curve.")

    # Formatting
    ax.set_xscale('log')
    ax.set_yscale('log')  # Log-log scale
    ax.set_ylim(top=1e-1)  # Only set upper limit to 0.1
    ax.set_xlabel(x_label, fontsize=16, fontweight='bold')

    loss_names = {
        'train_loss': 'Training Loss',
        'val_loss': 'Validation Loss'
    }
    ax.set_ylabel(loss_names[loss_type], fontsize=16, fontweight='bold')

    if title is None:
        title = f'Neural Network Scaling Curves - {loss_names[loss_type]} vs {x_label}'
    ax.set_title(title, fontsize=18, fontweight='bold', pad=20)

    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.tick_params(labelsize=12)

    # Colorbar showing parameter count
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('Parameters', rotation=270, labelpad=25, fontsize=14, fontweight='bold')
    cbar.ax.tick_params(labelsize=11)

    # Legend for scaling law
    if fit_scaling_law:
        ax.legend(fontsize=13, loc='upper right', framealpha=0.9)

    plt.tight_layout()

    # Save
    save_path = results_path / save_name
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Plot saved to: {save_path}")

    plt.show()

    return fig, ax


def plot_final_performance(results_path='/Users/lukavulicevic/Desktop/ScalingLaws/Output/',
                           metric='test_loss',  # 'test_loss' or 'test_r2'
                           x_axis='params',  # 'params' or 'compute'
                           fit_curve=True,
                           figsize=(12, 8),
                           title=None,
                           save_name='test_performance.png',
                           dpi=300):
    """
    Plot final test performance vs model size.
    """

    # Load results
    results_path = Path(results_path)
    try:
        with open(results_path / 'scaling_results.pkl', 'rb') as f:
            results = pickle.load(f)
    except:
        with open(results_path / 'scaling_results.json', 'r') as f:
            results = json.load(f)

    print(f"\nLoaded {len(results)} models")

    # Extract data
    params = np.array([r['actual_params'] for r in results])
    metric_values = np.array([r[metric] for r in results])

    if x_axis == 'compute':
        x_data = np.array([r['pf_days'] for r in results])
        x_label = 'Compute (PetaFLOP-days)'
        x_var = 'C'
    else:  # params
        x_data = params
        x_label = 'Parameters'
        x_var = 'N'

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')

    # Plot points
    scatter = ax.scatter(x_data, metric_values, c=params, cmap='viridis',
                         s=200, alpha=0.7, edgecolors='black', linewidth=1.5,
                         norm=plt.matplotlib.colors.LogNorm())

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label('Parameters', rotation=270, labelpad=25, fontsize=14, fontweight='bold')
    cbar.ax.tick_params(labelsize=11)

    # Fit curve
    if fit_curve and len(x_data) > 3:
        try:
            print(f"\nDiagnostic info for {metric}:")
            print(f"  X values: min={x_data.min():.2e}, max={x_data.max():.2e}")
            print(f"  Y values: min={metric_values.min():.6f}, max={metric_values.max():.6f}")
            print(f"  Y range: {metric_values.max() - metric_values.min():.6f}")

            if metric == 'test_loss':
                # Power law + constant fit for loss: L = L_inf + a * x^b
                def scaling_law(x, L_inf, a, b):
                    return L_inf + a * np.power(x, b)

                # Initial guess for parameters
                L_inf_guess = np.min(metric_values) * 0.9
                y_range = np.max(metric_values) - np.min(metric_values)
                a_guess = y_range * (np.max(x_data) ** 0.2)
                b_guess = -0.2

                print(f"  Initial guesses: L_inf={L_inf_guess:.6f}, a={a_guess:.4e}, b={b_guess:.4f}")

                # Set bounds
                bounds = (
                    [0, 1e-20, -2.0],  # lower bounds
                    [np.min(metric_values), np.inf, 0]  # upper bounds
                )

                # Fit the model
                popt, pcov = curve_fit(scaling_law, x_data, metric_values,
                                       p0=[L_inf_guess, a_guess, b_guess],
                                       bounds=bounds,
                                       maxfev=10000)

                L_inf, a, b = popt

                print(f"  Fitted parameters: L_inf={L_inf:.6f}, a={a:.4e}, b={b:.4f}")

                # Check for valid parameters
                if a <= 0 or b == 0 or np.isnan(a) or np.isnan(b) or np.isnan(L_inf):
                    raise ValueError(f"Invalid fitted parameters: L_inf={L_inf}, a={a}, b={b}")

                # Convert to (x/x0)^b form
                x0 = np.power(1.0 / a, 1.0 / b)

                if np.isnan(x0) or np.isinf(x0):
                    raise ValueError(f"x0 calculation resulted in NaN or Inf: x0={x0}")

                # Format x0 as "mantissa × 10^exponent"
                exponent = int(np.floor(np.log10(abs(x0))))
                mantissa = x0 / (10 ** exponent)

                # Generate smooth curve
                x_smooth = np.logspace(np.log10(x_data.min()), np.log10(x_data.max()), 100)
                y_smooth = scaling_law(x_smooth, L_inf, a, b)

                # Calculate R²
                y_pred = scaling_law(x_data, L_inf, a, b)
                ss_res = np.sum((metric_values - y_pred) ** 2)
                ss_tot = np.sum((metric_values - np.mean(metric_values)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)

                ax.plot(x_smooth, y_smooth, 'r--', linewidth=3, alpha=0.8,
                        label=f'Scaling Law: $L = {L_inf:.3f} + \\left(\\frac{{{x_var}}}{{{mantissa:.1f} \\times 10^{{{exponent}}}}}\\right)^{{{b:.2f}}}$, $R^2$: {r_squared * 100:.1f}%')

                print(f"\n✓ Successfully fitted scaling law:")
                print(f"  L_inf: {L_inf:.6f}")
                print(f"  x0: {mantissa:.1f} × 10^{exponent}")
                print(f"  b: {b:.4f}")
                print(f"  R²: {r_squared:.4f}")

            else:  # test_r2
                # For R², fit a logarithmic trend: R² = a + b * log10(x)
                log_x = np.log10(x_data)
                coeffs = np.polyfit(log_x, metric_values, 1)
                slope = coeffs[0]
                intercept = coeffs[1]

                x_smooth = np.logspace(np.log10(x_data.min()), np.log10(x_data.max()), 100)
                y_smooth = intercept + slope * np.log10(x_smooth)

                # Calculate R²
                y_pred = intercept + slope * log_x
                ss_res = np.sum((metric_values - y_pred) ** 2)
                ss_tot = np.sum((metric_values - np.mean(metric_values)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)

                ax.plot(x_smooth, y_smooth, 'r--', linewidth=3, alpha=0.8,
                        label=f'Trend: $R^2 = {intercept:.3f} + {slope:.3f} \\log_{{10}}({x_var})$, Fit $R^2$: {r_squared * 100:.1f}%')

                print(f"\n✓ Successfully fitted R² trend:")
                print(f"  Intercept: {intercept:.4f}")
                print(f"  Slope: {slope:.4f}")
                print(f"  Fit R²: {r_squared:.4f}")

        except Exception as e:
            print(f"\n✗ Could not fit curve: {e}")
            print(f"  This may happen when:")
            print(f"  - The values are too similar (not enough variation)")
            print(f"  - The data doesn't follow the expected trend")
            print(f"  - There aren't enough data points")
            print(f"  The plot will be generated without the fitted curve.")

    # Formatting
    ax.set_xscale('log')
    if metric == 'test_loss':
        ax.set_yscale('log')

    ax.set_xlabel(x_label, fontsize=16, fontweight='bold')

    metric_labels = {
        'test_loss': 'Test Loss (MSE)',
        'test_r2': 'Test R²'
    }
    ax.set_ylabel(metric_labels[metric], fontsize=16, fontweight='bold')

    if title is None:
        title = f'{metric_labels[metric]} vs Model Size'
    ax.set_title(title, fontsize=18, fontweight='bold', pad=20)

    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.tick_params(labelsize=12)

    if fit_curve:
        ax.legend(fontsize=13, loc='best', framealpha=0.9)

    plt.tight_layout()

    save_path = results_path / save_name
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"✓ Plot saved to: {save_path}")

    plt.show()

    return fig, ax


def plot_sharpe_ratio_scaling(results_path='/Users/lukavulicevic/Desktop/ScalingLaws/Output/',
                              breakpoint='50',  # '10', '30', or '50' - DEFAULT IS 50
                              x_axis='compute',  # 'compute' or 'params'
                              fit_curve=True,
                              figsize=(12, 8),
                              title=None,
                              save_name=None,
                              dpi=300):
    """
    Plot Sharpe ratio vs model size/compute for long-short portfolios.

    Parameters:
    -----------
    results_path : str
        Path to results directory
    breakpoint : str
        Which breakpoint to plot: '10', '30', or '50' (default: '50')
    x_axis : str
        'compute' or 'params'
    fit_curve : bool
        Whether to fit a scaling law
    """

    # Load results
    results_path = Path(results_path)
    try:
        with open(results_path / 'scaling_results.pkl', 'rb') as f:
            results = pickle.load(f)
    except:
        with open(results_path / 'scaling_results.json', 'r') as f:
            results = json.load(f)

    print(f"\nLoaded {len(results)} models")

    # Map breakpoint to key name
    breakpoint_key = f'LS_{breakpoint}'
    breakpoint_label = f'{breakpoint}% Breakpoint'

    # Extract data - only include models that have portfolio stats
    params_list = []
    sharpe_list = []
    compute_list = []

    for r in results:
        if 'portfolio_stats' in r and breakpoint_key in r['portfolio_stats']:
            params_list.append(r['actual_params'])
            sharpe_list.append(r['portfolio_stats'][breakpoint_key]['sharpe'])
            compute_list.append(r['pf_days'])

    if len(params_list) == 0:
        print("✗ No portfolio statistics found in results!")
        return None, None

    params = np.array(params_list)
    sharpe_values = np.array(sharpe_list)

    if x_axis == 'compute':
        x_data = np.array(compute_list)
        x_label = 'Compute (PetaFLOP-days)'
        x_var = 'C'
    else:  # params
        x_data = params
        x_label = 'Parameters'
        x_var = 'N'

    print(f"Plotting {len(x_data)} models for {breakpoint_label}")

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')

    # Plot points
    scatter = ax.scatter(x_data, sharpe_values, c=params, cmap='viridis',
                         s=200, alpha=0.7, edgecolors='black', linewidth=1.5,
                         norm=plt.matplotlib.colors.LogNorm())

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label('Parameters', rotation=270, labelpad=25, fontsize=14, fontweight='bold')
    cbar.ax.tick_params(labelsize=11)

    # Fit curve
    if fit_curve and len(x_data) > 3:
        try:
            print(f"\nDiagnostic info for Sharpe ratio:")
            print(f"  X values: min={x_data.min():.2e}, max={x_data.max():.2e}")
            print(f"  Sharpe values: min={sharpe_values.min():.4f}, max={sharpe_values.max():.4f}")
            print(f"  Sharpe range: {sharpe_values.max() - sharpe_values.min():.4f}")

            # For Sharpe ratio, we expect it to increase and plateau
            # Use: Sharpe = S_max - a * C^b (with b < 0)
            # This gives diminishing returns as compute increases

            def sharpe_scaling_law(x, S_max, a, b):
                """Sharpe approaches S_max as compute increases"""
                return S_max - a * np.power(x, b)

            # Initial guess for parameters
            S_max_guess = np.max(sharpe_values) * 1.1  # 110% of max
            sharpe_range = np.max(sharpe_values) - np.min(sharpe_values)
            a_guess = sharpe_range * (np.min(x_data) ** 0.2)
            b_guess = -0.2

            print(f"  Initial guesses: S_max={S_max_guess:.4f}, a={a_guess:.4e}, b={b_guess:.4f}")

            # Set bounds
            # S_max should be greater than max observed Sharpe
            # a should be positive
            # b should be negative (diminishing returns)
            bounds = (
                [np.max(sharpe_values), 1e-10, -2.0],  # lower bounds
                [np.max(sharpe_values) * 5, np.inf, -0.01]  # upper bounds
            )

            # Fit the model
            popt, pcov = curve_fit(sharpe_scaling_law, x_data, sharpe_values,
                                   p0=[S_max_guess, a_guess, b_guess],
                                   bounds=bounds,
                                   maxfev=10000)

            S_max, a, b = popt

            print(f"  Fitted parameters: S_max={S_max:.4f}, a={a:.4e}, b={b:.4f}")

            # Check for valid parameters
            if a <= 0 or b >= 0 or np.isnan(a) or np.isnan(b) or np.isnan(S_max):
                raise ValueError(f"Invalid fitted parameters: S_max={S_max}, a={a}, b={b}")

            # Convert to (C/C0)^b form
            # Since a * C^b = (C/C0)^b, we have C0 = (1/a)^(1/b)
            C0 = np.power(1.0 / a, 1.0 / b)

            if np.isnan(C0) or np.isinf(C0):
                raise ValueError(f"C0 calculation resulted in NaN or Inf: C0={C0}")

            # Format C0 as "mantissa × 10^exponent"
            exponent = int(np.floor(np.log10(abs(C0))))
            mantissa = C0 / (10 ** exponent)

            # Generate smooth curve
            x_smooth = np.logspace(np.log10(x_data.min()),
                                   np.log10(x_data.max()), 100)
            y_smooth = sharpe_scaling_law(x_smooth, S_max, a, b)

            # Calculate R²
            y_pred = sharpe_scaling_law(x_data, S_max, a, b)
            ss_res = np.sum((sharpe_values - y_pred) ** 2)
            ss_tot = np.sum((sharpe_values - np.mean(sharpe_values)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)

            # Plot scaling law
            ax.plot(x_smooth, y_smooth, 'r--', linewidth=3, alpha=0.8,
                    label=f'Scaling Law: $SR = {S_max:.2f} - \\left(\\frac{{{x_var}}}{{{mantissa:.1f} \\times 10^{{{exponent}}}}}\\right)^{{{b:.2f}}}$, $R^2$: {r_squared * 100:.1f}%')

            print(f"\n✓ Successfully fitted Sharpe ratio scaling law:")
            print(f"  S_max (asymptotic Sharpe): {S_max:.4f}")
            print(f"  C0 (scale parameter): {mantissa:.1f} × 10^{exponent}")
            print(f"  b (exponent): {b:.4f}")
            print(f"  R²: {r_squared:.4f}")

        except Exception as e:
            print(f"\n✗ Could not fit scaling law: {e}")
            print(f"  This may happen when:")
            print(f"  - The Sharpe values are too similar (not enough variation)")
            print(f"  - The data doesn't follow a power law")
            print(f"  - There aren't enough data points")
            print(f"  The plot will be generated without the fitted curve.")

    # Formatting
    ax.set_xscale('log')
    ax.set_xlabel(x_label, fontsize=16, fontweight='bold')
    ax.set_ylabel(f'Annualized Sharpe Ratio', fontsize=16, fontweight='bold')

    if title is None:
        title = f'Long-Short Portfolio Sharpe Ratio ({breakpoint_label}) vs {x_label}'
    ax.set_title(title, fontsize=18, fontweight='bold', pad=20)

    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.tick_params(labelsize=12)

    # Add horizontal line at Sharpe = 0
    ax.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)

    if fit_curve:
        ax.legend(fontsize=13, loc='best', framealpha=0.9)

    plt.tight_layout()

    # Save
    if save_name is None:
        save_name = f'sharpe_ratio_LS{breakpoint}_vs_{x_axis}.png'
    save_path = results_path / save_name
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"✓ Plot saved to: {save_path}")

    plt.show()

    return fig, ax

def create_all_plots(results_path='/Users/lukavulicevic/Desktop/ScalingLaws/Output/', dpi=300):
    """Create comprehensive set of plots."""

    print("\n" + "=" * 80)
    print("CREATING SCALING LAW VISUALIZATIONS")
    print("=" * 80)

    # Plot 1: Validation loss CURVES vs Compute (with scaling law)
    print("\n1. Validation Loss Training Curves vs Compute")
    plot_scaling_curves(
        results_path=results_path,
        loss_type='val_loss',
        x_axis='compute',
        save_name='scaling_curves_val_compute.png',
        fit_scaling_law=True,
        dpi=dpi
    )

    # # Plot 2: Training loss CURVES vs Compute (with scaling law)
    # print("\n2. Training Loss Training Curves vs Compute")
    # plot_scaling_curves(
    #     results_path=results_path,
    #     loss_type='train_loss',
    #     x_axis='compute',
    #     save_name='scaling_curves_train_compute.png',
    #     fit_scaling_law=True,
    #     dpi=dpi
    # )

    # Plot 5: Test Loss vs Compute
    print("\n5. Test Loss vs Compute")
    plot_final_performance(
        results_path=results_path,
        metric='test_loss',
        x_axis='compute',
        fit_curve=True,
        save_name='test_loss_vs_compute.png',
        dpi=dpi
    )

    # # Plot 6: Test R² vs Compute
    # print("\n6. Test R² vs Compute")
    # plot_final_performance(
    #     results_path=results_path,
    #     metric='test_r2',
    #     x_axis='compute',
    #     fit_curve=True,
    #     save_name='test_r2_vs_compute.png',
    #     dpi=dpi
    # )

    print("\n3. Sharpe Ratio vs Compute")
    plot_sharpe_ratio_scaling(
        results_path=results_path,
        breakpoint='50',  # Change to '10' or '30' if desired
        x_axis='compute',
        fit_curve=True,
        dpi=dpi
    )

    print("\n" + "=" * 80)
    print("ALL PLOTS CREATED SUCCESSFULLY")
    print("=" * 80)


def get_epochs(size):
    return max(int(8 * (size ** 0.3)),75)

def parse_size(size_str):
    """Convert '1K', '2M', etc. to actual numbers"""
    size_str = size_str.upper()
    if 'M' in size_str:
        return int(float(size_str.replace('M', '')) * 1_000_000)
    elif 'K' in size_str:
        return int(float(size_str.replace('K', '')) * 1_000)
    else:
        return int(size_str)

def get_scaled_batch_size(num_params, base_batch_size=512, min_batch=4096, max_batch=65536):
    """
    Scale batch size with model size.

    Rationale: Smaller models need smaller batches to:
    1. Avoid overfitting to sharp minima (Keskar et al., 2016)
    2. Maintain healthy parameter-to-batch ratio
    3. Allow sufficient gradient noise for regularization

    References:
    - Keskar et al. (2016): "On Large-Batch Training for Deep Learning"
    - Smith & Le (2018): "Don't Decay the Learning Rate, Increase the Batch Size"

    Rule: batch_size scales as sqrt(params), clamped to [min, max]
    """
    # Use sqrt scaling - keeps ratio of params/batch growing with model size
    scaled_batch = int(base_batch_size * np.sqrt(num_params / 10_000))

    # Clamp to reasonable range
    scaled_batch = max(min_batch, min(max_batch, scaled_batch))

    # Round to nearest power of 2 for GPU efficiency
    scaled_batch = 2 ** int(np.log2(scaled_batch))

    return scaled_batch


def get_scaled_learning_rate(num_params, batch_size,
                             base_lr=0.001, base_batch=512, base_params=10_000):
    """
    Scale learning rate based on both model size and batch size.

    Implements:
    - Linear scaling with batch size (Goyal et al., 2017)
    - Inverse sqrt scaling with model size (μP-inspired)

    References:
    - Goyal et al. (2017): "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour"
    - Yang & Hu (2021): "Tensor Programs V" (μP framework)
    """
    # Linear scaling with batch size (Goyal et al., 2017)
    batch_factor = batch_size / base_batch

    # Inverse sqrt scaling with model size (μP-inspired)
    size_factor = np.sqrt(base_params / num_params)

    # Combined scaling
    scaled_lr = base_lr * batch_factor * size_factor

    # Clamp to reasonable range
    scaled_lr = max(1e-5, min(0.1, scaled_lr))

    return scaled_lr

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    #####################################################################
    # 1. Load and prepare data
    #####################################################################
    print("\n" + "=" * 80)
    print("STEP 1: LOADING DATA")
    print("=" * 80)

    data = pd.read_csv("/Users/lukavulicevic/Desktop/ScalingLaws/Input/Freyberger_Neuhierl_Weber.csv")
    data = data[data['lme'] > 0]
    data = data[data['at'] > 0]
    data['log_at'] = np.log(data['at'])
    data['log_lme'] = np.log(data['lme'])

    data = get_excess_return(data)
    #data['xret'] = data['xret']
    df = data.copy()

    if not np.issubdtype(df["date"].dtype, np.datetime64):
        if {"yy", "mm"}.issubset(df.columns):
            df["date"] = pd.to_datetime(dict(year=df["yy"], month=df["mm"], day=1))
        else:
            df["date"] = pd.to_datetime(df["date"])

    #df = df[df["date"] >= "2000-01-01"]

    base_char_cols = [
        'beta', 'e2p', 'beme', 'q', 'cum_return_36_13', 'a2me', 'cum_return_1_0',
        'log_lme', 'cto', 'roe', 'cum_return_12_2', 'oa', 'lturnover', 'noa',
        'rel_to_high_price', 'idio_vol', 'ato', 'dpi2a', 'investment', 'pm',
        'rna', 'suv', 'roa', 'free_cf', 'ol', 'c', 'cum_return_12_7',
        'spread_mean', 'log_at', 'lev', 'prof', 's2p', 'sga2m', 'd2a',
        'fc2y', 'pcm'
    ]

    #####################################################################
    # 2. CREATE LAGGED FEATURES - EDIT THESE PARAMETERS
    #####################################################################

    N_ADDITIONAL_LAGS = 0  # Set to 2 for lags t-2, t-3; set to 11 for t-2 through t-12
    INCLUDE_HISTORICAL_AVG = True  # Set to True to include historical averages

    df, char_cols = create_lagged_features(
        df=df,
        base_char_cols=base_char_cols,
        n_additional_lags=N_ADDITIONAL_LAGS,
        include_historical_avg=INCLUDE_HISTORICAL_AVG,
        firm_id_col='permno'
    )

    print(f"✓ Data loaded: {len(df):,} observations")
    print(f"✓ Total features: {len(char_cols)}")

    #####################################################################
    # 3. Run scaling experiment - EDIT THESE VALUES
    #####################################################################

    PARAM_SIZES = [#'500',
                   '1K', '2K', '4K', '6K', '8K',
                   '10K', #'20K', '40K', '60K', '80K',
                   '100K', #'200K', '500K', '700K', '800K',
                   '1M', #'2M', '4M', '6M', '8M',
                   '10M', #'20M', '40M', '60M', '80M',
                   #'100M', '200M', '400M', '600M', '800M'
        ]

    # PARAM_SIZES = ['500',
    #                '1K', #'4K', '8K',
    #                '10K', #'50K',
    #                '100K', #'500K',
    #                '1M', #'2M',
    #                '10M', #'20M', '40M', '60M', '80M',
    #                '100M', #'200M', '400M', '600M', '800M'
    #                 ]

    STOP_AT = '100K'  # Change to control where to stop, or None for all
    #EPOCHS = 200
    LOSS_FUNCTION = 'MSE'  # 'MSE' or 'IC'
    SHOW_LIVE_PLOTS = True  # Set to False to disable live plotting during training

    #epoch_scaling
    data = []
    for size_str in PARAM_SIZES:
        size_num = parse_size(size_str)
        epochs = get_epochs(size_num)
        data.append({
            'Param Size': size_str,
            'Num Params': size_num,
            'Epochs': epochs
        })

    epochs = pd.DataFrame(data)
    print(epochs.to_string(index=False))
    epochs.to_clipboard()


    results = run_scaling_experiment(
        df=df,
        char_cols=char_cols,
        target_col='xret',
        param_sizes=PARAM_SIZES,
        stop_at_size=STOP_AT,
        epochs=get_epochs,
        min_batch_size=65536,#4096, #8192, #16384, #32768, #65536, #131072, #262144
        learning_rate=0.001, #0001, #001
        output_dir='/Users/lukavulicevic/Desktop/ScalingLaws/Output/',
        show_live_plots=SHOW_LIVE_PLOTS,
        random_state=42,
        loss_func=LOSS_FUNCTION
    )

    #####################################################################
    # 4. Create all plots
    #####################################################################

    create_all_plots(
        results_path='/Users/lukavulicevic/Desktop/ScalingLaws/Output/',
        dpi=300
    )

    print("\n" + "=" * 80)
    print("SCALING LAWS EXPERIMENT COMPLETE!")
    print("Check output directory for results and plots")
    print("=" * 80)
