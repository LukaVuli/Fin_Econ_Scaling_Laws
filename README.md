# Scaling Laws of Finance and Economics

## Overview
This project studies how neural network size affects stock return prediction performance - similar to scaling laws in language models, but applied to finance.

## What Does This Do?

**Returns.py** trains neural networks of different sizes (1K to 10M+ parameters) to predict stock returns, then analyzes:
- How prediction accuracy improves with model size
- How much compute is needed for each level of performance
- Whether larger models produce better trading strategies (measured by Sharpe ratios)

## Key Functions

### Data Preparation
- `get_excess_return()` - Calculates returns above the risk-free rate
- `create_lagged_features()` - Creates historical features (past months' data)

### Model Training
- `build_model_with_target_params()` - Builds networks of specific sizes
- `train_single_model()` - Trains one model and evaluates portfolio performance
- `run_scaling_experiment()` - Trains many models of different sizes

### Analysis & Visualization
- `plot_scaling_curves()` - Shows how loss decreases with compute
- `plot_final_performance()` - Test loss vs model size
- `plot_sharpe_ratio_scaling()` - Trading performance vs model size

## Configuration

Edit these parameters in the main section:

```python
N_ADDITIONAL_LAGS = 0          # Use past months as features
INCLUDE_HISTORICAL_AVG = True  # Include historical averages
PARAM_SIZES = ['1K', '10K', '100K', '1M', '10M']  # Model sizes
LOSS_FUNCTION = 'MSE'          # Loss function
```

## Output

- `scaling_results.json` - Model performance metrics
- `portfolio_returns.csv` - Returns for each model's trading strategy
- PNG plots showing scaling relationships

## Usage

```bash
python Neural_Network_Training/Returns.py
```