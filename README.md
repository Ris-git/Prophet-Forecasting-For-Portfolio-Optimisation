# Prophet Forecasting for Portfolio Optimisation

## Project Overview
An end-to-end machine learning project that forecasts stock and asset prices (specifically **Nifty 50** constituents) using Facebook/Meta Prophet time series forecasting model, then applies Markowitz portfolio optimisation to rebalance portfolios based on these forecasts. The project operates with **INR (₹)** as the base currency.

_(Needless to say it's for illustrative purposes and not financial advice)._

**Live Application**: This project is deployed on Streamlit Community Cloud and is accessible at [https://prophet-portfolio-optimisation.streamlit.app/](https://prophet-portfolio-optimisation.streamlit.app/)

**Presentation Slides**: [Here](https://gamma.app/docs/Prophet-Forecasting-for-Portfolio-Optimisation-7qsgynwy1h5x3it) are slides to accompany this project.

## Components

### 1. Prophet (Time Series Forecasting)

**What is Prophet?**

Prophet is Facebook's open-source time series forecasting tool designed for business forecasting. It handles trends, seasonality, and holidays automatically, making it robust and easy to use for forecasting time series data.

**How It Works in This Project:**

- Input: Historical price time series with datetime index
- Model: Prophet fits additive components (trend, seasonality, holidays)
- Output: Forecasted prices for each asset in the portfolio for the next trading day
- Training: The model fits to historical price data and generates one-step-ahead forecasts

### 2. Markowitz Portfolio Optimisation

**What is Markowitz Portfolio Optimisation?**

Markowitz portfolio optimisation, also known as Modern Portfolio Theory (MPT), is a mathematical framework for constructing optimal portfolios. Developed by Harry Markowitz in 1952, it balances the trade-off between expected returns and risk.

**Key Concepts:**

- **Expected Return**: The weighted average of expected returns of individual assets
- **Risk (Volatility)**: Measured as the standard deviation of portfolio returns
- **Correlation**: How assets move relative to each other
- **Efficient Frontier**: The set of optimal portfolios offering the highest expected return for a given level of risk

**The Optimisation Problem:**

```
Maximize: μᵀw - λ(wᵀΣw)

Subject to:
- Σwᵢ = 1 (weights sum to 1)
- wᵢ ≥ 0 (long-only portfolio, optional)
- Additional constraints (sector limits, etc.)
```

Where:
- `μ` = vector of expected returns (from Prophet price forecasts)
- `Σ` = covariance matrix of asset returns
- `w` = portfolio weights
- `λ` = risk aversion parameter (configurable in `src/settings.py`)

**How It Works in This Project:**

1. **Input**: Forecasted returns (derived from Prophet price predictions) for each asset
2. **Risk Estimation**: Historical covariance matrix calculated from asset returns
3. **Optimisation**: Solves for optimal weights that maximise risk-adjusted returns using SciPy's SLSQP solver
4. **Output**: Recommended portfolio allocation (weights for each asset)
5. **Rebalancing**: Portfolio is rebalanced based on these optimal weights

## Deployment on Streamlit Cloud

To deploy this app on Streamlit Community Cloud:

1.  **Fork this Repository**
    - Click "Fork" in the top-right corner of this GitHub page.

2.  **Create a Streamlit Cloud Account**
    - Sign up at [share.streamlit.io](https://share.streamlit.io/).

3.  **Deploy the App**
    - Click "New app".
    - Select your forked repository.
    - Set **Branch** to `main` (or your default branch).
    - Set **Main file path** to `src/streamlit_app.py`.
    - Click "Deploy".

4.  **Configure Secrets**
    - Once deployed (or during deployment), go to the app dashboard "⋮" menu -> **Settings** -> **Secrets**.
    - Add your Supabase credentials so the app can read the predictions:

    ```toml
    [general]
    SUPABASE_URL = "your-supabase-project-url"
    SUPABASE_KEY = "your-supabase-anon-key"
    ```

    - Save the secrets. The app should now auto-reload and display data from your Supabase database.

## Project Workflow

```
Historical Data Extraction
    ↓
Data Preprocessing
    ↓
Prophet Model Training
    ↓
Price Forecasting
    ↓
Markowitz Optimisation
    ↓
Optimal Portfolio Weights
    ↓
Results Saved to Supabase
    ↓
Streamlit Dashboard on Streamlit Cloud
```
## Installation

### Standard Installation

```bash
# Install dependencies using Poetry
make install-dev

# Or manually
poetry install
```

### Requirements

- Python 3.12+
  - I recommend installing through [PyEnv](https://github.com/pyenv/pyenv)
  - PyEnv can be installed through [Brew](https://brew.sh/).
- Poetry
  - [Basic usage](https://python-poetry.org/docs/basic-usage/)
- CircleCI account
  - [Setup guide](https://circleci.com/blog/setting-up-continuous-integration-with-github/)
- Supabase account and project
  - [Starting guide](https://supabase.com/docs/guides/getting-started)
- [Streamlit Community Cloud](https://streamlit.io/cloud) account
  - Required for deployment

## Usage

### Basic Usage

```bash
poetry run python -m src.main
```

Or using the Makefile:

```bash
make run
```

### Configuration

Edit `src/settings.py` to customise:

- **Portfolio Tickers**: Modify `PORTFOLIO_TICKERS` list (Default: Nifty 50 constituents)
- **Risk Aversion**: Adjust `RISK_AVERSION` (higher = more risk averse)
- **Minimum Allocation**: Change `MINIMUM_ALLOCATION` (minimum weight per asset)
- **Date Range**: Update `START_DATE` and `END_DATE` for historical data

Example:

```python
# src/settings.py
PORTFOLIO_TICKERS = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"]
RISK_AVERSION = 5
MINIMUM_ALLOCATION = 0.0
START_DATE = "2020-01-01"
```

### Programmatic Usage

```python
from src.main import run_optimisation

result = run_optimisation(
    tickers=["RELIANCE.NS", "TCS.NS", "INFY.NS"],
    start_date="2023-01-01",
    end_date="2023-12-31"
)

print(f"Optimal Weights: {result['weights']}")
print(f"Predicted Returns: {result['predicted_returns']}")
print(f"Current Prices: {result['current_prices']}")
print(f"Prediction Date: {result['prediction_date']}")
```

### Running the Streamlit Dashboard

The Streamlit dashboard reads from Supabase to display historical predictions, portfolio weights, and performance metrics:

```bash
poetry run streamlit run src/streamlit_app.py
```

Or using the Makefile:

```bash
make dashboard
```

The dashboard allows you to:
- View portfolio weights and predictions for any date
- Analywe individual stock performance over time
- Compare predicted vs actual prices
- Track prediction accuracy metrics

**Note:** The dashboard requires Supabase to be configured and populated with data from previous optimization runs.

