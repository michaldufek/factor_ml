import pandas as pd
import numpy as np
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from utils import calculate_risk_free_rate, add_benchmark, proces_results, calculate_performance_portfolio_metrics, make_backtest

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train and evaluate financial models.')
    parser.add_argument('--frequency', type=str, default='M', help='Frequency for backtest')
    parser.add_argument('--backtest_period', type=str, default='2023-01-01', help='Start date for the backtest period')
    parser.add_argument('--num_stocks', type=int, default=6, help='Number of stocks in the portfolio')
    parser.add_argument('--long_only', action='store_true', help='Flag for long-only portfolio')
    parser.add_argument('--simul_random_backtest', action='store_true', help='Flag for simulating random backtest')
    parser.add_argument('--exposure', type=float, default=0.5, help='Exposure of the portfolio')
    parser.add_argument('--leverage', type=float, default=1.5, help='Leverage of the portfolio')
    return parser.parse_args()


def scale(pdf):
    '''User-defined-function for vectorized crosssectional scaling'''
    excluded_cols = ['Date', 'Symbol']
    scaled_cols = [col for col in pdf.columns if col not in excluded_cols]
    if pdf['Symbol'].nunique() > 1:  # Cross-sectional scaling requires more than 1 stock
        scaler = StandardScaler()
        to_scaling = pdf[scaled_cols]
        scaled = pd.DataFrame(scaler.fit_transform(to_scaling), columns=scaled_cols)
        out = pd.concat([pdf[excluded_cols].reset_index(drop=True), scaled], axis='columns')
        return out.reindex(pdf.columns, axis='columns')
    return pdf

def train_model(X_train, y_train, algorithm):
    '''Function to train the model based on the selected algorithm'''
    if algorithm == "ridge":
        model = RidgeCV(alphas=[0.1, 1.0, 100.0], cv=10)
    elif algorithm == "lasso":
        model = LassoCV(alphas=np.logspace(-6, 6, 13), cv=5, random_state=42)
    elif algorithm == "elasticnet":
        model = ElasticNetCV(alphas=np.logspace(-4, 4, 10), l1_ratio=[.1, .5, .7, .9, .95, .99, 1], cv=5, random_state=42)
    elif algorithm == "xgboost":
        model = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1, max_depth=3, alpha=10, n_estimators=100, random_state=42)
    else:
        raise ValueError("Invalid algorithm choice")

    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    '''Function to evaluate the trained model'''
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    return mse, rmse, r2

def fixed_period():
    args = parse_arguments()

    # Processing of input parameters
    frequency = args.frequency
    backtest_period = args.backtest_period
    number_of_stocks = args.num_stocks
    long_only = args.long_only
    simul_random_backtest = args.simul_random_backtest
    exposure = args.exposure
    leverage = args.leverage

    # Load and preprocess data
    captions = pd.read_csv('data/feats_captions.csv').drop('Index', axis=1).values.squeeze().tolist()
    df_filled = pd.read_csv('data/features.csv').dropna()
    normalized_df = df_filled.groupby('Date').apply(scale).reset_index(drop=True)

    # Data Splitting
    start, insample = "2010-12-31", "2019-12-31"
    X, y = normalized_df.iloc[:, 3:], normalized_df.iloc[:, 0]
    X_train, y_train = X[(normalized_df.Date > start) & (normalized_df.Date <= insample)], y[(normalized_df.Date > start) & (normalized_df.Date <= insample)]
    X_test, y_test = X[normalized_df.Date > insample], y[normalized_df.Date > insample]

    # Model Training and Evaluation
    algorithm = "xgboost"  # Options: "ridge", "lasso", "elasticnet", "xgboost"
    model = train_model(X_train, y_train, algorithm)
    mse, rmse, r2 = evaluate_model(model, X_test, y_test)
    print(f"Algorithm: {algorithm}, MSE: {mse}, RMSE: {rmse}, R-squared: {r2}")

    # Additional Processing...
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    predictions = pd.DataFrame(y_pred)

    # Prepare data for testing perions assesment
    market_data = df_filled[df_filled.Date > insample].iloc[:, :3]

    remove_big_techs = False

    longs_share = (leverage + exposure) / 2
    shorts_share = leverage -longs_share
    print(f'long rate: {longs_share} and short rate: {shorts_share} and leverage  {leverage}')

    risk_free_rate = calculate_risk_free_rate(frequency)

    predictions.reset_index(drop=True,inplace=True)
    market_data.reset_index(drop=True,inplace=True)
    rebalances = pd.concat([market_data, predictions], axis='columns')
    rebalances.columns = ['Log_Return', 'Date', 'Symbol', 'Score']

    # Remove bg tech companies
    if remove_big_techs:
        big_techs = ['AAPL', 'MSFT', 'GOOG', 'GOOGL' 'META', 'AMZN']
        rebalances = rebalances[~rebalances.Symbol.isin(big_techs)]

    rebalances_for_backtest = rebalances.loc[rebalances.Date >= backtest_period]
    backtest_data = rebalances_for_backtest.groupby('Date', as_index=False).apply(lambda x: make_backtest(x, leverage, number_of_stocks, longs_share,shorts_share)).reset_index(drop=True)
    backtest_data = add_benchmark(backtest_data, start_from=backtest_period)

    pivot_df = rebalances.pivot_table(index='Date', columns='Symbol', values=['Log_Return', 'Score']).reset_index().dropna()
    pivot_df.columns = [f"{col[0]}_{col[1]}" for col in pivot_df.columns]
    pivot_df['Average_Log_Return'] = pivot_df.filter(like='Log_Return').mean(axis=1)

    equity_curves = proces_results(backtest_data).assign(Date=lambda df: pd.to_datetime(df['Date']))
    metrics = calculate_performance_portfolio_metrics(equity_curves, risk_free_rate, frequency)

    print(metrics)
    print(equity_curves)



if __name__ == "__main__":
    fixed_period()

