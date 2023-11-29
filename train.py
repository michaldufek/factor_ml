import pandas as pd
import numpy as np
import argparse
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from datetime import timedelta
from utils import calculate_risk_free_rate, add_benchmark, proces_results, calculate_performance_portfolio_metrics, make_backtest

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train and evaluate multi-factor models.')
    parser.add_argument('--frequency', type=str, default='M', help='Frequency for backtest')
    parser.add_argument('--backtest_period', type=str, default='2023-01-01', help='Start date for the backtest period')
    parser.add_argument('--num_stocks', type=int, default=6, help='Number of stocks in the portfolio')
    parser.add_argument('--long_only', action='store_true', help='Flag for long-only portfolio')
    parser.add_argument('--simul_random_backtest', action='store_true', help='Flag for simulating random backtest')
    parser.add_argument('--exposure', type=float, default=0.5, help='Exposure of the portfolio')
    parser.add_argument('--leverage', type=float, default=1.5, help='Leverage of the portfolio')
    model_types = ["xgboost", "ridge", "lasso"]
    parser.add_argument('--model', type=str, default="xgboost", choices=model_types, help=f'ML algorithm for training. Eligible options: {model_types}')
    parser.add_argument('--start', type=str, default="2005-12-31", help='Start of the training period')
    parser.add_argument('--test_start', type=str, default="2019-12-31", help='Start of the testing period. Relevant just for fixed_period.')
    training_modes = ["fixed_period", "walk_forward"]
    parser.add_argument('--training_mode', type=str, default="fixed_period", choices=training_modes, help="Mode of training: fixed_period or walk_forward")

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

def load_and_preprocess_data():
    captions = pd.read_csv('data/feats_captions.csv').drop('Index', axis=1).values.squeeze().tolist()
    df_filled = pd.read_csv('data/features.csv').dropna()
    df_filled['Date'] = pd.to_datetime(df_filled['Date'])
    normalized_df = df_filled.groupby('Date').apply(scale).reset_index(drop=True).sort_values(["Symbol", "Date"]).reset_index(drop=True)
    return df_filled, normalized_df

def perform_backtest(rebalances, training_end):
    rebalances_for_backtest = rebalances.loc[rebalances.Date > training_end]
    backtest_data = rebalances_for_backtest.groupby('Date', as_index=False).apply(
        lambda x: make_backtest(x, LEVERAGE, NUMBER_OF_STOCKS, (LEVERAGE + EXPOSURE) / 2, LEVERAGE - ((LEVERAGE + EXPOSURE) / 2))
    ).reset_index(drop=True)
    backtest_data = add_benchmark(backtest_data, start_from=training_end)
    equity_curves = proces_results(backtest_data).assign(Date=lambda df: pd.to_datetime(df['Date']))
    metrics = calculate_performance_portfolio_metrics(equity_curves, calculate_risk_free_rate(FREQUENCY), FREQUENCY)
    print(metrics)
    return backtest_data

def process_test_period(model, X_test, df_filled, training_end):
    predictions = pd.DataFrame(model.predict(X_test))
    if TRAINING_MODE == "walk_forward":
        market_data = df_filled[(df_filled['Date'] >= training_end) & (df_filled['Date'] <= training_end + TESTING_WINDOW)].iloc[:, :3]
    elif TRAINING_MODE == "fixed_period":
        # Convert training_end_str to a datetime object
        training_end = datetime.strptime(training_end, '%Y-%m-%d')
        market_data = df_filled[df_filled['Date'] >= training_end].iloc[:, :3]
    predictions.reset_index(drop=True, inplace=True)
    market_data.reset_index(drop=True, inplace=True)
    rebalances = pd.concat([market_data, predictions], axis='columns')
    rebalances.columns = ['Log_Return', 'Date', 'Symbol', 'Score']

    backtest_data = perform_backtest(rebalances, training_end)
    return backtest_data

def split_data(normalized_df, current_date, training_end, testing_end):
    mask_train = (normalized_df.Date >= current_date) & (normalized_df.Date < training_end)
    mask_test = (normalized_df['Date'] >= training_end) & (normalized_df['Date'] <= testing_end)
    X_train = normalized_df[mask_train].iloc[:, 3:]
    y_train = normalized_df[mask_train].iloc[:, 0]
    X_test = normalized_df[mask_test].iloc[:, 3:]
    y_test = normalized_df[mask_test].iloc[:, 0]
    return X_train, y_train, X_test, y_test

def aggregate_results(backtest_results):
    backtest_results = pd.concat(backtest_results)
    equity_curves = proces_results(backtest_results)
    metrics = calculate_performance_portfolio_metrics(equity_curves, calculate_risk_free_rate(FREQUENCY), FREQUENCY)
    return metrics, equity_curves

def walk_forward_analysis(df_filled, normalized_df):
    backtest_results = []
    current_date = normalized_df['Date'][normalized_df.Date > START].iloc[0]
    end_date = normalized_df['Date'].max()

    while current_date + TRAINING_WINDOW < end_date:
        training_end, testing_end = current_date + TRAINING_WINDOW, current_date + TRAINING_WINDOW + TESTING_WINDOW
        print(f"Testing period from {training_end} to {testing_end}")

        # Splitting the data
        X_train, y_train, X_test, y_test = split_data(normalized_df, current_date, training_end, testing_end)
        
        # Train and evaluate the model
        algorithm = MODEL
        model = train_model(X_train, y_train, algorithm)
        mse, rmse, r2 = evaluate_model(model, X_test, y_test)

        # Additional Processing
        backtest_data = process_test_period(model, X_test, df_filled, training_end)
        backtest_results.append(backtest_data)

        # Move to the next period
        current_date += TESTING_WINDOW

    return backtest_results

def fixed_period(df_filled, normalized_df):
    # Data Splitting
    start, insample = START, TEST_START
    X, y = normalized_df.iloc[:, 3:], normalized_df.iloc[:, 0]
    X_train, y_train = X[(normalized_df.Date >= start) & (normalized_df.Date < insample)], y[(normalized_df.Date >= start) & (normalized_df.Date < insample)]
    X_test, y_test = X[normalized_df.Date >= insample], y[normalized_df.Date >= insample]

    # Model Training and Evaluation
    algorithm = MODEL  # Options: "ridge", "lasso", "elasticnet", "xgboost"
    model = train_model(X_train, y_train, algorithm)
    mse, rmse, r2 = evaluate_model(model, X_test, y_test)
    print(f"Algorithm: {algorithm}, MSE: {mse}, RMSE: {rmse}, R-squared: {r2}")

    # Additional Processing
    backtest_data = process_test_period(model, X_test, df_filled, insample)
    equity_curves = proces_results(pd.DataFrame(backtest_data))
    metrics = calculate_performance_portfolio_metrics(equity_curves, calculate_risk_free_rate(FREQUENCY), FREQUENCY)

    return metrics, equity_curves

if __name__ == "__main__":
    args = parse_arguments()
    """
    FREQUENCY = "M"
    NUMBER_OF_STOCKS = 6
    LONG_ONLY = False
    SIMUL_RANDOM_BACKTEST = False
    EXPOSURE = 0.5
    LEVERAGE = 2.5
    TRAINING_WINDOW = timedelta(days=7 * 365)  # 7 years
    TESTING_WINDOW = timedelta(days=365)       # 1 year
    """

    FREQUENCY = args.frequency
    NUMBER_OF_STOCKS = args.num_stocks
    LONG_ONLY = args.long_only
    SIMUL_RANDOM_BACKTEST = args.simul_random_backtest
    EXPOSURE = args.exposure
    LEVERAGE = args.leverage
    MODEL = args.model
    TRAINING_MODE = args.training_mode
    START = args.start
    TEST_START = args.test_start
    TRAINING_WINDOW = timedelta(days=7 * 365)  # 7 years
    TESTING_WINDOW = timedelta(days=365)       # 1 year

    df_filled, normalized_df = load_and_preprocess_data()

    if TRAINING_MODE == "fixed_period":
        metrics, equity_curves = fixed_period(df_filled, normalized_df)

    elif TRAINING_MODE == "walk_forward":
       
       backtest_results = walk_forward_analysis(df_filled, normalized_df)
    
       # Calculate metrics for all periods
       metrics, equity_curves = aggregate_results(backtest_results)

    print("*************************************************************")
    print("                     EQUITY CURVES FOR LAST 4 YEARS          ")
    print("*************************************************************")
    print(equity_curves.iloc[8:])

    print("*************************************************************")
    print("                     FINAL RESULTS                           ")
    print("*************************************************************")
    print(metrics)
