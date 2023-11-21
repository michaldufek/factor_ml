import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV  # or LassoCV for Lasso regression
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import numpy as np

from utils import calculate_risk_free_rate, add_benchmark, proces_results, calculate_performance_portfolio_metrics, make_backtest

def simulate_synthetic_data(n_samples=1000, n_features=10, noise=0.1):
    """
    Generate synthetic data for linear regression.

    Parameters:
    - n_samples (int): Number of samples.
    - n_features (int): Number of features.
    - noise (float): The standard deviation of the Gaussian noise added to the output.

    Returns:
    - X (DataFrame): Features.
    - y (Series): Target variable.
    """
    # Random seed for reproducibility
    np.random.seed(42)

    # Generating random features
    X = np.random.randn(n_samples, n_features)

    # Generating coefficients
    coefficients = np.random.randn(n_features)

    # Generating the target variable with some noise
    y = np.dot(X, coefficients) + np.random.normal(0, noise, n_samples)

    # Converting to DataFrame and Series for compatibility with the provided code
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    y_series = pd.Series(y, name='target')

    return X_df, y_series

def scale(pdf):
    '''User-defined-function for vectorized crosssectional scaling'''
    out = pd.DataFrame()
    excluded_cols = ['Date', 'Symbol']
    scaled_cols = [ col for col in pdf.columns if col not in excluded_cols ]
    if pdf['Symbol'].nunique() > 1: # for cross-sectional scaling have to be more than 1 stocks
        scaler = StandardScaler() #PowerTransformer(method='yeo-johnson') #RobustScaler() #StandardScaler() 
        to_scaling = pdf[scaled_cols]
        scaled = pd.DataFrame(scaler.fit_transform(to_scaling), columns=scaled_cols)
        column_order = ['Log_Return', 'Date', 'Symbol'] + [col for col in pdf.columns if col not in ['Log_Return', 'Date', 'Symbol']]
        out = pd.concat([pdf[excluded_cols].reset_index(drop=True), scaled], axis='columns') # reset is necesary else it brings NaNs and NaTs to the dataframe
        out = out[column_order]
    return out

if __name__ == "__main__":
    # Final Captions with already removed correlated factors
    captions = pd.read_csv('data/feats_captions.csv').drop('Index', axis='columns').values.squeeze().tolist()
    factors = [ factor for factor in captions if factor not in ['Log_Return', 'Date', 'Symbol']]

    # Read factor data for all features
    df_filled = pd.read_csv('data/features.csv')#.drop('Index', axis='columns')
    #df_filled = df_filled[my_captions].dropna()
    df_filled = df_filled[captions].dropna()

    normalized_df = df_filled.groupby('Date').apply(scale).reset_index(drop=True)
    algorithm = "xgboost" # elasticnet, ridge, lasso

    # Prototyping
    #X, y = simulate_synthetic_data()
    X, y = normalized_df.iloc[:, 3:], normalized_df.iloc[:, 0]

    # Split the dataset into training and testing sets
    start = "2009-12-31"
    insample = "2019-12-31"
    X_train, y_train = X[(normalized_df.Date > start) & (normalized_df.Date <= insample)], y[(normalized_df.Date > start) & (normalized_df.Date <= insample)]
    X_test, y_test = X[normalized_df.Date > insample], y[normalized_df.Date > insample]

    if algorithm == "lasso":
        # Define and fit the Lasso model with cross-validation
        # Adjust the alphas array based on your regularization needs and computational resources
        model = LassoCV(alphas=np.logspace(-6, 6, 13), cv=5, random_state=42)
        model.fit(X_train, y_train)

    elif algorithm == "ridge":
        # Define the model with built-in cross-validation
        # The alphas array can be adjusted based on your regularization needs
        model = RidgeCV(alphas=[0.1, 1.0, 100.0], cv=10)  # Adjust alphas and cv as needed
        # Fit the model
        model.fit(X_train, y_train)

    elif algorithm == "elasticnet":
        # Define and fit the ElasticNet model with cross-validation
        # Adjust alphas and l1_ratio based on your needs
        model = ElasticNetCV(alphas=np.logspace(-4, 4, 10), l1_ratio=[.1, .5, .7, .9, .95, .99, 1], cv=5, random_state=42)
        model.fit(X_train, y_train)

    elif algorithm == "xgboost":
        model = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1, max_depth=3, alpha=10, n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)
    predictions = pd.DataFrame(y_pred)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f'MSE: {mse}')
    print(f'RMSE: {rmse}')
    print(f'R-squared: {r2}')

    market_data = df_filled[df_filled.Date > insample].iloc[:, :3]

    if algorithm != "xgboost":
        betas = pd.DataFrame(model.coef_, index=df_filled.columns[3:], columns=['Coeff_Value'])
        print(betas)
        betas.sort_values(by='Coeff_Value', ascending=False)

    # Backtest

    event = {
        "user-id": "9dcac14e-585c-4e72-ab8f-42177f11078f",
        "model-id": "9dcac1-sp100-linear-learner-703",
        "pool": "sp100",
        "frequency": "M",
        "backtest-period": "2020-01-01",
        "number-of-stocks": 6,
        "long-only": False,
        "simul-random-backtest": False,
        "n-simulations": 0,
        "exposure": 0.5,
        "leverage": 1.5
    }

    remove_big_techs = False

    leverage = event['leverage']
    exposure = event['exposure']
    longs_share = (leverage + exposure) / 2
    shorts_share = leverage -longs_share
    print(f'long rate: {longs_share} and short rate: {shorts_share} and leverage  {leverage}')
    frequency = event['frequency']

    risk_free_rate = calculate_risk_free_rate(frequency)

    #model_base = event['model-id']
    #backtest_id = 'Radek'
    #backtest_path = f's3://custom-metadata-models/{event["user-id"]}/models/{model_base}/backtests'
    #prediction_path = f's3://custom-metadata-models/{event["user-id"]}/models/{model_base}/backtests'
    #market_data_path = f's3://custom-metadata-models/{event["user-id"]}/models/{model_base}/backtests'
    
    #market_data = pd.read_csv("market_data.csv")
    #market_data = wr.s3.read_csv(f'{market_data_path}/market_data.csv', index_col='Index')
    #### Change ####
    #predictions = pd.read_csv('~/Downloads/training-data/predictions/data-part-0.csv', index_col=0)
    #backtest_part = len(market_data)
    #predictions = wr.s3.read_csv(f'{backtest_path}/predictions/', header=None) # replace with backtest_path
    #predictions = pd.read_csv('pred.csv', header=None)
    predictions.reset_index(drop=True,inplace=True)
    market_data.reset_index(drop=True,inplace=True)
    rebalances = pd.concat([market_data, predictions], axis='columns')
    rebalances.columns = ['Log_Return', 'Date', 'Symbol', 'Score']

    # Remove bg tech companies
    if remove_big_techs:
        big_techs = ['AAPL', 'MSFT', 'GOOG', 'GOOGL' 'META', 'AMZN']
        rebalances = rebalances[~rebalances.Symbol.isin(big_techs)]

    rebalances_for_backtest = rebalances.loc[rebalances.Date >= event['backtest-period']]

    backtest_data = rebalances_for_backtest.groupby('Date', as_index=False).apply(lambda x: make_backtest(x, leverage, event['number-of-stocks'], longs_share,shorts_share)).reset_index(drop=True)
    backtest_data = add_benchmark(backtest_data)

    pivot_df = rebalances.pivot_table(index='Date', columns='Symbol', values=['Log_Return', 'Score']).reset_index().dropna()
    pivot_df.columns = [f"{col[0]}_{col[1]}" for col in pivot_df.columns]
    pivot_df['Average_Log_Return'] = pivot_df.filter(like='Log_Return').mean(axis=1)

    symbols = rebalances['Symbol'].unique()

    equity_curves = proces_results(backtest_data).assign(Date=lambda df: pd.to_datetime(df['Date']))
    metrics = calculate_performance_portfolio_metrics(equity_curves, risk_free_rate, frequency)

    metrics