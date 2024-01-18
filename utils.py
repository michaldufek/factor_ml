# LAMBDA: Backtester
import json
import os
import statistics
from typing import Tuple, List
from collections import defaultdict, Counter

import boto3
#from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
import awswrangler as wr
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import confusion_matrix
import yfinance as yf

# Structute here https://lucid.app/lucidchart/14b08a83-e0f2-41cd-ba35-fdc75a5766b9/edit?viewport_loc=593%2C1059%2C2535%2C1475%2C0_0&invitationId=inv_74a325be-a0ec-4f6e-9d04-6a35815b8ab5
num_quantiles = 2
win_len = 30  # days

lambda_client = boto3.client('lambda')
#create_bucket_arn = os.environ['BACKTEST_CREATE_BUCKET_ARN']
logs = boto3.client('logs')

def calculate_risk_free_rate(frequency, annual_risk_free_rate=0.0001) -> float:
    """
    This function is used to calculate the risk-free rate of return for a given frequency.

    Args:
    frequency (str): Frequency of trading. Can be 'D' (daily), 'W' (weekly), 'M' (monthly), 'Q' (quarterly).
    annual_risk_free_rate (float, optional): The annual risk-free rate. Default is 0.02 (2%).

    Returns:
    float: The risk-free rate of return for the given frequency.
    """
    frequency_map = {'D': 252, 'W': 52, 'M': 12, 'Q': 4}
    n_of_periods = frequency_map.get(frequency, 0)

    # Convert annual risk-free rate to the rate for the given period
    risk_free_rate = annual_risk_free_rate / n_of_periods

    return risk_free_rate
def make_backtest(pdf: pd.DataFrame, leverage, number_of_stocks=20, long_share = 0.6, short_share = 0.4, random_backtest=False) -> pd.DataFrame:
    """
    This function is used for backtesting a portfolio strategy.

    It sorts the given portfolio by 'Score', then selects the top and bottom "number_of_stocks" as long and short positions.
    It then calculates the average log return for the long and short positions, and the difference between the long and short returns.

    Parameters:
    pdf (pd.DataFrame): DataFrame containing stock data. It is assumed that the DataFrame has a 'Score' column (used to rank stocks),
                        a 'Log_Return' column (used to calculate the return of each position), and a 'Date' column.

    number_of_stocks (int): Number of stocks to be selected for both long and short positions. Defaults to 20.

    random_backtest (bool): If True, the selection of stocks will be randomized instead of based on the 'Score'. Defaults to False.

    Returns:
    pd.DataFrame: A DataFrame containing the date, average return of long positions, average return of short positions, and the difference
                  between the long and short returns (long - short).
    """

    # Rebalance the portfolio: if random_backtest is False, stocks are sorted by Score. If it's True, they are shuffled
    #print(pdf.Date.unique())
    rebalance = pdf.sort_values(by='Score', ascending=False) if not random_backtest else shuffle(pdf)
    #print(rebalance)

    # Select the top number_of_stocks for long positions and the bottom number_of_stocks for short positions
    q5 = rebalance.iloc[:number_of_stocks, :] #long
    q1 = rebalance.iloc[-number_of_stocks:, :] #short
    #print('********************************************************************************')
    #print(f"Longs: {q5[['Symbol', 'Log_Return']].values.tolist()}")
    #print(f"Longs Return: {q5['Log_Return'].mean()}")
    #print(f"Shorts: {q1[['Symbol', 'Log_Return']].values.tolist()}")
    #print(f"Shorts Return: {q1['Log_Return'].mean()}")
    #print('********************************************************************************')

    # Calculate the average log return for the long and short positions
    longs_returns = q5.Log_Return.mean()
    shorts_returns = q1.Log_Return.mean()

    # Prepare the result DataFrame, with the date, longs return, shorts return, and the difference between longs and shorts
    result = pd.DataFrame({'Date': [pdf.Date.iloc[0]],
                           'Longs': [longs_returns],
                           #'Q2': [q2.Log_Return.mean()],
                            #'Q3': [q3.Log_Return.mean()],
                            #'Q4': [q4.Log_Return.mean()],
                           'Shorts': [shorts_returns],
                           'LongShort': [(longs_returns*long_share - shorts_returns*short_share)],
                           'Universe': [pdf.Log_Return.mean()]})
    
    #print('********************************************************************************')
    #print(f"Longs: {q5[['Symbol', 'Log_Return']].values.tolist()}")
    #print(f"Longs Return: {longs_returns*long_share}")
    #print(f"Shorts: {q1[['Symbol', 'Log_Return']].values.tolist()}")
    #print(f"Shorts Return: {shorts_returns*short_share}")
    #print(f"Overall  Result: {(longs_returns*long_share) - (shorts_returns*short_share)}")
    #print('********************************************************************************')

    return result

def rolling_windows_metrics(ec_without_date, ec_pct_change,risk_free_rate) -> pd.DataFrame:
    """
    This function calculates several financial metrics on a rolling window basis for the given equity curves.

    Parameters:
    ec_without_date (pd.DataFrame): DataFrame containing the equity curves (returns over time) without the dates.
    sp500_returns (pd.DataFrame): DataFrame containing the S&P 500 returns.
    ec_pct_change (pd.DataFrame): DataFrame containing the percent change in the equity curves.

    Returns:
    merged_df (pd.DataFrame): DataFrame containing calculated financial metrics.
    """
    print('Calculating performance metrics with a rolling window... ')

    # Calculate rolling window metrics and store them in a DataFrame
    metrics = pd.DataFrame({
        'Std': ec_without_date.rolling(window=win_len).std().iloc[-1],
        'Sharpe': (ec_without_date.iloc[-1, :] - 1 - risk_free_rate) /
                  ec_without_date.rolling(window=win_len).std().iloc[-1],
        'Max-Drawdown': ec_without_date.rolling(window=win_len).apply(lambda x: (x / x.cummax() - 1).min()).iloc[-1],
        'Win-Rate': (ec_pct_change > 0).rolling(window=win_len).mean().iloc[-1],
        'Skewness': ec_pct_change.rolling(window=win_len).skew().iloc[-1],
        'VaR95': ec_pct_change.rolling(window=win_len).apply(lambda x: np.quantile(x, 0.05)).iloc[-1],
        'CVaR95': ec_pct_change.rolling(window=win_len).apply(lambda x: x[x < np.quantile(x, 0.05)].mean()).iloc[-1],
        'RoMaD': ec_without_date.rolling(window=win_len).apply(lambda x: (x / x.cummax() - 1).mean()).iloc[-1],
        'Max-Drawdown-Duration': ec_without_date.rolling(window=win_len).apply(lambda x: ((x < x.cummax()) != (x < x.cummax()).shift()).cumsum().where(x < x.cummax()).value_counts().max()).iloc[-1]
    }).T.reset_index().rename(columns={'index': 'Metrics', 0: 'LongOnly', 1: 'LongTheShort', 2: 'LongShort'})

    # Drop rows with any missing values and reset index
    metrics = metrics.dropna().reset_index(drop=True)

    # Create empty dictionaries to hold alpha and beta calculations
    result_alpha, result_beta = {}, {}

    # Perform rolling window linear regression to calculate alpha and beta
    for column in ec_without_date.columns:
        x = ec_pct_change['SP500'].values
        y = ec_without_date[column].values
        result_alpha[column] = np.empty(len(x) - win_len)
        result_beta[column] = np.empty(len(x) - win_len)

        for i in range(win_len, len(x)):
            x_window = x[i - win_len:i]
            y_window = y[i - win_len:i]
            coeffs = np.polyfit(x_window, y_window, 1)
            result_alpha[column][i - win_len] = coeffs[1]  # alpha
            result_beta[column][i - win_len] = coeffs[0]  # beta

    # Create DataFrames from the calculated alphas and betas
    result_alpha_df = pd.DataFrame(pd.DataFrame(result_alpha).mean(), columns=['Alpha']).T.reset_index().rename(
        columns={'index': 'Metrics'})
    result_beta_df = pd.DataFrame(pd.DataFrame(result_beta).mean(), columns=['Beta']).T.reset_index().rename(
        columns={'index': 'Metrics'})

    # Concatenate the two dataframes vertically
    merged_df = pd.concat([metrics, result_alpha_df, result_beta_df])
    # Reset the index of the merged dataframe
    merged_df = merged_df.reset_index(drop=True)
    #print(merged_df)
    return merged_df

### Util functions ###
def calc_sharpe(simple_returns:pd.DataFrame, risk_free_rate:float, annualizator:int) -> pd.Series:
    '''
    Returns annualized sharpe ratio based adjusted excess returns.
     1. substract risk free rate from vector of returns (percentage change)
     2. calculate sharpe ratio for specific (D/W/M) timeframe
     3. annualize with * sqrt(annualization factor)
    '''
    excess_returns = simple_returns - risk_free_rate
    sharpe = ((excess_returns.mean())/ simple_returns.std()) * np.sqrt(annualizator)
    return sharpe

def calc_alpha_beta(simple_returns:pd.DataFrame, risk_free_rate:float, annualizator:int) -> pd.DataFrame:
    '''...'''
    # Create a DataFrame to hold results
    results = pd.DataFrame({
        'Metrics': ['Alpha-Universe', 'Alpha-SP500', 'Beta-Universe', 'Beta-SP500']
    })

    # Calculate benchmark returns
    universe_returns = simple_returns['Universe'].mean() * annualizator
    sp500_returns = simple_returns['SP500'].mean() * annualizator

    # loop through each appropriate column in data
    for column in simple_returns.columns:
        # Calculate strategy returns
        strategy_returns = simple_returns[column].mean() * annualizator
        
        # Calculate Beta
        (beta_universe, _) = np.polyfit(simple_returns.Universe, simple_returns[column], 1)
        (beta_sp500, _) = np.polyfit(simple_returns.SP500, simple_returns[column], 1)

        # Calculate Alpha
        alpha_universe = strategy_returns - (risk_free_rate + beta_universe * (universe_returns - risk_free_rate))
        alpha_sp500 = strategy_returns - (risk_free_rate + beta_sp500 * (sp500_returns - risk_free_rate))
        
        # Store the results
        results[column] = [alpha_universe, alpha_sp500, beta_universe, beta_sp500]
    
    results = results.set_index('Metrics')

    return results.round(8)

def calc_cagr(equities:pd.DataFrame, annualizator:int) -> pd.Series:
    '''CAGR = (End Value / Start Value)^(1/n) - 1'''
    # Calculate the number of years
    n = len(equities) / annualizator
    return (equities.iloc[-1] / 1) ** (1/n) - 1

def calc_sortino(simple_returns:pd.DataFrame, risk_free_rate, annualizator:int) -> pd.Series:
    '''Sortino Ratio = (Return of portfolio - Risk-free rate) / Downside deviation'''

    # Define downside return series with zero as the minimum
    downside_returns = simple_returns.where(simple_returns < 0, 0)

    # Calculate expected return and std deviation of downside
    expected_return = simple_returns.mean() * annualizator
    down_stdev = downside_returns.std() * np.sqrt(annualizator)

    # Calculate the sortino ratio
    sortino_ratio = (expected_return - risk_free_rate) / down_stdev
    return sortino_ratio

def calculate_performance_portfolio_metrics(equity_curves, risk_free_rate, frequency) -> pd.DataFrame:
    # Remove Date column from dataframe
    equities = equity_curves.iloc[:, 1:].fillna(method='ffill')

    ### Configuration ###
    # Convention for metrics annualization
    frequency_map = {'D': 252, 'W': 52, 'M': 12, 'Q': 4}
    annualizator = frequency_map[frequency]

    # Simple return derived from equity curves (cummulative returns)
    simple_returns = equities.pct_change().dropna()
    print('Calculating performance metrics... ')

    # Calculate all the metrics one by one
    final_equity = equities.iloc[-1]
    volatility = simple_returns.std() * np.sqrt(annualizator)
    sharpe = calc_sharpe(simple_returns, risk_free_rate, annualizator)
    alphas_betas = calc_alpha_beta(simple_returns, risk_free_rate, annualizator)
    cagrs = calc_cagr(equities, annualizator)
    maximum_drawdown = equities.apply(lambda x: (x / x.cummax() - 1).min())
    sortino_ratio = calc_sortino(simple_returns, risk_free_rate, annualizator)

    # Assign the results into one table
    metrics = pd.DataFrame({
        'Final-Equity': final_equity,
        'Volatility': volatility,
        'Sharpe': sharpe,
        'Alpha-Universe': alphas_betas.loc['Alpha-Universe'],
        'Alpha-SP500': alphas_betas.loc['Alpha-SP500'],
        'Beta-Universe': alphas_betas.loc['Beta-Universe'],
        'Beta-SP500': alphas_betas.loc['Beta-SP500'],
        'CAGR': cagrs,
        'Max-Drawdown': maximum_drawdown,
        'Average-Drawdown': equities.apply(lambda x: (x / x.cummax() - 1).mean()),
        'Win-Rate': (simple_returns > 0).mean(),
        'Skewness': simple_returns.skew(),
        'Kurtosis': simple_returns.kurtosis(),
        'VaR95': simple_returns.apply(lambda x: np.quantile(x, 0.05)),
        'CVaR95': simple_returns.apply(lambda x: x[x < np.quantile(x, 0.05)].mean()),
        'Max-Drawdown-Duration': equities.apply(lambda x: ((x < x.cummax()) != (x < x.cummax()).shift()).cumsum().where(x < x.cummax()).value_counts().max()),
        'Calmar': cagrs/abs(maximum_drawdown),
        'Sterling': equities.apply(lambda x: -((x.iloc[-1] - 1) / equities.shape[0] * annualizator) / np.mean(sorted(x / x.cummax() - 1)[:int(0.1*len(x))])), #The Sterling ratio is the ratio of the average annual rate of return over the average drawdown risk.
        'Sortino': sortino_ratio# Sortino is a variation of the Sharpe ratio that differentiates harmful volatility from total overall volatility by using the asset's standard deviation of negative portfolio returns—downside deviation—instead of the total standard deviation of portfolio returns.
    }).T

    return metrics

def proces_results(backtest_data:pd.DataFrame) -> pd.DataFrame:
    """
    This function processes the backtest results and calculates the equity curves for different strategies.

    Parameters:
    backtest_data (pd.DataFrame): DataFrame containing the backtest data, with columns for date and log returns for different strategies.

    Returns:
    equity_curves (pd.DataFrame): DataFrame containing the equity curves for different strategies over time.
    """
    # Convert log returns to simple returns and calculate the cumulative product, which gives us the equity curve
    long_only = np.cumprod(np.exp(backtest_data.Longs))
    long_the_short = np.cumprod(np.exp(backtest_data.Shorts))
    long_short = np.cumprod(np.exp(backtest_data.LongShort))
    universe = np.cumprod(np.exp(backtest_data.Universe))
    sp500 = backtest_data.SP500

    # Create a new DataFrame to hold the equity curves
    equity_curves = pd.DataFrame({
        'Date': backtest_data.Date,
        'LongOnly': long_only,
        'LongTheShort': long_the_short,
        'LongShort': long_short,
        'Universe': universe,
        'SP500': sp500
    })
    return equity_curves.fillna(method='ffill')

def universe_results(universe_data:pd.DataFrame) -> pd.DataFrame:
    universe = np.cumprod(np.exp(universe_data))
    return universe

def save_results(metrics: pd.DataFrame, backtest_path: str, equity: pd.DataFrame, metrics_win: pd.DataFrame = None,prefix: str = 'real') -> None:
    """
    This function saves the results of the backtest into csv files.

    Parameters:
    metrics (pd.DataFrame): DataFrame containing calculated performance metrics.
    backtest_path (str): String representing the path in the S3 bucket where the results will be stored.
    equity (pd.DataFrame): DataFrame containing the equity curves for different strategies.
    metrics_win (pd.DataFrame, optional): DataFrame containing rolling windows metrics. Default is None.
    prefix (str, optional): Prefix for the csv file names. Default is 'real'.

    Returns:
    None: This function does not return any value, it simply saves the data into csv files in the specified S3 bucket.
    """
    print('saving results...')
    wr.s3.to_csv(metrics, f'{backtest_path}/{prefix}_metrics.csv')
    wr.s3.to_csv(equity, f'{backtest_path}/{prefix}_equity.csv')
    if metrics_win is not None:  # For the simulation the metrics_win are not calculated for performance reasons
        wr.s3.to_csv(metrics_win, f'{backtest_path}/{prefix}_metrics_win.csv')

    #print(metrics)

def run_simulation(n_simulations, rebalances_for_backtest, number_of_stocks,ec_without_date,sp500_returns,ec_pct_change, longs_share,shorts_share, leverage):
    """
    This function runs simulations for the backtest of the investment strategy and calculates performance metrics for each run.

    Parameters:
    n_simulations (int): The number of simulations to be run.
    rebalances_for_backtest (pd.DataFrame): DataFrame containing the rebalances data to be used in the backtest.
    number_of_stocks (int): The number of stocks to be considered in the backtest.
    ec_without_date (pd.DataFrame): DataFrame with equity curves for different strategies without the date.
    sp500_returns (pd.DataFrame): DataFrame with the returns of the SP500 index.
    ec_pct_change (pd.DataFrame): DataFrame with percentage change in equity curves.

    Returns:
    Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two DataFrames. The first DataFrame contains the average performance metrics across all simulations. The second DataFrame contains the equity curves for each simulation run.
    """
    all_equities = pd.DataFrame()
    all_metrics = pd.DataFrame()

    for i in range(n_simulations):
        # Run a single simulation and calculate the performance metrics
        random_backtest_data = rebalances_for_backtest.groupby(by='Date', as_index=False).apply(make_backtest, **{
            'leverage': leverage,
            'number_of_stocks': number_of_stocks,
            'random_backtest': True,
            'long_share': longs_share,
            'short_share': shorts_share
        })
        random_backtest_data = random_backtest_data.reset_index(drop=True)
        one_run_equity = proces_results(random_backtest_data)
        one_run_metrics = pd.DataFrame(
            calculate_performance_portfolio_metrics(ec_without_date, sp500_returns, ec_pct_change))

        # Assign a simulation number to the metrics and equity curves for this run
        one_run_metrics['Simulation'] = i + 1
        one_run_equity['Simulation'] = i + 1

        # Append the metrics and equity curves from this run to the results from previous runs
        all_metrics = pd.concat([all_metrics, one_run_metrics])
        all_equities = pd.concat([all_equities, one_run_equity])

    # Calculate the average performance metrics across all simulations
    average_metrics = all_metrics.drop('Simulation', axis='columns').groupby('Metrics').mean(numeric_only=True).reset_index()
    return average_metrics, all_equities

def calculate_model_accuracy(pivot_df: pd.DataFrame, symbols):
    """
    This function calculates various accuracy metrics of the model such as Mean Squared Error (MSE), Mean Absolute Error (MAE),
    Root Mean Squared Error (RMSE), Mean Absolute Percentage Error (MAPE), and R-squared value (R2).

    Parameters:
    pivot_df (pd.DataFrame): The dataframe containing predicted scores and log return values for each stock symbol.
    rebalances (pd.DataFrame): The dataframe containing rebalance data for each stock symbol.

    Returns:
    None. The function prints out the average values of MSE, MAE, RMSE, MAPE, and R2 across all stock symbols.
    """
    print('Calculating model acccuracy metrics... ')
    mse_dict, mae_dict, rmse_dict, mape_dict, r2_dict = {}, {}, {}, {}, {}

    for symbol in symbols:
        # Extract the log returns and scores for the current symbol
        log_return_col = f'Log_Return_{symbol}'
        score_col = f'Score_{symbol}'

        # Shift the Log_Return column by the prediction horizon (e.g., 1 day)
        actual_values = pivot_df[log_return_col]
        predicted_values = pivot_df[score_col]

        # Calculate accuracy metrics
        mse = mean_squared_error(actual_values, predicted_values)
        mae = mean_absolute_error(actual_values, predicted_values)
        rmse = np.sqrt(mse)
        # To avoid dividing by zero
        actual_values = actual_values.replace(0, 1e-2)
        mape = np.mean(np.abs((actual_values - predicted_values) / actual_values)) * 100 / len(actual_values)
        r2 = 1 - (mse / np.var(actual_values))

        # Store accuracy metrics for the current symbol
        mse_dict[symbol] = mse
        mae_dict[symbol] = mae
        rmse_dict[symbol] = rmse
        mape_dict[symbol] = mape
        r2_dict[symbol] = r2

    # Print the average accuracy metrics
    print(f"Average value of MSE: {statistics.mean(mse_dict.values())}, "
          f"MAE: {statistics.mean(mae_dict.values())}, "
          f"RMSE: {statistics.mean(rmse_dict.values())}, "
          f"MAPE: {statistics.mean(mape_dict.values())}, "
          f"R2: {statistics.mean(r2_dict.values())}")
    
def get_top_quintile_stocks(row: pd.Series, cols: List[str]) -> List[str]:
    """
    This function sorts the stocks based on their values (either log return or score) in descending order and
    returns the top quintile stocks.

    Parameters:
    row (pd.Series): A row from the dataframe containing log return or score values for each stock.
    cols (List[str]): A list of column names in the row for which to determine the top quintile stocks.

    Returns:
    top_quintile_stocks (List[str]): A list of stock names in the top quintile.
    """
    # Sort the values in the row in descending order
    sorted_cols = row[cols].sort_values(ascending=False).index
    # Extract the stock names from the sorted column names
    if (len(sorted_cols[0].split('_')) > 2): #This is for the log return columns
        sorted_stocks = [col.split('_')[2] for col in sorted_cols]
    else: #this is for the score columns
        sorted_stocks = [col.split('_')[1] for col in sorted_cols]
    # Get the top quintile stocks
    top_quintile_stocks = sorted_stocks[:len(sorted_stocks) // 5]
    return top_quintile_stocks

def make_confusion_matrix(log_return_df: pd.DataFrame, score_df: pd.DataFrame,num_quantiles) -> pd.DataFrame:
    correct_predictions,incorrect_predictions = 0,0
    print('Calculating confusion matrix for each stock... ')

    """
    Creates a confusion matrix by comparing quantiles of scores and returns.

    Parameters:
    log_return_df (pd.DataFrame): DataFrame containing log return values for various tickers.
    score_df (pd.DataFrame): DataFrame containing scores for various tickers.

    Returns:
    np.array: Confusion matrix comparing quantiles of scores and returns.
    """

    # Initialize a zero matrix with dimensions corresponding to the number of quantiles
    confusion_matrix = np.zeros((num_quantiles, num_quantiles))

    # Iterate through each row of the score and log return dataframes simultaneously
    for (score_index, score_row), (return_index, return_row) in zip(score_df.iterrows(), log_return_df.iterrows()):
        # Sort score row in descending order and divide it into quantiles
        x = score_row.sort_values(ascending=False)
        #x = (x - x.mean()) / x.std()
        quintiles_score = pd.qcut(x, q=num_quantiles, labels=False)
        quintiles_score.index = quintiles_score.index.str.replace('Score_', '')

        # Solve quantiles division issue with repeated values and divide return row into quantiles
        x = return_row.sort_values(ascending=False).apply(lambda x: np.random.rand() * 0.001 if x == 0 else (1 + np.random.uniform(-0.001, 0.001) if x == 1 else x))
        #x = (x - x.mean()) / x.std()
        quintiles_return = pd.qcut(x, q=num_quantiles, labels=False)
        quintiles_return.index = quintiles_return.index.str.replace('Log_Return_', '')

        # Update confusion matrix for each ticker
        for ticker in quintiles_score.keys():
            i = quintiles_score[ticker]
            j = quintiles_return[ticker]
            confusion_matrix[i][j] += 1

    #print(confusion_matrix)
def make_confusion_matrix_quantiles(rebalances_for_backtest, number_of_stocks=20, random_backtest=False):
    """
    Calculates and returns a confusion matrix for quantiles of a dataset containing stock rebalance information.

    Parameters
    ----------
    rebalances_for_backtest : DataFrame
        A DataFrame containing rebalance information for stocks. It should contain columns 'Score', 'Log_Return', 'Symbol' and 'Date'.

    number_of_stocks : int, optional
        The number of stocks to consider for each quantile, by default 20.

    random_backtest : bool, optional
        This parameter is not used in the function. By default, it is set to False.

    Returns
    -------
    confusion_matrix : 2D numpy array
        A confusion matrix of shape (num_quantiles, num_quantiles) where num_quantiles is a global variable. The matrix cell [i][j] is incremented when the ith quantile of scores corresponds to the jth quantile of returns.

    Notes
    ----
    The function implicitly assumes the existence of a global variable 'num_quantiles' that determines the number of quantiles for categorizing the stocks.

    """
    # Nested function to calculate the mean log return for each quantile
    def quintiles(rebalance_df, n_of_stocks):
        # Initialize empty list to store dataframes of each quantile
        quintiles = []

        # Loop through each quantile
        for i in range(num_quantiles):
            # If it's the last quantile, select the last n_of_stocks rows
            if i == num_quantiles - 1:
                quint = rebalance_df.iloc[-n_of_stocks:, :].drop(['Symbol', 'Score', 'Date'], axis=1).assign(
                    Quantile=i + 1)
            # Otherwise select the rows corresponding to the current quantile
            else:
                quint = rebalance_df.iloc[i * n_of_stocks:(i + 1) * n_of_stocks, :].drop(['Symbol', 'Score', 'Date'],
                                                                                         axis=1).assign(Quantile=i + 1)

            # Append the quantile dataframe to the list
            quintiles.append(quint)

        # Stack the quantile dataframes vertically
        stacked_df = pd.concat(quintiles, axis=0)
        # Reset index to allow for later grouping
        stacked_df = stacked_df.reset_index(drop=True)
        # Group by quantile and calculate mean of Log_Return
        grouped_df = stacked_df.groupby('Quantile')['Log_Return'].mean()
        # Sort the group by Log_Return in ascending order
        grouped_df.sort_values(ascending=True, inplace=True)
        return grouped_df

    # Initialize confusion matrix as a zero matrix
    confusion_matrix = np.zeros((num_quantiles, num_quantiles))
    # Sort rebalances_for_backtest by Score in descending order
    rebalances_for_backtest_sorted_by_scores = rebalances_for_backtest.sort_values(by='Score', ascending=False)
    # Sort rebalances_for_backtest by Log_Return in descending order
    rebalances_for_backtest_sorted_by_returns = rebalances_for_backtest.sort_values(by='Log_Return', ascending=False)
    # Group rebalances_for_backtest_sorted_by_scores by quantile
    grouped_df_scores = quintiles(rebalances_for_backtest_sorted_by_scores, number_of_stocks)
    # Group rebalances_for_backtest_sorted_by_returns by quantile
    grouped_df_returns = quintiles(rebalances_for_backtest_sorted_by_returns, number_of_stocks)

    # For each pair of quantile indexes, increment the corresponding cell in the confusion matrix
    for (i, j) in zip(grouped_df_scores.index, grouped_df_returns.index):
        confusion_matrix[i - 1][j - 1] += 1

    return confusion_matrix

def sp_500_returns(df, date, frequency):
    # Convert the 'Date' column to datetime format
    if frequency == 'D':
        df['Cumulative_Return'] = (df['Return'])
        df_cumulative_returns = df.drop("Return", axis=1)

    else:
        df['Date'] = pd.to_datetime(df['Date'])

        # Set 'Date' as the index of the dataframe
        df.set_index('Date', inplace=True)

        # Convert df_dates to datetime and sort it
        df_dates = pd.to_datetime(date).sort_values()

        # Initialize an empty list to hold the cumulative returns
        cumulative_returns = []

        # Loop over the dates in df_dates, except the last one
        for i in range(len(df_dates) - 1):
            # Create a mask for the current date range
            mask = (df.index >= df_dates[i]) & (df.index < df_dates[i + 1])
            # Calculate the cumulative return for the current date range and add it to the list
            cumulative_return = (df.loc[mask, 'Return'] + 1).prod() - 1
            cumulative_returns.append(cumulative_return)

        # Create a new DataFrame with the start dates and corresponding cumulative returns
        df_cumulative_returns = pd.DataFrame({
            'Start_Date': df_dates[:-1].values,
            'Cumulative_Return': cumulative_returns
        })

    return df_cumulative_returns

def add_benchmark(backtest_data:pd.DataFrame, start_from="2020-01-01") -> pd.DataFrame:
    '''Add cumulative returns for SP500 to backtest data, default based on GSPC index'''
    #prices = wr.s3.read_csv(f's3://company-data-hub/prices/{index}.csv', index_col='Date', parse_dates=True)
    prices = yf.download("^GSPC", start=start_from)
    
    # Calculate log returns
    log_returns = np.log(prices['Adj Close'] / prices['Adj Close'].shift(1)).to_frame()
    log_returns.columns = ['SP500']

    # Set index as Date
    backtest_data.set_index('Date', inplace=True)
    backtest_data.index = pd.to_datetime(backtest_data.index)

    # Slice data for SP500 for the beginning of a backtest and fill missing values
    start_date = backtest_data.index[0]
    log_returns = log_returns.loc[start_date:].fillna(method='ffill')

    # Calculate cumulative returns as benchmarking equity curve
    cumulative_returns = np.cumprod(np.exp(log_returns))

    # Reindex SP500 data to fit backtest data format (Daily/Weekly/Monthly)
    cumulative_returns = cumulative_returns.reindex(backtest_data.index)

    # Reset indexes for both dataframes
    # Join both dataframes
    backtest_data['SP500'] = cumulative_returns 
    return backtest_data.reset_index()

def handler(event, context):
    """
    This function orchestrates the backtesting process including:
    - Reading prediction and market data
    - Evaluating the model's stock picking ability
    - Calculating performance metrics
    - Saving and visualizing the results

    Parameters:
    event (dict): A dictionary with paths and settings for backtest.
    """

    invoke_response = lambda_client.invoke(
        FunctionName=create_bucket_arn,
        InvocationType='RequestResponse',
        Payload=json.dumps(event)
    )
    backtest_id = invoke_response['Payload'].read().decode('utf-8').strip('"')
    print(f'backtest id is {backtest_id}')

    leverage = event['leverage']
    exposure = event['exposure']
    longs_share = (leverage + exposure) / 2
    shorts_share = leverage -longs_share
    print(f'long rate: {longs_share} and short rate: {shorts_share} and leverage  {leverage}')
    frequency = event['frequency']
    
    risk_free_rate = calculate_risk_free_rate(frequency)

    backtest_path = f's3://custom-metadata-models/{event["user-id"]}/models/{event["model-id"]}/backtests'
    is_simulation = event['simul-random-backtest']

    predictions = wr.s3.read_csv(f'{backtest_path}/predictions/', header=None)
    market_data = wr.s3.read_csv(f'{backtest_path}/market_data.csv', index_col='Index')

    predictions.reset_index(drop=True,inplace=True)
    market_data.reset_index(drop=True,inplace=True)
    rebalances = pd.concat([market_data, predictions], axis='columns')
    rebalances.columns = ['Log_Return', 'Date', 'Symbol', 'Score']
    rebalances_for_backtest = rebalances.loc[rebalances.Date >= event['backtest-period']]

    backtest_data = rebalances_for_backtest.groupby('Date', as_index=False).apply(lambda x: make_backtest(x, leverage, event['number-of-stocks'], longs_share,shorts_share)).reset_index(drop=True)
    backtest_data = add_benchmark(backtest_data)

    pivot_df = rebalances.pivot_table(index='Date', columns='Symbol', values=['Log_Return', 'Score']).reset_index().dropna()
    pivot_df.columns = [f"{col[0]}_{col[1]}" for col in pivot_df.columns]
    pivot_df['Average_Log_Return'] = pivot_df.filter(like='Log_Return').mean(axis=1)

    symbols = rebalances['Symbol'].unique()
    calculate_model_accuracy(pivot_df, symbols)
    log_return_df = pivot_df.filter(regex=r'^Log_Return_')
    score_df = pivot_df.filter(regex=r'^Score_')

    make_confusion_matrix(log_return_df, score_df, num_quantiles)

    equity_curves = proces_results(backtest_data).assign(Date=lambda df: pd.to_datetime(df['Date']))
    metrics = calculate_performance_portfolio_metrics(equity_curves, risk_free_rate, frequency)

    backtest_path = str(backtest_path) + '/' + str(backtest_id)
    save_results(metrics, backtest_path, equity_curves)

    '''
    if is_simulation:
        simul_metrics, simul_equity = run_simulation(
            n_simulations=event['n-simulations'],
            rebalances_for_backtest=rebalances_for_backtest,
            number_of_stocks=event['number-of-stocks'],
            ec_without_date=cumprod,
            sp500_returns=sp500,
            ec_pct_change=cumprod.pct_change().dropna(),
            longs_share= longs_share,
            shorts_share= shorts_share
        )
        save_results(simul_metrics, backtest_path, simul_equity, prefix='simul')
    '''
'''
if __name__ == '__main__':

    event = {
    "user-id": "9dcac14e-585c-4e72-ab8f-42177f11078f",
    "strategy-id": "radek-zadek",
    "model-id": "9dcac14e-585c-4e72-ab8f-42177f11078f-sp100-linear-learner-100",
    "instance-type-training": "ml.m5.4xlarge",
    "instance-count-training": 1,
    "max-runtime": 3600,
    "image": "664544806723.dkr.ecr.eu-central-1.amazonaws.com/linear-learner:latest",
    "hyperparameters": {
      "predictor_type": "regressor"
    },
    "pool": "sp100",
    "processing-image-uri": "384850799342.dkr.ecr.eu-central-1.amazonaws.com/processing-image:latest",
    "instance-type-processing": "ml.m5.4xlarge",
    "instance-count-processing": 1,
    "train-period-start": "1990-01-15",
    "train-period-end": "2000-12-30",
    "validation-period-end": "2015-12-30",
    "frequency": "M",
    "backtest-period": "2019-12-31",
    "number-of-stocks": 5,
    "long-only": False,
    "simul-random-backtest": False,
    "n-simulations": 0,
  	"exposure": 0.3,
  	"leverage": 1.5
}
    handler(event, None)'''