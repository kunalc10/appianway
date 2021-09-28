import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import coint, adfuller
import datetime
from itertools import combinations
import os

def stationarity_test(X, cutoff=0.01):
    # H_0 in adfuller is unit root exists (non-stationary)
    # We must observe significant p-value to convince ourselves that the series is stationary
    pvalue = adfuller(X)[1]
    if pvalue < cutoff:
        print('p-value = ' + str(pvalue) + ' The series ' + X.name +' is likely stationary.')
    else:
        print('p-value = ' + str(pvalue) + ' The series ' + X.name +' is likely non-stationary.')


def normalize_price(X, method="std"):
    return (X - X.mean())/X.std()


def check_cointegration(X, Y):
    """ Run all possible tests of cointegration"""

    ## Ensure their size is the same
    datelist1 = set(X.index)
    datelist2 = set(Y.index)
    common_datelist = list(datelist1.intersection(datelist2))

    X = X[X.index.isin(common_datelist)].sort_index()
    Y = Y[Y.index.isin(common_datelist)].sort_index()

    ## Normalize their prices
    X = normalize_price(X)
    Y = normalize_price(Y)

    result = {}
    score, pvalue, _ = coint(X, Y, trend="c")
    print("Cointegration test base with constant trend, pvalue = ", pvalue)
    result["constant"] = pvalue

    score, pvalue, _ = coint(X, Y, trend="c")
    print("Cointegration test base with linear trend, pvalue = ", pvalue)
    result["linear"] = pvalue

    score, pvalue, _ = coint(X, Y, trend="ctt")
    print("Cointegration test base with quadratic trend, pvalue = ", pvalue)
    result["quadratic"] = pvalue

    score, pvalue, _ = coint(X, Y, trend="nc")
    print("Cointegration test base with no trend, pvalue = ", pvalue)
    result["no_constant"] = pvalue

    return result

def check_cointegration_for_tickers(stock_tryout, df, start_date=None, end_date=None):
    """ Check cointegration for a list of tickers
    and return a dictionary"""
    res = list(combinations(stock_tryout, 2))
    cointegration_info = {}
    for pair in res:
        print(pair)
        X,Y = pair[0], pair[1]
        X = extract_time_series(df,X,"5. adjusted close", start_date=start_date, end_date=end_date)
        Y = extract_time_series(df,Y,"5. adjusted close", start_date=start_date, end_date=end_date)
        cointegration_info[pair] = check_cointegration(X,Y)
    return cointegration_info

def extract_time_series(df, ticker, property, start_date=None, end_date=None):
    if not start_date:
        start_date = df["date"].min()
    if not end_date:
        end_date = df['date'].max()
    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
    df = df.sort_values("date")
    df = df.set_index("date")
    return df[df["9. Ticker Sym"]==ticker][property]

## Strength of cointegration cannot be measured. It is a Yes/No phenomenon
def filter_cointegrated_pairs(co_dict, criteria, p_value):
    """ Filter cointegrated pairs on a criteria and given p value threshold """
    pairs = []
    for key,value in co_dict.items():
        if value[criteria] < p_value:
            pairs.append(key)
    return pairs


def get_common_dates(Y, X):
    datelist1 = set(X.index)
    datelist2 = set(Y.index)
    common_datelist = list(datelist1.intersection(datelist2))
    return common_datelist


def get_difference_series(Y, X):
    datelist = get_common_dates(Y, X)
    Y = Y[Y.index.isin(datelist)]
    X = X[X.index.isin(datelist)]
    return Y - X


def plot_individual_series(Y, title=""):
    Y.plot()
    plt.title(title)
    plt.xticks(rotation="45")


def get_stationary_series(Y, X):
    """ Try both ways """
    common_datelist = get_common_dates(Y,X)
    Y = Y[Y.index.isin(common_datelist)]
    X = X[X.index.isin(common_datelist)]
    Beta_x = np.cov(Y, X)[0][1] / np.var(X)
    residual_x = Y - Beta_x * X
    # Perform ADF
    pvalue_x = adfuller(residual_x)[1]

    Beta_y = np.cov(Y, X)[0][1] / np.var(Y)
    residual_y = Y - Beta_y * Y
    # Perform ADF
    pvalue_y = adfuller(residual_y)[1]
    if pvalue_x < pvalue_y:
        print("Co-integration series = X")
        return residual_x
    else:
        print("Co-integration series = Y")
        return residual_y


def plot_common_series(Y, X, title):
    common_dates = get_common_dates(Y, X)
    Y = Y[Y.index.isin(common_dates)]
    X = X[X.index.isin(common_dates)]
    Y.plot()
    X.plot()
    plt.title(title)
    plt.xticks(rotation="45")

def generate_mad_signals(tseries, threshold = 2, window=60):
    mad = lambda x: np.abs(x - x.median()).median()
    mad_series = tseries.rolling(window).apply(mad)
    mad_series = mad_series.dropna()

    median_series = tseries.rolling(window).median()
    median_series = median_series.dropna()

    datelist = get_common_dates(tseries, mad_series)
    tseries = tseries[tseries.index.isin(datelist)]
    mad_series = mad_series[mad_series.index.isin(datelist)]
    median_series = median_series[median_series.index.isin(datelist)]

    top_strategy = []
    value = 0
    trade_enter_up = False
    trade_enter_down = False
    for i in range(tseries.shape[0]):
        if tseries.iloc[i] > threshold*mad_series.iloc[i] and trade_enter_up == False:
            value = -1
            trade_enter_up = True
        elif tseries.iloc[i] < -threshold*mad_series.iloc[i] and trade_enter_down == False:
            value = 1
            trade_enter_down = True
        elif trade_enter_down == True and tseries.iloc[i] > median_series[i]:
            value = 0
            trade_enter_down = False
        elif trade_enter_up == True and tseries.iloc[i] < median_series[i]:
            value = 0
            trade_enter_up = False

        top_strategy.append(value)
    bottom_strategy = [-1*x for x in top_strategy]

    return top_strategy, bottom_strategy, datelist

def compute_profits(Y,X, top_strategy, bottom_strategy, datelist, cointegrating_series = "X"):
    Y = Y[Y.index.isin(datelist)]
    X = X[X.index.isin(datelist)]
    top_profit = 0
    bottom_profit = 0
    top_strategy = pd.Series(top_strategy, index = datelist)
    bottom_strategy = pd.Series(bottom_strategy, index = datelist)


    switch = False
    for sp in range(1,len(Y)):
        if top_strategy[sp] != 0:
            if switch == False:
                fp = sp
                switch = True
            else:
                top_profit+=Y.iloc[sp] - Y.iloc[fp]
        else:
            switch = False

    top_profit+=Y.iloc[sp] - Y.iloc[fp]

    for sp in range(1,len(X)):
        if bottom_strategy[sp] != 0:
            if switch == False:
                fp = sp
                switch = True
            else:
                bottom_profit+=X.iloc[sp] - X.iloc[fp]
        else:
            switch = False

    bottom_profit+=X.iloc[sp] - X.iloc[fp]

    if cointegrating_series == "X":
        bottom_profit = bottom_profit*abs(np.cov(Y,X)[0][1]/np.var(X))
    else:
        top_profit = top_profit*abs(np.cov(Y,X)[0][1]/np.var(Y))


    total_profit = top_profit + bottom_profit

    return top_profit, bottom_profit

## Create rolling predictions for 2021.
def AR1_forecasting(X, size):
    train, test = X[:-size], X[-size:]
    history = [x for x in train]
    predictions = []
    length = len(train)
    for t in range(len(test)):
        model = ARIMA(history, order=(1,0,0))
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
    return predictions

def add_time(date, days=1):
    return date + datetime.timedelta(days)


def rolling_window_single_series_forecast(Series, window):
    """Currently 1 period testing"""
    total_datelist = Series.index.tolist()
    testing_datelist = total_datelist[window:]
    predictions = []
    for i in range(0, len(total_datelist) - window + 1):
        train = Series.iloc[i:window + i + 1]
        prediction = AR1_forecasting(train, size=1)
        predictions.append(prediction[0])
    return pd.Series(predictions[:-1], index=testing_datelist)

def plot_twin_axis(Y,X, y1_label, y2_label, title=""):
# create figure and axis objects with subplots()
    common_dates = get_common_dates(Y,X)
    Y = Y[Y.index.isin(common_dates)]
    X = X[X.index.isin(common_dates)]
    fig,ax = plt.subplots()
    # make a plot
    ax.plot(Y.index,Y.values, color="red", marker="o")
    # set x-axis label
    ax.set_xlabel("date",fontsize=14)
    # set y-axis label
    ax.set_ylabel(y1_label,color="red",fontsize=14)

    ax2=ax.twinx()
    # make a plot with different y-axis using second axis object
    ax2.plot(X.index, X.values,color="blue",marker="o")
    ax2.set_ylabel(y2_label,color="blue",fontsize=14)
    plt.title()
    plt.show()


def impulse_response_func(ar_coef, nlags=20, include_const=True, ax=None):
    """Calculate and plot IRF for a number of lags
       using the AR(p) model coefficients"""
    if include_const == True:
        ar_coefs = ar_coef[1:]
    else:
        ar_coefs = ar_coef
    p = np.size(ar_coefs)
    pre = np.zeros(p - 1)

    irf_result_pre = np.append(pre, np.ones(nlags + 1))
    for j in np.arange(p, nlags + p):
        irf_result_pre[j] = np.dot(ar_coefs, np.flip(irf_result_pre[j - p:j]))

    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))

    irf_result = irf_result_pre[p - 1:]
    ax.stem(np.arange(nlags + 1).flatten(), irf_result)
    ax.set_xlabel("Years", fontsize=20)
    ax.set_title('AR({}): Impulse Response Function'.format(np.size(ar_coefs)), fontsize=20)

    if ax is None:
        fig.tight_layout()


def check_cointegration_consistency(X,Y, start_date, end_date, window, criteria, threshold):
    """ Roll over a pair of series and check for consistency"""
    common_datelist = get_common_dates(Y,X)
    common_datelist = sorted(common_datelist)
    closest_start_date = find_closest_date(start_date, common_datelist)
    closest_end_date = find_closest_date(end_date, common_datelist)
    start_index = common_datelist.index(closest_start_date)
    end_index = common_datelist.index(closest_end_date) - window
    window_start_list = []
    window_end_list = []
    cointegration_p_value = []
    clearing_list = []
    print(common_datelist)
    print(closest_start_date, closest_end_date)
    print(start_index, end_index)
    for index in range(start_index, end_index):
        X_temp = X[X.index.isin(common_datelist[index:index+window])]
        Y_temp = Y[Y.index.isin(common_datelist[index:index+window])]
        cointegration_result = check_cointegration(Y_temp, X_temp)
        if cointegration_result[criteria] < threshold:
            clearing_list.append(1)
        else:
            clearing_list.append(0)
        cointegration_p_value.append(cointegration_result[criteria])
        window_start_list.append(common_datelist[index])
        window_end_list.append(common_datelist[index+window])

    df = pd.DataFrame({"start_date":window_start_list, "end_date":window_end_list,
                       "p value":cointegration_p_value, "is_coint": clearing_list})
    return df

def find_closest_date(date, datelist):
    new_datelist = datelist.copy()
    new_datelist.append(date)
    new_datelist = sorted(new_datelist)
    index = new_datelist.index(date)
    if index == 0:
        return new_datelist[1]
    elif index == len(datelist)-1:
        return new_datelist[-2]
    else:
        return new_datelist[index - 1]

def get_data(base_path, filename):
    return os.path.join(base_path, filename)


# Create figure and subplot manually
# fig = plt.figure()
# host = fig.add_subplot(111)

# More versatile wrapper
def plot_triple_curves(S1,S2, S3,xlabel, ylabel, zlabel, S1_label, S2_label, S3_label):
    fig, host = plt.subplots(figsize=(18,5)) # (width, height) in inches

    par1 = host.twinx()
    par2 = host.twinx()

    host.set_xlabel(xlabel)
    host.set_ylabel(S1_label)
    par1.set_ylabel(S2_label)
    par2.set_ylabel(S3_label)

    color1 = plt.cm.viridis(0)
    color2 = plt.cm.viridis(0.5)
    color3 = plt.cm.viridis(.9)

    p1, = host.plot(S3.index, S3.values, color=color1, label=S3_label)
    p2, = par1.plot(S1.index, S2.values, color=color2, label=S1_label)
    p3, = par2.plot(S2.index, S2.values, color=color3, label=S2_label)

    lns = [p1, p2, p3]
    host.legend(handles=lns, loc='best')


def generate_coint_performance_df(df, pairs_list, start_date, end_date, window):
    df_list = []
    rejected_pairs = []
    for pair in pairs_list:
        try:
            X1,Y1 = pair[0], pair[1]
            X = extract_time_series(df,X1,"5. adjusted close",start_date=start_date,end_date=end_date)
            Y = extract_time_series(df,Y1,"5. adjusted close", start_date=start_date,end_date=end_date)
            temp_df = check_cointegration_consistency(X,Y, start_date=start_date,end_date=end_date,
                                        window=window, criteria="constant", threshold = 0.05)
            temp_df["pair"] = X1 + "~" + Y1
            df_list.append(temp_df)
        except:
            rejected_pairs.append(pair)
    return pd.concat(df_list), rejected_pairs