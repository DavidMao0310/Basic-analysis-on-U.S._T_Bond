import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.gofplots import qqplot
from statsmodels.tsa.stattools import arma_order_select_ic
from statsmodels.tsa.arima_model import ARIMA


#Stationary?
def look(x):
    plt.plot(x)
    plt.title('Close Price')
    plt.show()

def ADF(x):
    t = adfuller(x)
    output = pd.DataFrame(index=['Test Statistic Value', "p-value", "Lags Used", "Number of Observations Used", "Critical Value(1%)",
               "Critical Value(5%)", "Critical Value(10%)"], columns=['value'])
    output['value']['Test Statistic Value'] = t[0]
    output['value']['p-value'] = t[1]
    output['value']['Lags Used'] = t[2]
    output['value']['Number of Observations Used'] = t[3]
    output['value']['Critical Value(1%)'] = t[4]['1%']
    output['value']['Critical Value(5%)'] = t[4]['5%']
    output['value']['Critical Value(10%)'] = t[4]['10%']
    print(output)
#We need bulid stationary series
def diff1(x):
    t = x.diff(1).dropna()
    return t
#Since P-Value small, we have enough evidence to say stationary
#White Noise test
def WN(x):
    wn_pvalue = acorr_ljungbox(x, lags=1, return_df=True)
    print("White Noise Test", wn_pvalue)
#Since P-Value small, we have enough evidence to say non-WN
#ARMA order
def look_order(t):
    plot_acf(t, lags=20)
    plot_pacf(t, lags=20)
    plt.show()
def auto_order(x,ar,ma):
    z = arma_order_select_ic(x, max_ar=ar, max_ma=ma, ic='aic')
    print(z)
#Build Model
def arima_build(x, order):
    model = ARIMA(x, order=order)
    t = model.fit()
    return t
#Analysis residuals
def tsdisplay(y, figsize = (14, 8), lags = 20):
    tmp_data = pd.Series(y)
    fig = plt.figure(figsize = figsize)
    #Plot the time series
    tmp_data.plot(ax = fig.add_subplot(311), title = "Time Series of Residuals", legend = False)
    #Plot the ACF:
    plot_acf(tmp_data, lags = lags, zero = False, ax = fig.add_subplot(323))
    plt.xticks(np.arange(1,  lags + 1, 1.0))
    #Plot the PACF:
    plot_pacf(tmp_data, lags = 20, zero = False, ax = fig.add_subplot(324))
    plt.xticks(np.arange(1,  lags + 1, 1.0))
    #Plot the QQ plot of the data:
    qqplot(tmp_data, line='s', ax = fig.add_subplot(325))
    plt.title("QQ Plot")
    #Plot the residual histogram:
    fig.add_subplot(326).hist(tmp_data, bins = 40, density=True, range=[-5,5])
    plt.title("Histogram")
    #Fix the layout of the plots:
    plt.tight_layout()
    plt.show()

def predict(fit, oridata):
    fig = plt.figure(figsize=(14, 8))
    fit.plot_predict(end=oridata.size + 50 , ax=fig.add_subplot(211))
    plt.legend().remove()
    fit.plot_predict(start=oridata.size - 100, end=oridata.size + 50, ax=fig.add_subplot(212))
    plt.tight_layout()
    plt.show()
