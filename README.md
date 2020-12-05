## FM2-assignment-timeseries
Some probelms in prediction. If I Use the 'Date' index, then I will face the problems in the prediction.
Another problem is whether I should change the daily price to weekly price? This may be more general. 

[Reference](http://web.vu.lt/mif/a.buteikis/wp-content/uploads/2019/02/02_StationaryTS_Python.html)
**We use ARIMA model to predict the Close Price of US T-Bond futures**
----
First we check stationary of the data, if not, then we calculate its difference and recheck it until it tends to stationary.
Notice that the P-value of ADF test indicates there is enough evidence to say it is stationary.
Use Ljung-Box test to check whether it is a White Noise Process.
Auto-select the order of ARIMA then build the model.
Then we fit the model and do the residuals analysis. From the plot we can see the model fit well.


*Prediction*
(```)def predict(fit, oridata):
    fig = plt.figure(figsize=(14, 8))
    fit.plot_predict(end=oridata.size + 50 , ax=fig.add_subplot(211))
    plt.legend().remove()
    fit.plot_predict(start=oridata.size - 100, end=oridata.size + 50, ax=fig.add_subplot(212))
    plt.tight_layout()
    plt.show()
(```)



