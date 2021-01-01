import pandas as pd
from pandas.plotting import lag_plot
import matplotlib.pyplot as plt
import seaborn as sns
import formula as mao

sp = pd.read_csv("adj USTB-10 years.csv")
sns.boxplot(x=None, y='Adj Close', data=sp, color='lightgreen')
plt.show()
print(sp['Close'].sort_values())
sp = sp.query('Close < 186.938004')
sp['Date'] = pd.to_datetime(sp['Date'], format='%Y/%m/%d')
sp.set_index('Date', inplace=True)
data = sp['Adj Close']
#Can Arima?
mao.autocorr(data)

#Stationary?
mao.look(data)
print('ADF Test \n')
mao.ADF(data)

diff = mao.diff1(data)
print('ADF Test \n')
mao.ADF(diff)
print('White Noise check \n')
mao.WN(diff)
print('look order \n')
mao.look_order(diff)
#mao.auto_order(diff,2,2)
#Choose (1,1)

modelfit = mao.arima_buildfit(data,(1,1,1))
print(modelfit.summary())
residuals = modelfit.resid

mao.tsdisplay(residuals)
mao.predict(modelfit,data)



