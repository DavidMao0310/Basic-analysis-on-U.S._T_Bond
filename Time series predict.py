import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import formula as mao

sp = pd.read_csv("adj USTB-10 years.csv")
sp['Date'] = pd.to_datetime(sp['Date'], format='%Y/%m/%d')
#sp.set_index('Date', inplace=True)
data = sp['Close']
#Stationary?
mao.look(data)
mao.ADF(data)

diff = mao.diff1(data)
mao.ADF(diff)
mao.WN(diff)

mao.look_order(diff)
#mao.auto_order(diff,2,2)
#Choose (1,1)
modelfit = mao.arima_build(data,(1,1,1))
print(modelfit.summary())
residuals = modelfit.resid
mao.tsdisplay(residuals)
mao.predict(modelfit,data)




