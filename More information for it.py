import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
import pandas as pd
import talib

sp = pd.read_csv("adj USTB-10 years.csv")
sp['Date'] = pd.to_datetime(sp['Date'], format='%Y/%m/%d')
sp.set_index('Date', inplace=True)

sp['5d future close'] = sp['Adj Close'].shift(-5)
sp['5d close future pct'] = sp['5d future close'].pct_change(5)
sp['5d close pct'] = sp['Adj Close'].pct_change(5)
print(sp)
# Create moving averages and rsi for time-periods of 15, 50, 200
for n in [15, 50, 100]:
    # Create the simple moving average indicator
    sp['SMA' + str(n)] = talib.SMA(sp['Adj Close'].values, timeperiod=n)
    # Create the RSI indicator
    sp['RSI' + str(n)] = talib.RSI(sp['Adj Close'].values, timeperiod=n)


print(sp.columns.values.tolist())
print(sp)
#sp.to_csv('more information.csv')

####################################################################################
#Bollinger

sp['Std(30)'] = sp['Adj Close'].rolling(window=30).std()
sp['Upper Band'] = sp['SMA15'] + (sp['Std(30)'] * 2)
sp['Lower Band'] = sp['SMA15'] - (sp['Std(30)'] * 2)

# set style, empty figure and axes
plt.style.use('fivethirtyeight')
fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(111)

# Get index values for the X axis for DataFrame
x_axis = sp.index.get_level_values(0)

# Plot 15 days Bollinger Band
ax.fill_between(x_axis, sp['Upper Band'], sp['Lower Band'], color='silver')
# Plot Adjust Closing Price and Moving Averages
sns.scatterplot(x=x_axis, y=sp['Adj Close'],label='Adj Close', color='lightpink', size=0.0001, alpha=0.5, legend=False)
sns.lineplot(x=x_axis, y=sp['SMA15'],label='SMA(15)', color='mediumseagreen',lw=2)

# Set Title & Show the Image
ax.set_title('Bollinger Band')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
plt.show()