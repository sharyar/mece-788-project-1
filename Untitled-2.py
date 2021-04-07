#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Data set

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import matplotlib.dates as mdates
#import mplfinance as fplt

sns.set_style('dark')

STOCK_NAME = 'AADR'

df_aadr = pd.read_csv(f'/Users/nida-e-falak/Documents/MEC 788/Project/archive/Data/ETFs/{STOCK_NAME.lower()}.us.txt', index_col=0)
df_aadr.index = pd.to_datetime(df_aadr.index)

df_aadr.info()


# In[2]:


O = df_aadr['Open']
H = df_aadr['High']
L = df_aadr['Low']
C = df_aadr['Close']
V = df_aadr['Volume']
OInt = df_aadr['OpenInt']


# In[3]:


from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
sel.fit_transform([O, H, L])


# In[25]:


M = input()
m = (O+H)/2
sma0 = mean((O[:M] + H[:M])/2)
sm = [sma0]
for i in range(0,len(O)-M):
    m = sm[i] + (((O[]+H[])-())/2)
    sm.append()


# In[ ]:


all_data['SMA5_Volume'] = all_data.groupby('symbol')['Volume'].transform(lambda x: x.rolling(window = 5).mean())
all_data['SMA15_Volume'] = all_data.groupby('symbol')['Volume'].transform(lambda x: x.rolling(window = 15).mean())
all_data['SMA_Volume_Ratio'] = all_data['SMA5_Volume']/all_data['SMA15_Volume']


# In[16]:


short_rolling = df_aadr.rolling(window=10).mean()
short_rolling.head(30)
#plot(short_rolling)

long_rolling = df_aadr.rolling(window=100).mean()
long_rolling.tail()


# In[17]:


start_date = '2017-01-01'
end_date = '2017-12-31'

my_year_month_fmt = mdates.DateFormatter('%m/%y')

fig, ax = plt.subplots(figsize=(16,9))

ax.plot(df_aadr.loc[start_date:end_date, :].index, df_aadr.loc[start_date:end_date], label='Price')
ax.plot(long_rolling.loc[start_date:end_date, :].index, long_rolling.loc[start_date:end_date], label = '100-days SMA')
ax.plot(short_rolling.loc[start_date:end_date, :].index, short_rolling.loc[start_date:end_date], label = '20-days SMA')

ax.legend(loc='best')
ax.set_ylabel('Price in $')
ax.xaxis.set_major_formatter(my_year_month_fmt)


# In[18]:


ema_short = df_aadr.ewm(span=20, adjust=False).mean()

fig, ax = plt.subplots(figsize=(15,9))

ax.plot(df_aadr.loc[start_date:end_date, :].index, df_aadr.loc[start_date:end_date], label='Price')
ax.plot(ema_short.loc[start_date:end_date, :].index, ema_short.loc[start_date:end_date], label = 'Span 20-days EMA')
ax.plot(short_rolling.loc[start_date:end_date, :].index, short_rolling.loc[start_date:end_date], label = '20-days SMA')

ax.legend(loc='best')
ax.set_ylabel('Price in $')
ax.xaxis.set_major_formatter(my_year_month_fmt)


# In[ ]:


import talib
# Calculate parabolic sar
df_aadr['SAR'] = talib.SAR(data.High, data.Low, acceleration=0.02, maximum=0.2)


# In[6]:


exp1 = df_aadr.Close.ewm(span=12, adjust=False).mean()
exp2 = df_aadr.Close.ewm(span=26, adjust=False).mean()
macd = exp1-exp2
exp3 = macd.ewm(span=9, adjust=False).mean()

plt.plot(macd, label='AMD MACD', color = '#EBD2BE')
plt.plot(exp3, label='Signal Line', color='#E5A4CB')
plt.legend(loc='upper left')
plt.show()


# In[7]:


#Create the "L14" column in the DataFrame
df_aadr['L14'] = df_aadr['Low'].rolling(window=14).min()
#Create the "H14" column in the DataFrame
df_aadr['H14'] = df_aadr['High'].rolling(window=14).max()
#Create the "%K" column in the DataFrame
df_aadr['%K'] = 100*((df_aadr['Close'] - df_aadr['L14']) / (df_aadr['H14'] - df_aadr['L14']) )
#Create the "%D" column in the DataFrame
df_aadr['%D'] = df_aadr['%K'].rolling(window=3).mean()


# In[8]:


fig, axes = plt.subplots(nrows=2, ncols=1,figsize=(20,10))
df_aadr['Close'].plot(ax=axes[0]); axes[0].set_title('Close')
df_aadr[['%K','%D']].plot(ax=axes[1]); axes[1].set_title('Oscillator')


# In[9]:


df_aadr['L14'] = df_aadr['Low'].rolling(window=14).min()
df_aadr['H14'] = df_aadr['High'].rolling(window=14).max()
df_aadr['%K'] = 100*((df_aadr['Close'] - df_aadr['L14']) / (df_aadr['H14'] - df_aadr['L14']) )
df_aadr['%D'] = df_aadr['%K'].rolling(window=3).mean()
df_aadr['Sell Entry'] = ((df_aadr['%K'] < df_aadr['%D']) & (df_aadr['%K'].shift(1) > df_aadr['%D'].shift(1))) & (df_aadr['%D'] > 80)
df_aadr['Buy Entry'] = ((df_aadr['%K'] > df_aadr['%D']) & (df_aadr['%K'].shift(1) < df_aadr['%D'].shift(1))) & (df_aadr['%D'] < 20)
#Create empty "Position" column
df_aadr['Position'] = np.nan 
#Set position to -1 for sell signals
df_aadr.loc[df_aadr['Sell Entry'],'Position'] = -1 
#Set position to -1 for buy signals
df_aadr.loc[df_aadr['Buy Entry'],'Position'] = 1 
#Set starting position to flat (i.e. 0)
df_aadr['Position'].iloc[0] = 0 
#Forward fill the position column to show holding of positions through time
df_aadr['Position'] = df_aadr['Position'].fillna(method='ffill')
#Set up a column holding the daily Apple returns
df_aadr['Market Returns'] = df_aadr['Close'].pct_change()
#Create column for Strategy Returns by multiplying the daily Apple returns by the position that was held at close
#of business the previous day
df_aadr['Strategy Returns'] = df_aadr['Market Returns'] * df_aadr['Position'].shift(1)
#Finally plot the strategy returns versus Apple returns
df_aadr[['Strategy Returns','Market Returns']].cumsum().plot(figsize=(20,10))


# In[10]:


df_aadr['MA20'] = df_aadr['Close'].rolling(window=20).mean()
df_aadr['20dSTD'] = df_aadr['Close'].rolling(window=20).std() 

df_aadr['Upper'] = df_aadr['MA20'] + (df_aadr['20dSTD'] * 2)
df_aadr['Lower'] = df_aadr['MA20'] - (df_aadr['20dSTD'] * 2)

df_aadr[['Close','MA20','Upper','Lower']].plot(figsize=(10,4))
plt.grid(True)
#plt.title(stock + ' Bollinger Bands')
plt.axis('tight')
plt.ylabel('Price')


# In[31]:


df_aadr_daily_returns = df_aadr['Close'].pct_change()
df_aadr_monthly_returns = df_aadr['Close'].resample('M').ffill().pct_change()
print(df_aadr_daily_returns.head())

print(df_aadr_monthly_returns.head())

fig = plt.figure()
ax1 = fig.add_axes([0.1,0.1,0.8,0.8])
ax1.plot(df_aadr_daily_returns)
ax1.set_xlabel("Date")
ax1.set_ylabel("Percent")
ax1.set_title("df_aadr daily returns data")
plt.show()

fig = plt.figure()
ax1 = fig.add_axes([0.1,0.1,0.8,0.8])
ax1.plot(df_aadr_monthly_returns)
ax1.set_xlabel("Date")
ax1.set_ylabel("Percent")
ax1.set_title("Netflix monthly returns data")
plt.show()

fig = plt.figure()
ax1 = fig.add_axes([0.1,0.1,0.8,0.8])
df_aadr_daily_returns.plot.hist(bins = 60)
ax1.set_xlabel("Daily returns %")
ax1.set_ylabel("Percent")
ax1.set_title("AADR daily returns data")
ax1.text(-0.35,200,"Extreme Low\nreturns")
ax1.text(0.25,200,"Extreme High\nreturns")
plt.show()


# In[32]:


#df_aadr = pd.read_csv(df_aadr.Open).tail(10)  # Gets the last 10 days of trading for the current stock in iteration
pos_move = []  # List of days that the stock price increased
neg_move = []  # List of days that the stock price increased
OBV_Value = 2 # Sets the initial OBV_Value to zero
new_df_aadr = []
count = 0
while (count < 10):  # 10 because we are looking at the last 10 trading days
    if df_aadr.iloc[count,1] < df_aadr.iloc[count,4]:  # True if the stock increased in price
        pos_move.append(count)  # Add the day to the pos_move list
    elif df_aadr.iloc[count,1] > df_aadr.iloc[count,4]:  # True if the stock decreased in price
        neg_move.append(count)  # Add the day to the neg_move list
    count += 1
count2 = 0
for i in pos_move:  # Adds the volumes of positive days to OBV_Value, divide by opening price to normalize across all stocks
    OBV_Value = round(OBV_Value + (df_aadr.iloc[i,5]/df_aadr.iloc[i,1]))
for i in neg_move:  # Subtracts the volumes of negative days from OBV_Value, divide by opening price to normalize across all stocks
    OBV_Value = round(OBV_Value - (df_aadr.iloc[i,5]/df_aadr.iloc[i,1]))
#Stock_Name = ((os.path.basename(df_aadr.Open)).split(".csv")[0])  # Get the name of the current stock we are analyzing
new_df_aadr.append(['AADR', OBV_Value])  # Add the stock name and OBV value to the new_df_aadr list

print(OBV_Value)


# In[33]:


df = pd.DataFrame(new_data, columns = ['Stock', 'OBV_Value'])  # Creates a new dataframe from the new_data list
df["Stocks_Ranked"] = df["OBV_Value"].rank(ascending = False)  # Rank the stocks by their OBV_Values
df.sort_values("OBV_Value", inplace = True, ascending = False)  # Sort the ranked stocks
df.to_csv("<Your path>\\OBV_Ranked.csv", index = False)  # Save the dataframe to a csv without the index column


# In[34]:


TP = (df_aadr['High'] + df_aadr['Low'] + df_aadr['Close']) / 3 
CCI = pd.Series((TP - TP.rolling(ndays).mean()) / (0.015 * TP.rolling(ndays).std()),name = 'CCI') 
#df_aadr = df_aadr.join(CCI)


# In[38]:


from pandas_datareader import data as pdr
df_aadr_C = df_aadr

def CCI(df_aadr_C, ndays): 
    TP = (df_aadr_C['High'] + df_aadr_C['Low'] + df_aadr_C['Close']) / 3 
    CCI = pd.Series((TP - TP.rolling(ndays).mean()) / (0.015 * TP.rolling(ndays).std()),
                    name = 'CCI') 
    df_aadr_C = df_aadr_C.join(CCI) 
    return df_aadr_C

# Compute the Commodity Channel Index(CCI) for NIFTY based on the 20-day Moving average
n = 20
NIFTY_CCI = CCI(df_aadr_C, n)
CCI = NIFTY_CCI['CCI']

# Plotting the Price Series chart and the Commodity Channel index below
fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(2, 1, 1)
ax.set_xticklabels([])
plt.plot(df_aadr_C['Close'],lw=1)
plt.title('NSE Price Chart')
plt.ylabel('Close Price')
plt.grid(True)
bx = fig.add_subplot(2, 1, 2)
plt.plot(CCI,'k',lw=0.75,linestyle='-',label='CCI')
plt.legend(loc=2,prop={'size':9.5})
plt.ylabel('CCI values')
plt.grid(True)
plt.setp(plt.gca().get_xticklabels(), rotation=30)


# In[39]:


def EVM(df_aadr_C, ndays): 
    dm = ((df_aadr_C['High'] + df_aadr_C['Low'])/2) - ((df_aadr_C['High'].shift(1) + df_aadr_C['Low'].shift(1))/2)
    br = (df_aadr_C['Volume'] / 100000000) / ((df_aadr_C['High'] - df_aadr_C['Low']))
    EVM = dm / br 
    EVM_MA = pd.Series(EVM.rolling(ndays).mean(), name = 'EVM') 
    df_aadr_C = df_aadr_C.join(EVM_MA) 
    return df_aadr_C 

# Compute the 14-day Ease of Movement for AAPL
n = 14
AAPL_EVM = EVM(df_aadr_C, n)
EVM = AAPL_EVM['EVM']

# Plotting the Price Series chart and the Ease Of Movement below
fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(2, 1, 1)
ax.set_xticklabels([])
plt.plot(df_aadr_C['Close'],lw=1)
plt.title('AAPL Price Chart')
plt.ylabel('Close Price')
plt.grid(True)
bx = fig.add_subplot(2, 1, 2)
plt.plot(EVM,'k',lw=0.75,linestyle='-',label='EVM(14)')
plt.legend(loc=2,prop={'size':9})
plt.ylabel('EVM values')
plt.grid(True)
plt.setp(plt.gca().get_xticklabels(), rotation=30)


# In[ ]:




