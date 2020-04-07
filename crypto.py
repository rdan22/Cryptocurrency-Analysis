#Description: This is a python program for cryptocurrency analysis
#Data can be found at: https://www.coindesk.com/price/litecoin

#Import libraries
import numpy as np 
import pandas as pd 

#Load the data
df_btc = pd.read_csv('filepath')
df_eth = pd.read_csv('filepath')
df_ltc = pd.read_csv('filepath')

print(df_btc)
print(df_eth)
print(df_ltc)

#print the data 
print(df_btc.head())
print(df_eth.head())
print(df_ltc.head())

#Create a new dataframe that holds the closing price of all 3 cryptocurrencies
df = pd.DataFrame({'BTC': df_btc['Closing Price (USD)'], 
				   'ETH': df_eth['Closing Price (USD)'],
				   'LTC': df_ltc['Closing Price (USD)']
})

#Show the new dataframe
print(df)

#Get statistics in the data
print(df.describe())

#Visualize the cryptocurrency closing prices
import matplotlib.pyplot as plt 
plt.style.use('fivethirtyeight')

my_crypto = df
plt.figure(figsize = (12.2, 4.5))
for c in my_crypto.columns.values:
	plt.plot(my_crypto[c], label = c)

plt.title('Cryptocurrency Graph')
plt.xlabel('Days')
plt.ylabel('Crypto Price ($)')
plt.legend(my_crypto.columns.values, loc= 'upper left')
plt.show()

#Scale the data 
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler(feature_range = (0, 100))
scaled = min_max_scaler.fit_transform(df)
print(scaled)

#Convert the scaled data into a dataframe
df_scale = pd.DataFrame(scaled, columns = df.columns)

#Visualize the scaled data
my_crypto = df_scale 
plt.figure(figsize = (12.4, 4.5))
for c in my_crypto.columns.values:
	plt.plot(my_crypto[c], label = c)

plt.title('Cryptocurrency Scaled Graph')
plt.xlabel('Days')
plt.ylabel('Crypto Scaled Price ($)')
plt.legend(my_crypto.columns.values, loc= 'upper left')
plt.show()

#Get the daily simple return
DSR = df.pct_change(1)
print(DSR)

#Visualize the DSR
plt.figure(figsize=(12, 4.5))
for c in DSR.columns.values:
	plt.plot(DSR.index, DSR[c], label = c, lw = 2, alpha = 0.7)

plt.title('Daily Simple Returns')
plt.xlabel('Days')
plt.ylabel('Percentage (in decimal form)')
plt.legend(DSR.columns.values, loc='upper right')
plt.show()

#Let's look at the volatility of all three cryptocurrencies
print('The cryptocurrency volatility:')
print(DSR.std())

#What's the average DSR 
print(DSR.mean())

#Get the correlation matrix
#Essentially if one changes how do the others change
print(DSR.corr())

#Visalize the correlation
import seaborn as sns

plt.subplots(figsize = (11,11))
print(sns.heatmap(DSR.corr(), annot = True, fmt = '.2%'))

#Get the daily cumulative simple returns 
DCSR = (DSR + 1).cumprod()

#Show
print(DCSR)

#Visualize DCSR
plt.figure(figsize= (12.2, 4.5))
for c in DCSR.columns.values:
	plt.plot(DCSR.index, DCSR[c], label = c, lw = 2)

plt.title('Daily Cumulative Simple Returns')
plt.xlabel('Days')
plt.ylabel('Growth of $1 investment')
plt.legend(DCSR.columns.values, loc = 'upper left', fontsize = 10)
plt.show()

#What would have happened if we'd invested $1 a year ago in all three of these assets?
'''
On the 100th day, for Bitcoin, the price would have been $3. Ether would've been about $2.25
and Litecoin would've been about $2. It looks like on the current day, Ether and Litecoin would've
lost me money. Bitcoin would have some profit. Looks like all data points to investing in Bitcoin 
(not financial advice though). 
'''




