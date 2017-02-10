import pandas as pd
import matplotlib.pyplot as plt
import os
from data_reader import *

def test_run():
	dates = pd.date_range('2014-01-01','2016-12-31')
	symbols = ['SPY','GOOGL','GLD']
	df = get_data(symbols, dates)
	#plot_data(df)
	#Compute global statistics
	# print('Average')
	# print(df.mean())
	# print('Median')
	# print(df.median())
	# print('Standard Deviation')
	# print(df.std())

	#rolling_mean(df)

	# df1 = df.ix[:,['SPY']]
	# rm_SPY = get_rolling_mean(df['SPY'], window=20)

	# # 2. Compute rolling standard deviation
	# rstd_SPY = get_rolling_std(df['SPY'], window=20)

	# # 3. Compute upper and lower bands
	# upper_band, lower_band = get_bollinger_bands(rm_SPY, rstd_SPY)
	
	# # Plot raw SPY values, rolling mean and Bollinger Bands
	# ax = df['SPY'].plot(title="Bollinger Bands", label='SPY')
	# rm_SPY.plot(label='Rolling mean', ax=ax)
	# upper_band.plot(label='upper band', ax=ax)
	# lower_band.plot(label='lower band', ax=ax)

	# # Add axis labels and legend
	# ax.set_xlabel("Date")
	# ax.set_ylabel("Price")
	# ax.legend(loc='upper left')
	# plt.show()
	daily_returns = compute_daily_returns(df)
	plot_data(daily_returns, title="Daily returns", ylabel="Daily returns")

def rolling_mean(df):
	df1 = df.ix[:,['SPY']]
	ax = df1['SPY'].plot(title='SPY rolling mean',label='SPY')
	rmSPY = pd.rolling_mean(df['SPY'], window=20)
	rmSPY.plot(label='Rolling mean',ax=ax)
	ax.set_xlabel("Date")
	ax.set_ylabel("Price")
	ax.legend(loc="uper_left")
	plt.show()

def get_rolling_mean(values,window):
	return pd.rolling_mean(values,window=window)

def get_rolling_std(values,window):
	return pd.rolling_std(values,window=window)

def get_bollinger_bands(rmean,rstd):
	upper_band = rmean + (2*rstd)
	lower_band = rmean - (2*rstd)
	return upper_band, lower_band

def compute_daily_returns(df):
	daily_returns = df.copy()	
	#daily_returns[1:] = (df[1:] / df[:-1].values) - 1 #using numpy
	daily_returns = (df / df.shift(1)) - 1 # pure pandas
	daily_returns.ix[0,:] = 0
	return daily_returns


if __name__=="__main__":
	test_run()