import pandas as pd
import matplotlib.pyplot as plt
from data_reader import get_data, plot_data
from statistics import compute_daily_returns

dates = pd.date_range('2014-01-01', '2015-02-10')

def test_run():
	df = get_data([],dates)
	#plot_data(df)

	daily_returns = compute_daily_returns(df)
	#plot_data(daily_returns, title='Daily Returns', ylabel='Daily returns')

	#plot 
	daily_returns.hist(bins=20)
	#plt.show()

	mean = daily_returns['SPY'].mean()
	print("mean=",mean)
	std = daily_returns['SPY'].std()
	print("std=",std)

	plt.axvline(mean,color='w',linestyle='dashed',linewidth=2)
	plt.axvline(std,color='r',linestyle='dashed',linewidth=2)
	plt.axvline(-std,color='r',linestyle='dashed',linewidth=2)
	plt.show()

	print(daily_returns.kurtosis())

def plot_2_histograms():
	df = get_data(['SPY','GOOGL'],dates)
	#plot_data(df)

	daily_returns = compute_daily_returns(df)
	#plot_data(daily_returns, title="Daily returs")

	#daily_returns.hist(bins=20)
	daily_returns['SPY'].hist(bins=20,label='SPY')
	daily_returns['GOOGL'].hist(bins=20,label='GOOGL')
	plt.legend(loc='upper right')
	plt.show()

if __name__ == "__main__":
	#test_run()
	plot_2_histograms()
