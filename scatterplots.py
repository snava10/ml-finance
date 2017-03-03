
import pandas as pd
import matplotlib.pyplot as plt

from data_reader import get_data, plot_data
from statistics import compute_daily_returns
import numpy as np

def test_run():
	dates = pd.date_range('2014-01-01', '2015-02-10')
	symbols = ['SPY','googl','gld']
	df = get_data(symbols, dates)
	#plot_data(df)

	dailyReturns = compute_daily_returns(df)
	#plot_data(dailyReturns,title='daily returns',ylabel='daily returns',xlabel='date')
	
	#fig, axes = plt.subplots(nrows=2, ncols=1)
	#print(axes)
	dailyReturns.plot(kind='scatter',x='SPY',y='GOOGL')
	betaGoogle,alphaGoogle = np.polyfit(dailyReturns['SPY'],dailyReturns['GOOGL'],1)
	#using the line equation plot the scatter plot with the line. beta is the slope and alpha is the intersect with the x axis
	plt.plot(dailyReturns["SPY"], betaGoogle * dailyReturns['SPY'] + alphaGoogle, '-',color='red')
	#plt.subplot(2,1,1)
	dailyReturns.plot(kind='scatter',x='SPY',y='GLD')
	betaGold,alphaGold = np.polyfit(dailyReturns['SPY'],dailyReturns['GLD'],1)
	plt.plot(dailyReturns['SPY'],betaGold * dailyReturns['SPY'] + alphaGold, '-', color='red')
	print('Alpha Gold:',alphaGold)
	print('Beta Gold:', betaGold)
	print('Alpha Google:',alphaGoogle)
	print('Beta Google:',betaGoogle)
	plt.show()

	print(dailyReturns.corr(method='pearson'))
	

if __name__=="__main__":
	test_run()