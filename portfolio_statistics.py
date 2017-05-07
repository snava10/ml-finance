
import pandas as pd
import matplotlib.pyplot as plt
import statistics as st

from data_reader import plot_data
from statistics import compute_daily_returns
import numpy as np
import math
import scipy.optimize as spo

class Portfolio:

	def __init__(self, startValue, symbols, startDate, endDate, allocation):
		self.startValue = startValue
		self.symbols = symbols
		self.startDate = startDate
		self.endDate = endDate
		self.allocation = allocation
		#self.rawdf = self._loadData(startDate, endDate)
		#self.__loadData()
	
	def plot_daily_returns(self, symbols=None):
		if not symbols:
			symbols = self.symbols
		df = Portfolio.get_data(symbols, self.startDate, self.endDate)
		daily_returns = Portfolio.compute_daily_returns(df)
		Portfolio.plot_data(daily_returns,title="Daily returns", xlabel="Dates", ylabel="Return")

	def plot_raw(self):
		df = Portfolio.get_data(self.symbols, self.startDate, self.endDate)
		Portfolio.plot_data(df)

	def plot_portfolio_raw(self, symbols=None):
		if not symbols:
			symbols = self.symbols
		df = Portfolio.get_data(symbols, self.startDate, self.endDate)
		Portfolio.plot_data(df.sum(axis=1), title="Total Portfolio Value")

	def plot_portfolio_daily_returns(self, symbols=None):
		if not symbols:
			symbols=self.symbols
		df = Portfolio.get_data(symbols, self.startDate, self.endDate)
		df = pd.DataFrame(df.sum(axis=1),columns=['Portfolio'])
		df = Portfolio.compute_daily_returns(df)
		Portfolio.plot_data(df, title="Portfolio Daily Returns", xlabel="Dates", ylabel="Return")

	def __loadData(self):
		dates = pd.date_range(self.startDate, self.endDate)
		df = get_data(self.symbols,dates)
		df = df/df.ix[0,:] #normalizing
		self.normalized_data = df
		print(self.normalized_data)
		for i in range(len(self.symbols)): #allocating
			df[self.symbols[i]] = df[self.symbols[i]]*self.allocation[i]
		
		df = df * self.startValue
		self.portVals = df.sum(axis=1)
		dr = (self.portVals / self.portVals.shift(1)) - 1
		self.dailyReturns = dr[1:]

	def get_basic_statistics(self, symbols=None):
		if not symbols:
			symbols = self.symbols

		df = Portfolio.get_data(symbols, self.startDate, self.endDate)
		df = df.sum(axis=1)
		#print(df)
		netRet = df[0]-df[-1]
		cumRet = Portfolio.get_cumulative_return(df)

		df = pd.DataFrame(df,columns=['Portfolio'])
		daily_returns = Portfolio.compute_daily_returns(df)
		avgDailyRet = Portfolio.get_average_daily_return(daily_returns)[0]

		risk = Portfolio.get_risk(daily_returns)[0]
		sr = Portfolio.get_sharp_ratio(daily_returns)[0]
		res = { 
			'Cumulative Return': cumRet,
			'Average Daily Return': avgDailyRet,
			'Risk': risk,
			'Sharp Ratio': sr,
			'Net Return' : netRet
		}
		return res

	def optimize_for_sharp_ratio(self):
		start_allocation = self.allocation
		def f(alloc,data):
			df = data.copy()
			for i in range(len(self.symbols)): #allocating
				df[self.symbols[i]] = df[self.symbols[i]]*alloc[i]
			portVals = df.sum(axis=1)
			dr = (portVals / portVals.shift(1)) - 1
			dr = dr[1:]
			return -(math.sqrt(252) * dr.mean()/dr.std())
		bounds = [(0,1) for s in self.symbols]
		constraint = {
			'type' : 'eq',
			'fun' : lambda x : sum(x) - 1
		}
		result = spo.minimize(f, start_allocation,args=(self.normalized_data,),bounds=bounds, 
			constraints=constraint,method='SLSQP',options={'disp':True})
		return result.x

	def plot_portfolio(self):
		plot_data(self.dailyReturns,title="Daily returns")

	'''
	Functions
	'''
	def compute_daily_returns(df):
		daily_returns = df.copy()	
		#daily_returns[1:] = (df[1:] / df[:-1].values) - 1 #using numpy
		daily_returns = (df / df.shift(1)) - 1 # pure pandas
		daily_returns.ix[0,:] = 0
		return daily_returns

	def get_cumulative_return(df):
		return (df[-1]/df[0])-1

	def get_average_daily_return(daily_returns):
		return daily_returns.mean()

	def get_risk(daily_returns):
		return daily_returns.std()

	def get_sharp_ratio(dailyReturns, samplesPerYear=252, riskFreeRate=0):
		k = math.sqrt(samplesPerYear)
		dr = dailyReturns.copy() - riskFreeRate
		return k * dr.mean()/dailyReturns.std()
	
	def plot_data(df, title='Stock prices',xlabel="Date",ylabel="Price"):
		ax = df.plot(title=title)
		ax.set_xlabel(xlabel)
		ax.set_ylabel(ylabel)
		plt.show()

	def get_data(symbols, startDate, endDate, include_spy=False):
		"""Read stock data (adjusted close) for given symbols from CSV files."""
		dates = pd.date_range(startDate, endDate)
		df = pd.DataFrame(index=dates)
		if include_spy and 'SPY' not in symbols:  # add SPY for reference, if absent
			symbols.insert(0, 'SPY')

		for symbol in symbols:
			symbolDf = pd.read_csv("data/{}.csv".format(symbol.lower()),
				index_col="Date",
				parse_dates=True,
				usecols=['Date','Adj Close'],
				na_values=['nan'])
			symbolDf = symbolDf.rename(columns={'Adj Close':symbol.upper()})
			df = df.join(symbolDf, how='inner')
		return df

if __name__=="__main__":
	#dates = pd.date_range('2014-01-01', '2016-12-31')
	symbols = ['SPY','GOOGL','MSFT','YHOO']
	portfolio = Portfolio(100000,symbols,'2014-01-01','2016-12-31',[0.4,0.4,0.1,0.1])
	#print(portfolio.get_basic_statistics())
	#print(portfolio.optimize_for_sharp_ratio())
	#portfolio.plot_portfolio()

	portfolio.plot_raw()
	portfolio.plot_daily_returns()
	portfolio.plot_daily_returns(symbols=['GOOGL'])
	portfolio.plot_portfolio_raw()
	portfolio.plot_portfolio_daily_returns()
	print(portfolio.get_basic_statistics())