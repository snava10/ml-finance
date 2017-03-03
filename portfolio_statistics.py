
import pandas as pd
import matplotlib.pyplot as plt
import statistics as st

from data_reader import get_data, plot_data
from statistics import compute_daily_returns
import numpy as np
import math
import scipy.optimize as spo

class Portfolio:
	def __init__(self,startValue,symbols,startDate,endDate,allocation):
		self.startValue = startValue
		self.symbols = symbols
		self.startDate = startDate
		self.endDate = endDate
		self.allocation = allocation
		self.__loadData()
	
	def __loadData(self):
		dates = pd.date_range(self.startDate, self.endDate)
		df = get_data(self.symbols,dates)
		df = df/df.ix[0,:] #normalizing
		self.normalized_data = df
		for i in range(len(self.symbols)): #allocating
			df[self.symbols[i]] = df[self.symbols[i]]*self.allocation[i]
		
		df = df * self.startValue
		self.portVals = df.sum(axis=1)
		dr = (self.portVals / self.portVals.shift(1)) - 1
		self.dailyReturns = dr[1:]

	def get_cumulative_return(self):
		return (self.portVals[-1]/self.portVals[0])-1

	def get_average_daily_return(self):
		return self.dailyReturns.mean()

	def get_risk(self):
		return self.dailyReturns.std()

	def get_sharp_ratio(self, samplesPerYear=252, riskFreeRate=0):
		k = math.sqrt(samplesPerYear)
		dr = self.dailyReturns.copy() - riskFreeRate
		return k * dr.mean()/self.dailyReturns.std()

	def get_basic_statistics(self):
		cumRet = self.get_cumulative_return()
		avgDailyRet = self.get_average_daily_return()
		risk = self.get_risk()
		sr = self.get_sharp_ratio()
		res = { 
			'Cumulative Return': cumRet,
			'Average Daily Return': avgDailyRet,
			'Risk': risk,
			'Sharp Ratio': sr
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

if __name__=="__main__":
	dates = pd.date_range('2014-01-01', '2016-12-31')
	symbols = ['SPY','GOOGL','MSFT','YHOO']
	portfolio = Portfolio(100000,symbols,'2014-01-01','2016-12-31',[0.4,0.4,0.1,0.1])
	print(portfolio.get_basic_statistics())
	print(portfolio.optimize_for_sharp_ratio())



