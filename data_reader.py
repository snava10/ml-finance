import pandas as pd
import matplotlib.pyplot as plt
import os

def test_run():
	df = pd.read_csv("data/googl.csv")
	print("something")
	print(df.head())

def get_max_close(symbol):
	df = pd.read_csv("data/{}.csv".format(symbol))
	return df['Volume'].mean()

def get_mean_volume(symbol):
	df = pd.read_csv("data/{}.csv".format(symbol))
	return df['Volume'].mean()

def plot_column(symbol,column):
	df = pd.read_csv("data/{}.csv".format(symbol))
	print(df[column])
	df[column].plot()
	plt.show()

def plot_columns(symbol,columns):
	df = pd.read_csv("data/{}.csv".format(symbol))
	df[columns].plot()
	plt.show()

def aggregate(symbols):
	startDate = '2014-01-01'
	endDate = '2014-12-31'
	dates = pd.date_range(startDate,endDate)
	df1 = pd.DataFrame(index=dates)

	for symbol in symbols:
		symbolDf = pd.read_csv("data/{}.csv".format(symbol.lower()),
			index_col="Date",
			parse_dates=True,
			usecols=['Date','Adj Close'],
			na_values=['nan'])
		symbolDf = symbolDf.rename(columns={'Adj Close':symbol.upper()})
		#print(googlDf)
		df1 = df1.join(symbolDf,how='inner')
	print(df1)

def symbol_to_path(symbol):
	return os.path.join(base_dir,"{}.csv".format(str(symbol).lower()))

def get_data(symbols, dates):
	"""Read stock data (adjusted close) for given symbols from CSV files."""
	df = pd.DataFrame(index=dates)
	if 'SPY' not in symbols:  # add SPY for reference, if absent
		symbols.insert(0, 'SPY')

	for symbol in symbols:
		symbolDf = pd.read_csv("data/{}.csv".format(symbol.lower()),
			index_col="Date",
			parse_dates=True,
			usecols=['Date','Adj Close'],
			na_values=['nan'])
		symbolDf = symbolDf.rename(columns={'Adj Close':symbol.upper()})
		df = df.join(symbolDf,how='inner')
	return df

def plot_data(dataFrame, title='Stock prices',xlabel="Date",ylabel="Price"):
	ax = dataFrame.plot(title=title)
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	plt.show()

def plot_selected(df, columns, start_index, end_index):
	#print(df.ix[start_index:end_index,['SPY', 'YHOO']])
	plot_data(df.ix[start_index:end_index,['SPY', 'YHOO']],title="Selected data")

def normalize_data(df):
	return df/df.ix[0,:]

if __name__ == "__main__":
	# for symbol in ['MSFT', 'GOOGL']:
	# 	print("Mean Volume")
	# 	print(symbol, get_mean_volume(symbol.lower()))

	#plot_columns('googl',['Adj Close'])
	#plot_columns('googl',['Adj Close','Close'])
	#aggregate(['googl','msft','yhoo'])
	#dates = pd.date_range('2014-01-22', '2016-01-26')
	#df = get_data(['googl','msft','yhoo'],dates)
	#print(df)

	#plot_data(df)

	dates = pd.date_range('2014-01-01', '2015-02-10')
	symbols = ['googl', 'yhoo', 'gld']  # SPY will be added in get_data()
	df = get_data(symbols, dates)
	#print(df/df.ix[0,:])
	#dates = pd.date_range('2014-01-01', '2014-01-31')
	df = normalize_data(df)
	print(df)
	plot_selected(df, ['SPY', 'YHOO'], '2014-03-01', '2014-01-31')