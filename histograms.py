import pandas as pd
import matplotlib.pyplot as plt
from data_reader import get_data, plot_data
from statistics import compute_daily_returns

def test_run():
	dates = pd.date_range('2014-01-01', '2015-02-10')
	df = get_data([],dates)
	plot_data(df)

	daily_returns = compute_daily_returns(df)
	

if __name__ == "__main__":
	test_run()
