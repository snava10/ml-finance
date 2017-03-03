import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



def get_data():
	df = pd.read_csv("data/mergeSuppliersTestReport.csv",
		parse_dates=True,
		usecols=["TimeChecksum","TimeWithoutChecksum","DimensionLines"])
	df['TimeChecksum'] = df['TimeChecksum'].apply(time_spam_to_minutes)
	df['TimeWithoutChecksum'] = df['TimeWithoutChecksum'].apply(time_spam_to_minutes)
	return df.set_index(['DimensionLines'])

def time_spam_to_minutes(ts):
	if isinstance(ts,str):
		l = list(map(int,ts.split(':')))
		return l[0]*60 + l[1] + l[2]/60.0

if __name__=="__main__":
	df = get_data()
	df['TimeWithoutChecksum'] = df['TimeWithoutChecksum'].interpolate(method='quadratic')
	print(df)
	ax = df.plot(kind='line',title="Merging Suppliers",loglog=True,style='-o')
	ax.set_xlabel("Suppliers")
	ax.set_ylabel("Time in minutes")
	lines,labels = ax.get_legend_handles_labels()
	labels[0]="Improved"
	labels[1] = "Current"
	ax.legend(lines,labels)
	plt.show()