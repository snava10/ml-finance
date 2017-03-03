import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#import scipy.optimize as spo


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

def error_poly(coef, data):
	"""
	Computes the error between given polynomial and observed data

	Parameters
	coef: numpy.poly1D object or equivalent array representing polynomial coeficients
	data: 2D array where each row is a point (x,y)
	"""
	err = np.sum((data[:,1] - np.polyval(coef,data[:,0]))**2)
	return err

def fit_poly(data, error_func, degree=3):
	"""
	Fit a polynomial to given data, using supplied error function
	"""
	Cguess = np.poly1d(np.ones(degree + 1, dtype=np.float32))

	#Plot initial guess
	x = np.linspace(-5,5,21)
	plt.plot(x,np.polyval(Cguess,x),'r--', linewidth=2.0, label="Initial Guess")

	#Call the optimizer
	result = spo.minimize(error_func, Cguess, args=(data,), method='SLSQP',options={'disp':True})
	return np.poly1d(result.x)

if __name__=="__main__":
	df = get_data()
	poly1 = fit_poly(df.ix[:,['TimeChecksum']],error_poly)
	print(df.ix[:,['TimeChecksum']])
	#print(df)
	#ax = df.plot(kind='line',title="Merging Suppliers",loglog=True,style='-o')
	#ax.set_xlabel("Suppliers")
	#ax.set_ylabel("Time in minutes")
	#lines,labels = ax.get_legend_handles_labels()
	#labels[0]="Improved"
	#labels[1] = "Current"
	#ax.legend(lines,labels)
	#plt.show()