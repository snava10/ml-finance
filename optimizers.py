
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo

def f(x):
	y = (x-1.5)**2 + 0.5
	print("x = {} y={}".format(x,y))
	return y

def error_poly(coef, data):
	"""
	Computes the error between given polynomial and observed data

	Parameters
	coef: numpy.poly1D object or equivalent array representing polynomial coeficients
	data: 2D array where each row is a point (x,y)
	"""
	err = np.sum((data[:,1] - np.polyval(coef,data[:,0]))**2)
	return err

def error(line, data):
	"""Compute error between given line model and observed data
	
	Parameters
	---------
	line: tuple/list/array (C0,C1) where C0 is the slope and C1 is the Y-intercept
	data: 2D array where each row is a point (x,y)
	"""
	err = np.sum((data[:,1] - (line[0] * data[:,0] + line[1]))**2)
	return err

def fit_line_example():
	#Define the original line
	l_orig = np.float32([4,2])
	print("Original line: C0 = {}, C1 = {}".format(l_orig[0],l_orig[1]))
	xorig = np.linspace(0,10,21)
	yorig = l_orig[0] * xorig + l_orig[1]
	plt.plot(xorig,yorig,'b--',linewidth=2.0,label='Original Line')

	#Generate noisy data points
	noise_sigma = 3.0
	noise = np.random.normal(0, noise_sigma, yorig.shape)
	data = np.asarray([xorig,yorig + noise]).T
	plt.plot(data[:,0],data[:,1], 'go', label="Data points")

	l_fit = fit_line(data,error)
	print("Fitted line: C0={},C1={}".format(l_fit[0],l_fit[1]))
	plt.plot(data[:,0],l_fit[0]*data[:,0] + l_fit[1],'r--',linewidth=2.0,label="Fitted line")

	plt.show()

def fit_line(data,error):
	"""
	Parameters
	data: a 2D array where each row is a point (X0,Y)
	error: function that computes the error between a line and observed data

	Returns line that minimizes the error function
	"""

	#Generate an initial guess for the model
	l = np.float32([0, np.mean(data[:,1])]) #slope = 0, intercept = mean(y values)

	#Plot the initial guess
	x_ends = np.float32([-5,5])
	plt.plot(x_ends,l[0]*x_ends + l[1], 'm--', linewidth=2.0, label="Initial guess")
	result = spo.minimize(error,l,args=(data,),method='SLSQP',options={'disp':True})
	return result.x

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

def test_run():
	xguess = 2.0
	min_result = spo.minimize(f,xguess,method='SLSQP',options={'disp':True})
	print("minima found at")
	print("x = {} y={}".format(min_result.x,min_result.fun))

	#plot the function
	xplot = np.linspace(0.5,2.5,21)
	yplot = f(xplot)
	plt.plot(xplot,yplot)
	plt.plot(min_result.x,min_result.fun,'ro')
	plt.title("Minima of an objective function")
	plt.show()

def fit_poly_example():
	coefs = np.poly1d(np.random.randint(-10,10,size=4))
	print(coefs)
	xorig = np.linspace(0,10,21)
	yorig = np.polyval(coefs,xorig)
	print(yorig)
	plt.plot(xorig,yorig,'b--',linewidth=2.0,label='Original Line')

	#Generate noisy data points
	noise_sigma = 5.0
	noise = np.random.normal(0, noise_sigma, yorig.shape)
	data = np.asarray([xorig,yorig + noise]).T
	plt.plot(data[:,0],data[:,1],'go',label="Data points")
	c_fit = fit_poly(data,error_poly)
	print("Fitted polynomial: C={}".format(c_fit))

	plt.plot(data[:,0],np.polyval(c_fit,data[:,0]),'m--', linewidth=2.0, label="Final Guess")
	#plt.plot(data[:,0],l_fit[0]*data[:,0] + l_fit[1],'r--',linewidth=2.0,label="Fitted line")
	plt.legend(loc='upper right')
	plt.show()

if __name__=="__main__":
	#test_run()
	#fit_line_example()
	fit_poly_example()