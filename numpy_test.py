import numpy as np
import numpy.random as rand
from time import time
import pandas as pd

def test_run():
	#print(np.array([(2,3,4),(5,6,7)]))

	#empty array
	print(np.empty(5))
	print(np.empty((5,4)))
	print(np.ones((5,4),dtype=np.int_))
	print(np.random.random((5,4))) #random from uniformal dist
	print(np.random.rand(5,4)) #random from uniformal dist
	print(np.random.normal(size=(5,4)))
	print(np.random.normal(50,10,size=(5,4))) #changing the mean and the standard deviation

	print(rand.randint(10))
	print(rand.randint(0,10))
	print(rand.randint(0,10,size=5))
	print(rand.randint(0,10,size=(2,3)))

	a = rand.randint(0,10,(5,4))
	print(a.shape)
	print(a.size)
	print(a.min(axis=0))
	print(a.max(axis=1))
	print(a.mean())

	a = np.array([9, 6, 2, 3, 12, 14, 7, 10], dtype=np.int32)  # 32-bit integer array
	print("Array:", a)
	# Find the maximum and its index in array
	print("Maximum value:", a.max())
	print("Index of max.:", get_max_index(a))

	a = rand.randint(0,10,(5,4))
	print(a)
	print(a[3,2])
	print(a[0,1:3])
	a[0,0]=1
	print(a)
	a[0,:]=2
	print(a)
	a[:,3] = [1,2,3,4,5]
	print(a)

	b = rand.rand(5)
	indices = np.array([1,1,2,3])
	print(b[indices])

	b = np.array([(20,15,12,12,23),(32,5,6,7,1)])
	mean = b.mean()
	print(b[b<mean])
	b[b<mean]=mean
	print(b)

	aritmetic_operations()

def aritmetic_operations():
	a = np.array([(12,34,5,4,6),(1,2,3,4,5)])
	b = np.array([(20,15,12,12,23),(32,5,6,7,1)])
	print(2*b)
	print(b//2)
	print(a+b)
	print(b-a)
	print(a*b) #element wise multiplication
	print(b//a) #element wise division
	#for matrix multiplication use function dot
	x = np.array([(1,2,3,4),(1,2,3,4)])
	y = np.array([(1,2,3,4)])
	print(x/y)
	df = pd.DataFrame(x)
	print(df/pd.DataFrame(y).ix[0,:])

def get_max_index(a):
	#return np.where(a==a.max())[0][0]
	return a.argmax()

if __name__ == "__main__":
	#test_run()
	aritmetic_operations()

