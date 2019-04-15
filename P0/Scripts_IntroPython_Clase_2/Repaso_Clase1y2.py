def funcion1(arr,value=3):
	return([x*2 for x in arr if x < value//2])

funcion1([3,6,8,10,1,2,1],5)

######################################################


import numpy as np
X1 = np.zeros((10,1))
X2 = np.ones((10,1))
X3 = np.ones((10,1))*2
X4 = np.ones((10,1))*3
X = np.concatenate((X1,X2,X3,X4))
X = np.reshape(X,(10,4))
y = np.sum(X,axis=1)
clases = np.unique(y)
X_class = [X[y==c_i] for c_i in clases]

######################################################

def fun1(b):
        b.append(1)
        b = 'New Value'
        print('Dentro de fun1: ', b)

a = [0]
fun1(a)
print('Despues de fun1: ', a)

######################################################

Z = np.arange(10) 
v = np.random.uniform(0,10) 
index = (np.abs(Z-v)).argmin() 
print(Z[index])
######################################################

a = np.arange(10).reshape(2,-1)
b = np.repeat(1, 10).reshape(2,-1)
np.concatenate([a, b], axis=0)

######################################################

import numpy as np
x = np.array([3, 6, 9, 12])
x/3.0
print(x)


y = [3, 6, 9, 12]
y/3.0
print(y)


######################################################

