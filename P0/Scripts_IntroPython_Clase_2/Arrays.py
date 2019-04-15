# -*- coding: utf-8 -*-

#SWAP AXES
import numpy as np

x = np.array([[1,2,3,4],[5,6,7,8]])
print(x)
y = x.swapaxes(0,1)
print(y)

a = np.arange(8)
print(a)
a2 = a.reshape(4,2)
print(a2)

x = np.array([[[0,1],[2,3]],[[4,5],[6,7]]])
x
x.swapaxes(0,2)

#COPY ARRAY
x = np.array([1, 2, 3])
y = x
z = np.copy(x)
y[0] = 10
print(x, y, z)


x = np.array([1, 2, 3])
y = x
z = np.copy(x)
y = np.array([1,1])
x
y
z

# Create a simple numpy array
a=np.array([1, 2, 3, 4, 5])
# Assign 'b' to point to the same numpy array
b=a
# Test to see if b and a point to the same thing
b is a
b = a.copy()
b is a

#SORT ARRAY
import numpy as np
a=np.array([5,3,7,8,1,2,3])
a.sort()
a
a=np.array([5,3,7,8,1,2,3])
np.argsort(a)

a=np.array([5,3,7,8,1,2,3])
a[::-1].sort() 
a
a=np.array([5,3,7,8,1,2,3])
np.sort(a)[::-1] 
a



#operaciones elemento a elemento
#matrices
array2 = np.array([[0,1,0],[1,0,1]])

array1 = np.copy(array2)

array2

array1

array1.dot(array2.transpose())

np.trace(array1.dot(array2.transpose()))


#Where
a = np.arange(5,10)
np.where(a < 8)

a = np.arange(4,10).reshape(2,3)
idxs = np.where(a > 5)
idxs
result = a[idxs]
result


#Ejes
new_array = np.array([[1,2,3],[4,5,6],[7,8,9]])
new_array.shape

new_array.sum(axis=0)
new_array.sum(axis=1)

new_array = np.array([[[1,2,3],[4,5,6],[7,8,9]],[[10,10,10],[20,20,20],[30,30,30]]])
new_array.shape

new_array.sum(axis=0)
new_array.sum(axis=1)
new_array.sum(axis=2)

