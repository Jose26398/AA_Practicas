# -*- coding: utf-8 -*-

import numpy as np
x = np.zeros((1,25))
z = np.repeat([1.e-05], 25)
np.allclose(x,z,atol=1.e-05)

np.allclose(x,z,atol=1.e-06)

z = np.repeat([1.e-05, 0, 3], 10)
z

z[z<=0] #Get values

z<=0 #Get boolean values about the condition

[i for i in range(len(z)) if z[i] <= 0] #Get indexes

