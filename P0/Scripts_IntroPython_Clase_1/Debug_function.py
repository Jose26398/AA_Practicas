# -*- coding: utf-8 -*-

import numpy as np

def funcion1(x):
    x.append(1)
    print(np.shape(x))
    print(len(x))
    print(type(x))
    print(dir(x))
    x = (2,3)
    print(np.shape(x))
    print(len(x))
    print(type(x))
    print(dir(x))
    
y = [0, 1, 2, 3, 4, 5]
funcion1(y)


