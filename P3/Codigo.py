"""
TRABAJO 3
Nombre Estudiante: Jose Maria Sanchez Guerrero
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing

# Fijamos la semilla
np.random.seed(1)


print('\n\n\nVISUALIZAR LOS DATOS')


# Optical Recognition of Handwritten Digits Data Set
df = pd.read_csv('datos/optdigits.tra')
trainX_optdigits = np.array(df)[:,:-1]
trainY_optdigits = np.array(df)[:,-1:]

df = pd.read_csv('datos/optdigits.tes')
testX_optdigits = np.array(df)[:,:-1]
testY_optdigits = np.array(df)[:,-1:]


# Airfoil Self-Noise Data Set
data = np.loadtxt('datos/airfoil_self_noise.dat')
trainX_airfoil = preprocessing.normalize(data[:,:-1])
# trainX_airfoil = data[:,:-1]
trainY_airfoil = data[:,-1:]