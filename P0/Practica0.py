from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import sys


############################3#####################
# PRACTICA 0-1

# Leer base de datos
iris = datasets.load_iris()

# Obtener caracteristicas y clases
x = np.array(iris.data)
y = np.array(iris.target)

# Asigno los colores (rojo, verde y azul)
aux = []
for i in y:
    if i == 0:
        aux.append('Red')
    elif i == 1:
        aux.append('Green')
    else:
        aux.append('Blue')

colors = np.copy(aux)

# Visualizo con un Scatter Plot las 2 últimas características
plt.scatter(x[:49,2], x[:49,3], c='Red', label=iris.target_names[0])
plt.scatter(x[50:99,2], x[50:99,3], c='Green', label=iris.target_names[1])
plt.scatter(x[100:,2], x[100:,3], c='Blue', label=iris.target_names[2])
plt.legend(loc='upper left')
plt.show()



###################################################
# Practica 0-2

# Obtengo una semilla aleatoria
ran = np.random.randint(0,sys.maxsize)

# Mezclo los arrays con la misma semilla aleatoria
np.random.seed(ran)
np.random.shuffle(x)

np.random.seed(ran)
np.random.shuffle(colors)

# Saco el 80% y 20% para el training y el test respectivamente
trainX = x[:(x.size*20)//100]
testX  = x[(x.size*20)//100:]
trainY = colors[:(colors.size*80)//100]
testY  = colors[(colors.size*80)//100:]

# Muestro los valores del training y test
print('Training X:\n {}'.format(trainX))
print('Training Y:\n {}'.format(trainY))
print('\nTest X:\n{}'.format(testX))
print('Test Y:\n{}'.format(testY))
print('\n\nTamaño del training: {}'.format(trainX.shape[0]))
print('Tamaño del test: {}'.format(testX.shape[0]))



###############################################
# Practica 0-3

# Obtengo los 100 valores equiespaciados entre 0 y 2pi
x = np.linspace(0, 2*np.pi, 100)

# Seno, coseno y seno+coseno de los valores anteriores
s = np.sin(x)
c = np.cos(x)
sc = np.sin(x) + np.cos(x)

# Visualización de las tres curvas
plt.plot(x, s, 'k--')
plt.plot(x, c, 'b--')
plt.plot(x, sc,'r--')
plt.show()