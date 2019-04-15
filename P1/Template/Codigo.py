# -*- coding: utf-8 -*-
"""
TRABAJO 1. 
Nombre Estudiante: Jose Maria Sanchez Guerrero
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

print('\n\n\n\nEJERCICIO SOBRE LA BUSQUEDA ITERATIVA DE OPTIMOS')

####### EJERCICIO 1 #######

# Implementación del gradiente descendente
def gradient_descent(initial_point, eta, maxIter, error2get, func):
    w = np.copy(initial_point)  # Copiamos en w el punto inicial
    iterations = 0              # Creamos una variable para las iteraciones

    # Comienzo del bucle
    while True:
        # Dependiendo del parámetro hace el gradiente de:
        #   E(u,v) -> Ejercicio 2
        #   f(x,y) -> Ejercicio 3
        if func=='E':
            gradiente = gradE(*w)
        else:
            gradiente = gradF(*w)

        w = w - eta * gradiente     # Cuerpo de la función del gradiente descendente
        iterations += 1             # Aumentamos en 1 las iteraciones

        # Dependiendo del parámetro termina el while de la siguiente forma:
        #   E -> Ejercicio 2, apartado b) -> cuando el error llegue a 10^-14
        #   f -> Ejercicio 3 -> solo paran cuando lleguen al numero de iteraciones máximo
        if func == 'E':
            if E(*w) < error2get or iterations >= maxIter:
                break
        else:
            if iterations >= maxIter:
                break
    return w, iterations


####### EJERCICIO 2 #######

### Apartado a) ###
def E(u, v):
    return ((u ** 2) * (np.exp(v)) - 2 * (v ** 2) * (np.exp(-u))) ** 2

# Derivada parcial de E con respecto a u
def dEu(u, v):
    return 4 * np.exp(-2 * u) * ((u ** 2) * (np.exp(u + v)) - 2 * (v ** 2)) * (u * np.exp(u + v) + v ** 2)

# Derivada parcial de E con respecto a v
def dEv(u, v):
    return 2 * np.exp(-2 * u) * ((u ** 2) * (np.exp(u + v)) - 4 * v) * ((u ** 2) * np.exp(u + v) - 2 * (v ** 2))

# Gradiente de E
def gradE(u, v):
    return np.array([dEu(u, v), dEv(u, v)])

# Inicializamos las variables a los valores que nos piden
eta = 0.1          # Tasa de aprendizaje
maxIter = 10000     # Número máximo de iteraciones
error2get = 1e-14   # Valor del error
initial_point = np.array([1.0, 1.0])    # Punto donde comienza la función

# Llamada a la función con los parámetros anteriores y el argumento E
w, it = gradient_descent(initial_point, eta, maxIter, error2get, 'E')

print('\nEjercicio 2')
### Apartado b) ###
print('Apartado b) Numero de iteraciones: ', it)
### Apartado c) ###
print('Apartado c) Coordenadas obtenidas: (', w[0], ', ', w[1], ')')


# Mostramos el gráfico resultante de calcular el descenso de gradiente
from mpl_toolkits.mplot3d import Axes3D

x = np.linspace(-30, 30, 50)
y = np.linspace(-30, 30, 50)
X, Y = np.meshgrid(x, y)
Z = E(X, Y)

fig = plt.figure()
ax = Axes3D(fig)

surf = ax.plot_surface(X, Y, Z, edgecolor='none', rstride=1, cstride=1, cmap='jet')

min_point = np.array([w[0], w[1]])
min_point_ = min_point[:, np.newaxis]

ax.plot(min_point_[0], min_point_[1], E(min_point_[0], min_point_[1]), 'r*', markersize=10)
ax.set(title='Ejercicio 1.2. Función sobre la que se calcula el descenso de gradiente')
ax.set_xlabel('u')
ax.set_ylabel('v')
ax.set_zlabel('E(u,v)')

plt.show()

input("\n--- Pulsar tecla para continuar ---\n")


####### EJERCICIO 3 #######
def F(x, y):
    return (x**2) + 2*(y**2) + 2*np.sin(2*np.pi*x)*np.sin(2*np.pi*y)

# Derivada parcial de F con respecto a x
def dFu(x, y):
    return 2 * (2*np.pi * np.cos(2*np.pi*x) * np.sin(2*np.pi*y) + x)

# Derivada parcial de F con respecto a y
def dFv(x, y):
    return 4 * (np.pi * np.sin(2*np.pi*x) * np.cos(2*np.pi*y) + y)

# Gradiente de F
def gradF(x, y):
    return np.array([dFu(x, y), dFv(x, y)])


### Apartado a) ###
# Asignamos los nuevos valores que nos piden
eta = 0.01          # Tasa de aprendizaje
maxIter = 50        # Número máximo de iteraciones
initial_point = np.array([0.1, 0.1])    # Punto donde comienza la función

# Llamada a la función con los parámetros anteriores y con el argumento F
w, it = gradient_descent(initial_point, eta, maxIter, 0, 'F')

print('Ejercicio 3')
print('Numero de iteraciones con eta = 0.01: ', it)
print('Coordenadas obtenidas con eta = 0.01: (', w[0], ', ', w[1], ')')

# Ahora cambiamos el eta y repetimos el experimento
eta = 0.1       # Tasa de aprendizaje
w, it = gradient_descent(initial_point, eta, maxIter, 0, 'F')
print('Numero de iteraciones con eta = 0.1: ', it)
print('Coordenadas obtenidas con eta = 0.1: (', w[0], ', ', w[1], ')')

# Generamos gráfico para ver cómo desciende el valor de la función con las iteraciones
from mpl_toolkits.mplot3d import Axes3D

x = np.linspace(-30, 30, 50)
y = np.linspace(-30, 30, 50)
X, Y = np.meshgrid(x, y)
Z = F(X, Y)

fig = plt.figure()
ax = Axes3D(fig)

surf = ax.plot_surface(X, Y, Z, edgecolor='none', rstride=1, cstride=1, cmap='jet')

min_point = np.array([w[0], w[1]])
min_point_ = min_point[:, np.newaxis]

ax.plot(min_point_[0], min_point_[1], F(min_point_[0], min_point_[1]), 'r*', markersize=10)
ax.set(title='Ejercicio 1.3. Función sobre la que se calcula el descenso de gradiente')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('F(u,v)')

plt.show()


### Apartado b) ###
# Volvemos a poner la tasa de aprendizaje en 0.01
eta = 0.01      # Tasa de aprendizaje

# Ahora cambiamos los punto iniciales y realizamos los gradientes descendentes para cada uno de ellos
initial_point = np.array([0.1, 0.1])
w, it = gradient_descent(initial_point, eta, maxIter, error2get, 'F')
print('\n\nP. inicio\t|\t\tValor de X\t\t\t|\t\tValor de Y\t\t\t|\tValor mínimo')
print('------------------------------------------------------------------------------------------')
print('(0.1,0.1)\t|\t', w[0], '\t|\t', w[1], '\t|\t', F(*w))

initial_point = np.array([1.0, 1.0])
w, it = gradient_descent(initial_point, eta, maxIter, error2get, 'F')
print('------------------------------------------------------------------------------------------')
print('(1.0,1.0)\t|\t', w[0], '\t|\t', w[1], '\t|\t', F(*w))

initial_point = np.array([-0.5, -0.5])
w, it = gradient_descent(initial_point, eta, maxIter, error2get, 'F')
print('------------------------------------------------------------------------------------------')
print('(-0.5,-0.5)\t|\t', w[0], '\t|\t', w[1], '\t|\t', F(*w))

initial_point = np.array([-1.0, -1.0])
w, it = gradient_descent(initial_point, eta, maxIter, error2get, 'F')
print('------------------------------------------------------------------------------------------')
print('(-1.0,-1.0)\t|\t', w[0], '\t|\t', w[1], '\t|\t', F(*w))

input("\n--- Pulsar tecla para continuar ---\n")





# ------------------------------------------------------------------- #
# ------------------------------------------------------------------- #
# ------------------------------------------------------------------- #
# ------------------------------------------------------------------- #

print('\n\n\nEJERCICIO SOBRE REGRESION LINEAL')

label5 = 1
label1 = -1

# Funcion para leer los datos
def readData(file_x, file_y):
    # Leemos los ficheros
    datax = np.load(file_x)
    datay = np.load(file_y)
    y = []
    x = []
    # Solo guardamos los datos cuya clase sea la 1 o la 5
    for i in range(0, datay.size):
        if datay[i] == 5 or datay[i] == 1:
            if datay[i] == 5:
                y.append(label5)
            else:
                y.append(label1)
            x.append(np.array([1, datax[i][0], datax[i][1]]))

    x = np.array(x, np.float64)
    y = np.array(y, np.float64)

    return x, y


# Funcion para calcular el error
def Err(x, y, w):
    y = y.reshape(-1,1)
    # Realizamos el producto de x*w, le restamos 'y' y lo elevamos al cuadrado
    error = (x.dot(w) - y)**2
    # Hacemos la media
    error = np.mean(error, axis=0)
    # Asignamos una forma determinada al array
    error = error.reshape(-1, 1)

    return error


# Funcion para calcular la derivada del error
def dErr(x, y, w):
    # Asignamos un shape a la 'y' para que no de error
    y = y.reshape(-1,1)
    # Realizamos el producto de x*w y le restamos 'y'
    dError = x.dot(w) - y
    # Hacemos la media
    dError = 2*np.mean(x*dError, axis=0)
    # Asignamos una forma determinada al array
    dError = dError.reshape(-1,1)

    return dError


# Gradiente Descendente Estocastico
def sgd(x, y, initial_point, eta, maxIter):
    w = np.copy(initial_point)  # Copiamos en w el punto inicial
    iterations = 1

    # Obtenemos un vector aleatorio de índices de tamaño 64
    index = np.random.choice(y.size, size=64, replace=False)
    # Asignamos los valores de los índices de 'x' e 'y' en los minibatches
    minibatch_x = x[index,:]
    minibatch_y = y[index]

    # Calculamos el gradiente descendente
    while iterations < maxIter:
        w = w - eta * dErr(minibatch_x, minibatch_y, w)
        iterations += 1

    return w


# Pseudoinversa
def pseudoinverse(x,y):
    return (np.linalg.pinv(x.T.dot(x)).dot(x.T)).dot(y)


####### EJERCICIO 1 #######

# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy')
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy')

# Cálculo de la w del modelo de regresión utilizando gradiente descendente estocástico
sgdW = sgd(x, y, np.zeros((3,1)), 0.01, 2000)
# Cálculo de la w del modelo de regresión utilizando la pseudoinversa
pinvW = pseudoinverse(x,y).transpose()
pinvW = pinvW.reshape((3,1))

# Genero puntos del tamaño de y entre 0 y 1. Luego calculo su valor en 'y' para el sgd
sgdX = np.linspace(0, 1, y.size)
sgdY = (-sgdW[0] - sgdW[1]*sgdX) / sgdW[2]

# Genero puntos del tamaño de y entre 0 y 1. Luego calculo su valor en 'y' para la pseudoinversa
pinvX = np.linspace(0, 1, y.size)
pinvY = (-pinvW[0] - pinvW[1]*pinvX) / pinvW[2]

# Mostramos la gráfica por pantalla y el error
plt.scatter(x[:,1], x[:,2], c=y)
plt.plot(sgdX, sgdY, 'r-', linewidth=2, label='SGD')
plt.plot(pinvX, pinvY, 'b-', linewidth=2, label='Pseudoinversa')

plt.title('Ejercicio 2.1. Modelo de regresión lineal con el SGD y con la pseudoinversa')
plt.xlabel('Intensidad promedio')
plt.ylabel('Simetria')
plt.legend()
plt.show()

print("\nEjercicio 1")
print('Bondad del resultado para el gradiente descendente estocastico:')
print("Ein: ", Err(x, y, sgdW))
print("Eout: ", Err(x_test, y_test, sgdW))

print('Bondad del resultado para la pseudoinversa:')
print("Ein: ", Err(x, y, pinvW))
print("Eout: ", Err(x_test, y_test, pinvW))

input("\n--- Pulsar tecla para continuar (tardará bastante) ---\n")



# Apartado a)
# Simula datos en un cuadrado [-size,size]x[-size,size]
def simula_unif(N, d, size):
    return np.random.uniform(-size, size, (N, d))

# Creamos una muestra de 1000 puntos en el cuadrado X = [-1,1] x [-1,1]
x = simula_unif(1000, 2, 1);
plt.scatter(x[:,0], x[:,1])
plt.show();


# Apartado b)
# Función para asignar una etiqueta a cada punto
def f(x1, x2, ruido):
    f = []  # Creamos un array vacío

    # Recorremos todos los valores de x1 y x2 (que tienen el mismo tamaño)
    for i in range(x1.size):
        # Si el resultado de la función es mayor que 0, le asignamos un 1
        if  ((x1[i]-0.2)**2 + x2[i]**2 - 0.6) >= 0:
            f.append(1)
        # Se es menor, un -1
        else:
            f.append(-1)

    f = np.array(f)   # Lo convertimos a un np.array

    # Si queremos añadirle ruido, dependerá del parámetro pasado como argumento
    if ruido:
        # Sacamos indices aleatorios con el 10% del tamaño
        index = np.random.choice(int(f.size), size=int((f.size)/10), replace=False)
        # Cambiamos los 1 por -1 y viceversa
        for i in range(f.size):
            if np.isin(i,index):
                f[i] = -f[i]

    return f


#Calculamos un nuevo 'y' con ruido y lo mostramos por pantalla
y = f(x[:,0],x[:,1],True)
plt.scatter(x[:,0], x[:,1], c=y)
plt.show()


# Apartado c)
# Añadimos una columna de 1 al principio de la matriz
x = np.c_[np.ones((1000, 1), np.float64), x]
# Calculamos el SGD y éste los valores para 'x' e 'y' para la línea de regresión
w = sgd(x, y, np.zeros((3,1)), 0.01, 200 )
lineX = np.linspace(-1, 1, y.size)
lineY = (-w[0] - w[1]*lineX) / w[2]

# Mostramos el gráfico por pantalla
plt.scatter(x[:,1], x[:,2], c=y)
plt.plot(lineX, lineY, 'r-', linewidth=2)

plt.title('Ejercicio 2.2. Modelo de regresión lineal con ruido')
plt.ylim(-1.0,1.0)
plt.show()

print('\nEjercicio 2')
print("Ein: ", Err(x, y, w))


# Apartado d)
M = 1000    # Número de experimentos
N = 1000    # Número de muestras

# Inicializamos los siguientes valores a 0
iterations = 0
EinM = 0    # Media del Ein
EoutM = 0   # Media del Eout

while iterations < M:
    # Generamos una muestra aleatoria como hemos hecho anteriormente sin ruido
    x = simula_unif(N, 2, 1)
    y = f(x[:,0], x[:,1], False)
    x = np.c_[np.ones((N, 1), np.float64), x]

    # Generamos una muestra aleatoria para el test, esta vez con ruido
    x_test = simula_unif(N, 2, 1)
    y_test = f(x_test[:, 0], x_test[:, 1], True)
    x_test = np.c_[np.ones((N, 1), np.float64), x_test]

    # Obtenemos el SGD para calcular los errores
    w = sgd(x, y, np.zeros((3,1)), 0.01, 200)
    EinM = EinM + Err(x, y, w)                  # Vamos acumulando los valores
    EoutM = EoutM + Err(x_test, y_test, w)      # en estas variables
    iterations += 1

# Una vez terminado, dividimos por el M para obtener la media
EinM = EinM/M
EoutM = EoutM/M

# Imprimimos los errores
print('Valores medios de los errores:')
print("Ein: ", EinM)
print("Eout: ", EoutM)