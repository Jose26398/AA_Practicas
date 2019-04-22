# -*- coding: utf-8 -*-
"""
TRABAJO 2
Nombre Estudiante: Jose Maria Sanchez Guerrero
"""

import numpy as np
import matplotlib.pyplot as plt

# Fijamos la semilla
np.random.seed(1)

print('\n\n\nEJERCICIO SOBRE LA COMPLEJIDAD DE H Y EL RUIDO')


def simula_unif(N, dim, rango):
    return np.random.uniform(rango[0], rango[1], (N, dim))


def simula_gaus(N, dim, sigma):
    media = 0
    out = np.zeros((N, dim), np.float64)
    for i in range(N):
        # Para cada columna dim se emplea un sigma determinado. Es decir, para 
        # la primera columna (eje X) se usará una N(0,sqrt(sigma[0])) 
        # y para la segunda (eje Y) N(0,sqrt(sigma[1]))
        out[i, :] = np.random.normal(loc=media, scale=np.sqrt(sigma), size=dim)

    return out


def simula_recta(intervalo):
    points = np.random.uniform(intervalo[0], intervalo[1], size=(2, 2))
    x1 = points[0, 0]
    x2 = points[1, 0]
    y1 = points[0, 1]
    y2 = points[1, 1]
    # y = a*x + b
    a = (y2 - y1) / (x2 - x1)  # Calculo de la pendiente.
    b = y1 - a * x1  # Calculo del termino independiente.

    return a, b


# EJERCICIO 1.1: Dibujar una gráfica con la nube de puntos de salida correspondiente

print('\nEjercicio 1')

# Generamos una muestra de números aleatorios de tamaño 50 en el rango (-50, 50)
x = simula_unif(50, 2, [-50, 50])
# Mostramos por pantalla
plt.scatter(x[:, 0], x[:, 1])
plt.title("Ejercicio 1.1 - Apartado a")
plt.show()

# Generamos una muestra mediante la gaussiana de tamaño 50 con un sigma de [5, 7]
x = simula_gaus(50, 2, np.array([5, 7]))
# Mostramos por pantalla
plt.scatter(x[:, 0], x[:, 1])
plt.title("Ejercicio 1.1 - Apartado b")
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")


###############################################################################
###############################################################################
###############################################################################


# EJERCICIO 1.2: Dibujar una gráfica con la nube de puntos de salida correspondiente

print('\nEjercicio 2')

# La funcion np.sign(0) da 0, lo que nos puede dar problemas
def signo(x):
    if x >= 0:
        return 1
    return -1


def f(x, y, a, b):
    return signo(y - a * x - b)


# Generamos otra muestra de puntos 2D
x = simula_unif(50, 2, [-50, 50])
y = []

# Asignamos etiquetas a las muestras generadas anteriormenteutilizando
# las funciones simula_recta(), f() y signo() proporcionadas por el profesor
a, b = simula_recta([-50, 50])
for i in range(x.shape[0]):
    y.append(f(x[i, 0], x[i, 1], a, b))
y = np.array(y)

# Ahora generamos la línea que divide los datos
lineaX = np.linspace(-50, 50, y.size)
# Usamos la funcion f(x,y)=y-ax-b, que es la distancia de cada
# punto hasta la recta
lineaY = a * lineaX + b

# Mostramos la gráfica por pantalla
plt.scatter(x[:, 0], x[:, 1], c=y)
plt.plot(lineaX, lineaY, 'r-', linewidth=2)
plt.title("Ejercicio 1.2 - Apartado a")
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")



# 1.2.b. Dibujar una gráfica donde los puntos muestren el resultado de su etiqueta, junto con la recta usada para ello
# Array con 10% de indices aleatorios para introducir ruido

# Inicializamos dos listas vacías para los indices positivos y negativos
indicesPositivos = []
indicesNegativos = []

# Le metemos los valores correspondientes del array y de etiquetas
for i in enumerate(y):
    if y[i[1]] == 1:
        indicesPositivos.append(i[0])
    else:
        indicesNegativos.append(i[0])

# Los convertimos en np.arrays
indicesPositivos = np.array(indicesPositivos)
indicesNegativos = np.array(indicesNegativos)

# Sacamos aleatoriamente un 10% de índices positivos y otro 10% de índices negativos
indexP = np.random.choice(int(indicesPositivos.size), size=int((indicesPositivos.size)/10), replace=False)
indexN = np.random.choice(int(indicesNegativos.size), size=int((indicesNegativos.size)/10), replace=False)

# Los volvemos a meter en listas
indicesPositivos = indicesPositivos.tolist()
indicesNegativos = indicesNegativos.tolist()

# Intercambiamos el 10% de índices positivos sacados anteriormente por
# el otro 10% de índices negativos de la siguiente forma
for j in range(len(indicesPositivos)):
    if np.isin(j,indexP):
        # Introducimos el positivo en la lista de negativos
        indicesNegativos.append(indicesPositivos[j])
        # Borramos el positivo cambiado
        del indicesPositivos[j]

for k in range(len(indicesNegativos)):
    if np.isin(k, indexN):
        # Introducimos el negativo en la lista de positivos
        indicesPositivos.append(indicesNegativos[k])
        # Borramos el negativo cambiado
        del indicesNegativos[k]

# Los unimos todos en un nuevo array de etiquetas llamado ruido
ruido = np.zeros(y.size, np.int64)
for m in range(len(indicesPositivos)):
    ruido[indicesPositivos[m]] = 1
for n in range(len(indicesNegativos)):
    ruido[indicesNegativos[n]] = -1

# Mostramos por pantalla las muestras con ruido
plt.scatter(x[:, 0], x[:, 1], c=ruido)
plt.plot(lineaX, lineaY, 'r-', linewidth=2)
plt.title("Ejercicio 1.2 - Apartado b")
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")


###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 1.3: Supongamos ahora que las siguientes funciones definen la frontera
# de clasificación de los puntos de la muestra en lugar de una recta

print('\nEjercicio 3')

def plot_datos_cuad(X, y, fz, title='Point cloud plot', xaxis='x axis', yaxis='y axis'):
    # Preparar datos
    min_xy = X.min(axis=0)
    max_xy = X.max(axis=0)
    border_xy = (max_xy - min_xy) * 0.01

    # Generar grid de predicciones
    xx, yy = np.mgrid[min_xy[0] - border_xy[0]:max_xy[0] + border_xy[0] + 0.001:border_xy[0],
             min_xy[1] - border_xy[1]:max_xy[1] + border_xy[1] + 0.001:border_xy[1]]
    grid = np.c_[xx.ravel(), yy.ravel(), np.ones_like(xx).ravel()]
    pred_y = fz(grid)
    # pred_y[(pred_y>-1) & (pred_y<1)]
    pred_y = np.clip(pred_y, -1, 1).reshape(xx.shape)

    # Plot
    f, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(xx, yy, pred_y, 50, cmap='RdBu', vmin=-1, vmax=1)
    ax_c = f.colorbar(contour)
    ax_c.set_label('$f(x, y)$')
    ax_c.set_ticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
    ax.scatter(X[:, 0], X[:, 1], c=y, s=50, linewidth=2,
               cmap="RdYlBu", edgecolor='white')

    XX, YY = np.meshgrid(np.linspace(round(min(min_xy)), round(max(max_xy)), X.shape[0]),
                         np.linspace(round(min(min_xy)), round(max(max_xy)), X.shape[0]))
    positions = np.vstack([XX.ravel(), YY.ravel()])
    ax.contour(XX, YY, fz(positions.T).reshape(X.shape[0], X.shape[0]), [0], colors='black')

    ax.set(
        xlim=(min_xy[0] - border_xy[0], max_xy[0] + border_xy[0]),
        ylim=(min_xy[1] - border_xy[1], max_xy[1] + border_xy[1]),
        xlabel=xaxis, ylabel=yaxis)
    plt.title(title)
    plt.show()


# Simplemente transformamos a código las funciones del enunciado
def f1(grid):
    return (grid[:,0]-10)**2+(grid[:,1]-20)**2-400

def f2(grid):
    return 0.5*(grid[:,0]+10)**2+(grid[:,1]-20)**2-400

def f3(grid):
    return 0.5*(grid[:,0]-10)**2-(grid[:,1]+20)**2-400

def f4(grid):
    return grid[:,1]-20*grid[:,0]**2-5*grid[:,0]+3

# Se las pasamos como parámetro a la funcion plot_datos_cuad() del profesor
plot_datos_cuad(x, ruido, f1)
plot_datos_cuad(x, ruido, f2)
plot_datos_cuad(x, ruido, f3)
plot_datos_cuad(x, ruido, f4)


input("\n--- Pulsar tecla para continuar al ejercicio 2 ---\n")




###############################################################################
###############################################################################
###############################################################################



print('\n\n\nMODELOS LINEALES')


# EJERCICIO 2.1: ALGORITMO PERCEPTRON

print('\nEjercicio 1')

# Implementacion de la función Perceptrón que calcula el hiperplano solución
def ajusta_PLA(datos, label, max_iter, vini):
    w = np.array(vini)  # Copiamos en w el punto inicial
    iter = 0            # Creamos una variable para las iteraciones
    converge = False    # Booleano para detener la ejecución cuando converja

    # Comienzo del bucle
    while not converge:
        converge = True # Ponemos en True
        iter += 1       # Aumentamos el número de iteraciones

        # Recorremos todas las filas de datos (todos los puntos)
        for i in range(datos.shape[0]):
            # Realizamos el producto puntual de wT*xi
            prod = np.dot(w, datos[i])

            # Comprobamos que el signo del producto sea difeerente al de la etiqueta
            if (prod >= 0 and label[i] < 0) or (prod < 0 and label[i] > 0):
                w += label[i]*datos[i]  # Si cumple la condición, actualizamos w
                converge = False        # y seleccionamos que no converge

        # Corta el bucle en caso de que llegue a las iteraciones máximas
        if iter >= max_iter:
            break

    return w, iter


# Añadimos una columna de 1 al principio
x = np.c_[np.ones((x.shape[0], 1), np.float64), x]

# Ejecutamos el algoritmo PLA con el vector de ceros
w, iter = ajusta_PLA(x, y, 1000, np.zeros(x.shape[1]))

# Imprimimos el resultado (Número de iteraciones)
print('(Array de ceros) Valor de las iteraciones necesario para converger: ', iter)

# Extra
# Simplemente muestro la gráfica con los puntos y la recta que divide los datos
# utilizando el algoritmo del Perceptron y el vector de ceros
PLA_w = np.copy(w)
PLA_x = np.linspace(-50, 50, y.size)
PLA_y = (-PLA_w[0] - PLA_w[1]*PLA_x) / PLA_w[2]

# Mostramos la gráfica por pantalla
plt.scatter(x[:,1], x[:,2], c=y)
plt.plot(PLA_x, PLA_y, 'r-', linewidth=2)
plt.title("Ejercicio 2.1 - Apartado a")
plt.show()


# Para hacerlo con aleatorios hacemos lo siguiente:
iterations = []
for i in range(0, 10):
    # Generamos un nuevo vector aleatorio para cada iteración
    w_random = np.random.uniform(low=-1, high=1, size=3)
    # Ejecutamos el algoritmo PLA
    w, iter = ajusta_PLA(x, y, 1000, w_random)
    # Lo metemos en el array
    iterations.append(iter)

# Hacemos la media y la imprimimos por pantalla
print('(Array aleatorio) Valor medio de iteraciones necesario para converger: {}'.format(np.mean(np.asarray(iterations))))


input("\n--- Pulsar tecla para continuar ---\n")



# Ahora con los datos del ejercicio 1.2.b

# Ejecutamos el algoritmo PLA con ruido y el vector de ceros
w, iter = ajusta_PLA(x, ruido, 1000, np.zeros(x.shape[1]))

# Imprimimos el resultado (Número de iteraciones)
print('(Array de ceros) Valor de las iteraciones necesario para converger: ', iter)


# Para hacerlo con aleatorios hacemos lo siguiente:
iterations = []
for i in range(0, 10):
    # Generamos un nuevo vector aleatorio para cada iteración
    w_random = np.random.uniform(low=-1, high=1, size=3)
    # Ejecutamos el algoritmo PLA con ruido
    w, iter = ajusta_PLA(x, ruido, 1000, w_random)
    # Lo metemos en el array
    iterations.append(iter)

print('(Array aleatorio) Valor medio de iteraciones necesario para converger: {}'.format(np.mean(np.asarray(iterations))))


input("\n--- Pulsar tecla para continuar ---\n")


###############################################################################
###############################################################################
###############################################################################


# EJERCICIO 2.2: REGRESIÓN LOGÍSTICA

print('\nEjercicio 2')

# Implementación del algoritmo de regresión logística utilizando el gradiente descendente estocástico
def sgdLR(x, y, initial_point, eta):
    w = np.copy(initial_point)  # Copiamos en w el punto inicial
    w = w.reshape(1,-1)         # Le asignamos una forma determinada al w

    # Calculamos el gradiente descendente
    while True:
        # Aplicamos una permutación aleatoria del tamaño de la muestra
        index = np.random.permutation(x.shape[0])
        # Asignamos los valores de los índices de 'x' e 'y' en los minibatches
        minibatch_x = x[index,:]
        minibatch_y = y[index]

        # Copiamos el w antes de modificarlo para comprobar posteriormente si parar
        w_ant = np.copy(w)

        # Calculamos el gradiente descendente
        for xn, yn in zip(minibatch_x, minibatch_y):
            gradiente = clasificadorRL(xn, yn, w)
            w -= eta * gradiente

        # Si la distancia euclídea entre el w_ant y este es menor que 0.01 paramos el bucle
        if np.linalg.norm(w_ant - w) < 0.01:
            break

    return w.reshape(-1,)


# Función para la clasificación logística
def clasificadorRL(x, y, w):
    # Numerador de la fracción
    h = y*x
    # Calculo del valor del exponente de e
    z = y*w.dot(x.reshape(-1,))
    # Devolvemos el clasificador de la regresión logística
    return - (h) / (1 + np.exp(z))


# Función para estimar el error
def estimarError(x, y, w):
    y = y.reshape(-1,1) # Le asignamos una forma determinada al y
    w = w.reshape(-1,1) # Le asignamos una forma determinada al w

    # Realizamos el producto puntual de wT·x
    z = (x.dot(w))
    # Calculamos el ERM de la regresión logística
    Eout = np.log(1 + np.exp(-(y*z)))
    # Devolvemos la media de este valor
    return np.mean(Eout, axis=0)


# Generamos parámetros a y b del cuadrado X = [0,2]x[0,2]
a, b = simula_recta([0, 2])

# Generamos muestra de entrenamiento de tamaño 100 en el cuadrado X = [0,2]x[0,2]
x_train = simula_unif(100, 2, [0, 2])
x_train = np.c_[np.ones((x_train.shape[0], 1), np.float64), x_train]

# Asignamos etiquetas a las muestras generadas anteriormenteutilizando
# las funciones simula_recta(), f() y signo() proporcionadas por el profesor
y_train = []
for i in range(x_train.shape[0]):
    y_train.append(f(x_train[i, 1], x_train[i, 2], a, b))
y_train = np.array(y_train)


# Extra
# Simplemente muestro la gráfica con los puntos y la recta que divide los datos
# utilizando el gradiente descendente estocástico con regresión logística
sgdLR_w = sgdLR(x_train, y_train, np.zeros((3,1)), 0.01)
sgdLR_x = np.linspace(0, 2, y_train.size)
sgdLR_y = (-sgdLR_w[0] - sgdLR_w[1]*sgdLR_x) / sgdLR_w[2]

# Mostramos la gráfica por pantalla
plt.scatter(x_train[:,1], x_train[:,2], c=y_train)
plt.plot(sgdLR_x, sgdLR_y, 'r-', linewidth=2)
plt.title("Ejercicio 2.2 - Apartado a")
plt.show()


input("\n--- Pulsar tecla para continuar ---\n")

# Usar la muestra de datos etiquetada para encontrar nuestra solución g y estimar Eout
# usando para ello un número suficientemente grande de nuevas muestras (>999).

# Generamos muestra de test de tamaño 1000 en el cuadrado X = [0,2]x[0,2]
x_test = simula_unif(1000, 2, [0, 2])
x_test = np.c_[np.ones((x_test.shape[0], 1), np.float64), x_test]

# Asignamos etiquetas a las muestras generadas anteriormenteutilizando
# las funciones simula_recta(), f() y signo() proporcionadas por el profesor
y_test = []
for i in range(x_test.shape[0]):
    y_test.append(f(x_test[i, 1], x_test[i, 2], a, b))
y_test = np.array(y_test)

# Imprimimos los errores Ein y Eout
print("Ein: ", estimarError(x_train, y_train, sgdLR_w))
print("Eout: ", estimarError(x_test, y_test, sgdLR_w))

# Extra
# Simplemente muestro la gráfica con los puntos y la recta que divide los datos
# utilizando el gradiente descendente estocástico con regresión logística
sgdLR_x = np.linspace(0, 2, y_test.size)
sgdLR_y = (-sgdLR_w[0] - sgdLR_w[1]*sgdLR_x) / sgdLR_w[2]

# Mostramos la gráfica por pantalla
plt.scatter(x_test[:,1], x_test[:,2], c=y_test)
plt.plot(sgdLR_x, sgdLR_y, 'r-', linewidth=2)
plt.title("Ejercicio 2.2 - Apartado b")
plt.show()

###############################################################################
###############################################################################
###############################################################################
