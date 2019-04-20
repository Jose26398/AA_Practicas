# -*- coding: utf-8 -*-
"""
TRABAJO 2
Nombre Estudiante: Jose Maria Sanchez Guerrero
"""

import numpy as np
import matplotlib.pyplot as plt

# Fijamos la semilla
np.random.seed(1)


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

x = simula_unif(50, 2, [-50, 50])
plt.scatter(x[:, 0], x[:, 1])
plt.show()

x = simula_gaus(50, 2, np.array([5, 7]))
plt.scatter(x[:, 0], x[:, 1])
plt.show()

#input("\n--- Pulsar tecla para continuar ---\n")


###############################################################################
###############################################################################
###############################################################################


# EJERCICIO 1.2: Dibujar una gráfica con la nube de puntos de salida correspondiente

# La funcion np.sign(0) da 0, lo que nos puede dar problemas
def signo(x):
    if x >= 0:
        return 1
    return -1


def f(x, y, a, b):
    return signo(y - a * x - b)


x = simula_unif(50, 2, [-50, 50])
y = []

a, b = simula_recta([-50, 50])
for i in range(x.shape[0]):
    y.append(f(x[i, 0], x[i, 1], a, b))
y = np.array(y)


lineaX = np.linspace(-50, 50, y.size)
lineaY = a * lineaX + b

# Mostramos la gráfica por pantalla y el error
plt.scatter(x[:, 0], x[:, 1], c=y)
plt.plot(lineaX, lineaY, 'r-', linewidth=2)
plt.show()

#input("\n--- Pulsar tecla para continuar ---\n")

# 1.2.b. Dibujar una gráfica donde los puntos muestren el resultado de su etiqueta, junto con la recta usada para ello
# Array con 10% de indices aleatorios para introducir ruido

indicesPositivos = []
indicesNegativos = []

for i in enumerate(y):
    if y[i[1]] == 1:
        indicesPositivos.append(i[0])
    else:
        indicesNegativos.append(i[0])

indicesPositivos = np.array(indicesPositivos)
indicesNegativos = np.array(indicesNegativos)

indexP = np.random.choice(int(indicesPositivos.size), size=int((indicesPositivos.size)/10), replace=False)
indexN = np.random.choice(int(indicesNegativos.size), size=int((indicesNegativos.size)/10), replace=False)

indicesPositivos = indicesPositivos.tolist()
indicesNegativos = indicesNegativos.tolist()

# Cambiamos los 1 por -1 y viceversa
for j in range(len(indicesPositivos)):
    if np.isin(j,indexP):
        indicesNegativos.append(indicesPositivos[j])
        del indicesPositivos[j]
for k in range(len(indicesNegativos)):
    if np.isin(k, indexN):
        indicesPositivos.append(indicesNegativos[k])
        del indicesNegativos[k]


ruido = np.zeros(y.size, np.int64)
for m in range(len(indicesPositivos)):
    ruido[indicesPositivos[m]] = 1
for n in range(len(indicesNegativos)):
    ruido[indicesNegativos[n]] = -1

plt.scatter(x[:, 0], x[:, 1], c=ruido)
plt.plot(lineaX, lineaY, 'r-', linewidth=2)
plt.show()

##input("\n--- Pulsar tecla para continuar ---\n")


###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 1.3: Supongamos ahora que las siguientes funciones definen la frontera
# de clasificación de los puntos de la muestra en lugar de una recta

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


def f1(grid):
    return (grid[:,0]-10)**2+(grid[:,1]-20)**2-400

def f2(grid):
    return 0.5*(grid[:,0]+10)**2+(grid[:,1]-20)**2-400

def f3(grid):
    return 0.5*(grid[:,0]-10)**2-(grid[:,1]+20)**2-400

def f4(grid):
    return grid[:,1]-20*grid[:,0]**2-5*grid[:,0]+3

plot_datos_cuad(x, ruido, f1)
plot_datos_cuad(x, ruido, f2)
plot_datos_cuad(x, ruido, f3)
plot_datos_cuad(x, ruido, f4)


input("\n--- Pulsar tecla para continuar al ejercicio 2 ---\n")


###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 2.1: ALGORITMO PERCEPTRON

def ajusta_PLA(datos, label, max_iter, vini):
    w = np.array(vini)
    iter = 0
    converge = False
    while not converge:
        converge = True
        iter += 1
        for i in range(datos.shape[0]):
            prod = np.dot(w, datos[i])

            if (prod >= 0 and label[i] < 0) or (prod < 0 and label[i] > 0):
                w += label[i]*datos[i]
                converge = False

        if iter >= max_iter:
            break

    return w, iter


x = np.c_[np.ones((x.shape[0], 1), np.float64), x]
w_0 = np.zeros(x.shape[1])

# Random initializations
iterations = []
for i in range(0, 10):
    w, iter = ajusta_PLA(x, y, 1000, w_0)
    iterations.append(iter)

print('(Array de ceros) Valor medio de iteraciones necesario para converger: {}'.format(np.mean(np.asarray(iterations))))


iterations = []
for i in range(0, 10):
    w_random = np.random.uniform(low=-1, high=1, size=3)
    w, iter = ajusta_PLA(x, y, 1000, w_random)
    iterations.append(iter)

print('(Array aleatorio) Valor medio de iteraciones necesario para converger: {}'.format(np.mean(np.asarray(iterations))))

##input("\n--- Pulsar tecla para continuar ---\n")



# Ahora con los datos del ejercicio 1.2.b

# Random initializations
iterations = []
for i in range(0, 10):
    w, iter = ajusta_PLA(x, ruido, 1000, w_0)
    iterations.append(iter)

print('(Array de ceros) Valor medio de iteraciones necesario para converger: {}'.format(np.mean(np.asarray(iterations))))


iterations = []
for i in range(0, 10):
    w_random = np.random.uniform(low=-1, high=1, size=3)
    w, iter = ajusta_PLA(x, ruido, 1000, w_random)
    iterations.append(iter)

print('(Array aleatorio) Valor medio de iteraciones necesario para converger: {}'.format(np.mean(np.asarray(iterations))))


##input("\n--- Pulsar tecla para continuar ---\n")


###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 3: REGRESIÓN LOGÍSTICA CON STOCHASTIC GRADIENT DESCENT

def sgdLR(x, y, initial_point, eta):
    w = np.copy(initial_point)

    # Calculamos el gradiente descendente
    while True:
        # Obtenemos un vector aleatorio de índices de tamaño 64
        index = np.random.permutation(x.shape[0])
        # Asignamos los valores de los índices de 'x' e 'y' en los minibatches
        minibatch_x = x[index,:]
        minibatch_y = y[index]

        w_ant = np.copy(w)

        for xn, yn in zip(minibatch_x, minibatch_y):
            gradient = sigmoid_clasifier(xn, yn, w)
            w -= eta * gradient

        if np.linalg.norm(w_ant - w) < 0.01:
            break

    return w.reshape(-1,)


def sigmoid_clasifier(x, y, w):
    x = x.reshape(1, 3)
    z = np.dot(x, w)
    h = 1 / (1 + np.exp(-z))
    if h >= 0.5:
        return np.dot(x.T, 1)
    else:
        return np.dot(x.T, -1)


def estimar_Eout(x, y, w):
    Eout = 0
    for xn, yn in zip(x,y):
        xn = xn.reshape(1, 3)
        z = yn*w.T*xn
        Eout += np.log(1 + np.exp(-(z)))
    return -(Eout/y.shape[0])



sgdLR_w = sgdLR(x, y, np.zeros((3,1)), 0.01)
sgdLR_x = np.linspace(-50, 50, y.size)
sgdLR_y = (-sgdLR_w[0] - sgdLR_w[1]*sgdLR_x) / sgdLR_w[2]

# Mostramos la gráfica por pantalla y el error
plt.scatter(x[:,1], x[:,2], c=y)
plt.plot(sgdLR_x, sgdLR_y, 'r-', linewidth=2)
plt.show()

#input("\n--- Pulsar tecla para continuar ---\n")

# Usar la muestra de datos etiquetada para encontrar nuestra solución g y estimar Eout
# usando para ello un número suficientemente grande de nuevas muestras (>999).


x_test = simula_unif(1000, 2, [-50, 50])
x_test = np.c_[np.ones((x_test.shape[0], 1), np.float64), x_test]

y_test = []
for i in range(x_test.shape[0]):
    y_test.append(f(x_test[i, 0], x_test[i, 1], a, b))
y_test = np.array(y_test)

print("Eout: ", estimar_Eout(x_test, y_test, sgdLR_w))


#input("\n--- Pulsar tecla para continuar ---\n")


###############################################################################
###############################################################################
###############################################################################
