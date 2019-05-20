"""
TRABAJO 3
Nombre Estudiante: Jose Maria Sanchez Guerrero
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler, normalize
from sklearn import svm, metrics, linear_model, neighbors
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Fijamos la semilla
np.random.seed(76067801)


# ----------------------------------------------------------------------------------------- #
#############################################################################################
##################### OPTICAL RECOGNITION OF HANDWRITTEN DIGITS DATA SET ####################
#############################################################################################
# ----------------------------------------------------------------------------------------- #
# Leemos a partir del fichero
df = pd.read_csv('datos/optdigits.tra')

# Guardamos en una variable auxiliar los datos pero sin el
# valor de las etiquetas
df_aux = df.copy()
df_aux = df_aux.iloc[:, :-1]

# Asignamos un escalador y lo aplicamos al conjunto de características
scaled_df = df_aux.copy()
scaler = MaxAbsScaler()
scaled = scaler.fit_transform(df_aux)
scaled_df.loc[:,:] = scaled

# Metemos en la variables el nuevo conjunto escalado
trainX_optdigits = np.array(scaled_df)[:,:-1]
trainY_optdigits = np.array(df)[:,-1:]


# Leemos a partir del fichero
df = pd.read_csv('datos/optdigits.tes')

# Guardamos en una variable auxiliar los datos pero sin el
# valor de las etiquetas
df_aux = df.copy()
df_aux = df_aux.iloc[:, :-1]

# Asignamos un escalador y lo aplicamos al conjunto de características
scaled_df = df_aux.copy()
scaler = MaxAbsScaler()
scaled = scaler.fit_transform(df_aux)
scaled_df.loc[:,:] = scaled

# Metemos en la variables el nuevo conjunto escalado
testX_optdigits = np.array(scaled_df)[:,:-1]
testY_optdigits = np.array(df)[:,-1:]



# ----------------------------------------------------------------------------------------- #
def analisisClasificacion(expected, predicted):
    print("Classification report:\n%s\n"
          % (metrics.classification_report(expected, predicted)))

    print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))



# ----------------------------------------------------------------------------------------- #
def Perceptron(penality, max_iter, tol):
    print('\n\n\nPERCEPTRON')
    # Creamos el clasificador y lo asignamos a una variable
    classifier = linear_model.Perceptron(penalty=penality, max_iter=max_iter, tol=tol)

    # Entrenamos nuestro clasificador con los datos de entrenamiento
    classifier.fit(trainX_optdigits, trainY_optdigits.ravel())

    # Creamos dos variables: valor esperado y el que predecimos
    # gracias al vector de entrenamiento
    expected = testY_optdigits
    predicted = classifier.predict(testX_optdigits)

    analisisClasificacion(expected, predicted)

Perceptron('l2', 15000, 1e-4)
input("\n--- Pulsar tecla para continuar ---\n")



# ----------------------------------------------------------------------------------------- #
def RegresionLogistica(penality, max_iter):
    print('\n\n\nREGRESION LOGISTICA')
    # Creamos el clasificador y lo asignamos a una variable
    classifier = linear_model.LogisticRegression(penalty=penality, max_iter=max_iter, solver='liblinear', multi_class='ovr')

    # Entrenamos nuestro clasificador con los datos de entrenamiento
    classifier.fit(trainX_optdigits, trainY_optdigits.ravel())

    # Creamos dos variables: valor esperado y el que predecimos
    # gracias al vector de entrenamiento
    expected = testY_optdigits
    predicted = classifier.predict(testX_optdigits)

    # Funcion que muestra por pantalla los resultados
    analisisClasificacion(expected, predicted)

RegresionLogistica('l2', 15000)
input("\n--- Pulsar tecla para continuar ---\n")



# ----------------------------------------------------------------------------------------- #
def SGDClassifier(penality, max_iter, alpha, tol):
    print('\n\n\nGRADIENTE DESCENDENTE ESTOCASTICO (Classifier)')
    # Creamos el clasificador y lo asignamos a una variable
    classifier = linear_model.SGDClassifier(penalty=penality, max_iter=max_iter, alpha=alpha, tol=tol)

    # Entrenamos nuestro clasificador con los datos de entrenamiento
    classifier.fit(trainX_optdigits, trainY_optdigits.ravel())

    # Creamos dos variables: valor esperado y el que predecimos
    # gracias al vector de entrenamiento
    expected = testY_optdigits
    predicted = classifier.predict(testX_optdigits)

    # Funcion que muestra por pantalla los resultados
    analisisClasificacion(expected, predicted)

SGDClassifier('l2', 15000, 0.001, 1e-4)
input("\n--- Pulsar tecla para continuar ---\n")



# ----------------------------------------------------------------------------------------- #
def LinearSVC(penality, C, max_iter, tol):
    print('\n\n\nSUPPORT VECTOR MACHINE (SVC)')
    # Creamos el clasificador y lo asignamos a una variable
    classifier = svm.LinearSVC(penalty=penality, C=C, max_iter=max_iter, tol=tol)

    # Entrenamos nuestro clasificador con los datos de entrenamiento
    classifier.fit(trainX_optdigits, trainY_optdigits.ravel())

    # Creamos dos variables: valor esperado y el que predecimos
    # gracias al vector de entrenamiento
    expected = testY_optdigits
    predicted = classifier.predict(testX_optdigits)

    # Funcion que muestra por pantalla los resultados
    analisisClasificacion(expected, predicted)

LinearSVC('l2', 1.0, 15000, 1e-4)
input("\n--- Pulsar tecla para continuar ---\n")






# ----------------------------------------------------------------------------------------- #
#############################################################################################
################################# AIRFOIL SELF-NOISE DATA SET ###############################
#############################################################################################
# ----------------------------------------------------------------------------------------- #
data = pd.read_csv('datos/airfoil_self_noise.dat', delim_whitespace=True)
df = pd.DataFrame(data)

# Guardamos en una variable auxiliar los datos pero sin el
# valor de las etiquetas
df_aux = df.copy()
df_aux = df_aux.iloc[:, :-1]

# Asignamos un escalador y lo aplicamos al conjunto de características
scaled_df = df_aux.copy()
scaler = MaxAbsScaler()
scaled = scaler.fit_transform(df_aux)
scaled_df.loc[:,:] = scaled

# Metemos en la variables el nuevo conjunto escalado
X_airfoil = np.array(scaled_df)[:,:-1]
y_airfoil = np.array(df)[:,-1:]


trainX_airfoil, testX_airfoil, trainY_airfoil, testY_airfoil = \
    train_test_split(X_airfoil, y_airfoil, test_size=0.2, shuffle=True)

X_airfoil = normalize(X_airfoil)




# ----------------------------------------------------------------------------------------- #
def analisisRegresion(expected, predicted):
    print("Mean Absolut Error: %s"
          % (metrics.mean_absolute_error(expected, predicted)))

    print("Mean Squared Error: %s"
          % (metrics.mean_squared_error(expected, predicted)))

    linea = np.linspace(100, 150, predicted.shape[0])

    # Mostramos la gráfica por pantalla
    plt.scatter(expected, predicted)
    plt.plot(linea, linea, 'r-', linewidth=2)
    plt.xlabel('Predicted (dB)')
    plt.ylabel('Expected (dB)')
    plt.show()



# ----------------------------------------------------------------------------------------- #
def RegresionLineal():
    print('\n\n\nREGRESION LINEAL')
    # Creamos el regresor y lo asignamos a una variable
    regressor = linear_model.LinearRegression()

    # Lo entrenamos con los datos de entrenamiento
    regressor.fit(trainX_airfoil, trainY_airfoil.ravel())

    # Creamos dos variables: valor esperado y el que predecimos
    # gracias al vector de entrenamiento
    expected = testY_airfoil
    predicted = regressor.predict(testX_airfoil)

    # Funcion que muestra por pantalla los resultados
    analisisRegresion(expected, predicted)

RegresionLineal()
input("\n--- Pulsar tecla para continuar ---\n")



# ----------------------------------------------------------------------------------------- #
def SGDRegressor(penality, max_iter, alpha, tol):
    print('\n\n\nGRADIENTE DESCENDENTE ESTOCASTICO')
    # Creamos el regresor y lo asignamos a una variable
    regressor = linear_model.SGDRegressor(penalty=penality, max_iter=max_iter, alpha=alpha, tol=tol)

    # Lo entrenamos con los datos de entrenamiento
    regressor.fit(trainX_airfoil, trainY_airfoil.ravel())

    # Creamos dos variables: valor esperado y el que predecimos
    # gracias al vector de entrenamiento
    expected = testY_airfoil
    predicted = regressor.predict(testX_airfoil)

    # Funcion que muestra por pantalla los resultados
    analisisRegresion(expected, predicted)

SGDRegressor('l2', 15000, 0.001, 1e-4)
input("\n--- Pulsar tecla para continuar ---\n")



# ----------------------------------------------------------------------------------------- #
def LinearSVR(C, max_iter, tol):
    print('\n\n\nSUPPORT VECTOR MACHINE (SVR)')
    # Creamos el regresor y lo asignamos a una variable
    regressor = svm.LinearSVR(C=C, max_iter=max_iter, tol=tol)

    # Lo entrenamos con los datos de entrenamiento
    regressor.fit(trainX_airfoil, trainY_airfoil.ravel())

    # Creamos dos variables: valor esperado y el que predecimos
    # gracias al vector de entrenamiento
    expected = testY_airfoil
    predicted = regressor.predict(testX_airfoil)

    # Funcion que muestra por pantalla los resultados
    analisisRegresion(expected, predicted)

LinearSVR(1.0, 15000, 1e-4)
input("\n--- Pulsar tecla para continuar ---\n")





# ----------------------------------------------------------------------------------------- #
#############################################################################################
########################### MODIFICACIONES DE METODOS Y PARAMETROS ##########################
#############################################################################################
# ----------------------------------------------------------------------------------------- #
SGDClassifier('l1', 15000, 0.001, 1e-4)
SGDClassifier('l2', 15000, 0.0001, 1e-5)

LinearSVR(5.0, 15000, 1e-5)
LinearSVR(0.1, 1000, 1e-3)




# ----------------------------------------------------------------------------------------- #
print('\n\n\nkNN VECINO MAS CERCANO')
# Creamos el clasificador y lo asignamos a una variable
classifier = neighbors.KNeighborsClassifier()

# Entrenamos nuestro clasificador con los datos de entrenamiento
classifier.fit(trainX_optdigits, trainY_optdigits.ravel())

# Creamos dos variables: valor esperado y el que predecimos
# gracias al vector de entrenamiento
expected = testY_optdigits
predicted = classifier.predict(testX_optdigits)

# Funcion que muestra por pantalla los resultados
analisisClasificacion(expected, predicted)

input("\n--- Pulsar tecla para continuar ---\n")



# ----------------------------------------------------------------------------------------- #
print('\n\n\nkNN VECINO MAS CERCANO')
# Creamos el regresor y lo asignamos a una variable
regressor = neighbors.KNeighborsRegressor()

# Lo entrenamos con los datos de entrenamiento
regressor.fit(trainX_airfoil, trainY_airfoil.ravel())

# Creamos dos variables: valor esperado y el que predecimos
# gracias al vector de entrenamiento
expected = testY_airfoil
predicted = regressor.predict(testX_airfoil)

# Funcion que muestra por pantalla los resultados
analisisRegresion(expected, predicted)

input("\n--- Pulsar tecla para continuar ---\n")




