"""
TRABAJO 3
Nombre Estudiante: Jose Maria Sanchez Guerrero
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler, normalize
from sklearn import svm, metrics, linear_model, neighbors
from sklearn.model_selection import train_test_split, cross_val_score
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

    print("Classification report for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(expected, predicted)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

    print("\nMean Absolut Error: %s"
          % (metrics.mean_absolute_error(expected, predicted)))

    print("Mean Squared Error: %s"
          % (metrics.mean_squared_error(expected, predicted)))

Perceptron('l2', 15000, 1e-4)
input("\n--- Pulsar tecla para continuar ---\n")



# ----------------------------------------------------------------------------------------- #
def RegrsionLogistica(penality, max_iter):
    print('\n\n\nREGRESION LOGISTICA')
    # Creamos el clasificador y lo asignamos a una variable
    classifier = linear_model.LogisticRegression(penalty=penality, max_iter=max_iter, solver='liblinear', multi_class='ovr')

    # Entrenamos nuestro clasificador con los datos de entrenamiento
    classifier.fit(trainX_optdigits, trainY_optdigits.ravel())

    # Creamos dos variables: valor esperado y el que predecimos
    # gracias al vector de entrenamiento
    expected = testY_optdigits
    predicted = classifier.predict(testX_optdigits)

    print("Classification report for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(expected, predicted)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

    print("\nMean Absolut Error: %s"
          % (metrics.mean_absolute_error(expected, predicted)))

    print("Mean Squared Error: %s"
          % (metrics.mean_squared_error(expected, predicted)))

RegrsionLogistica('l2', 15000)
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

    print("Classification report for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(expected, predicted)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

    print("\nMean Absolut Error: %s"
          % (metrics.mean_absolute_error(expected, predicted)))

    print("Mean Squared Error: %s"
          % (metrics.mean_squared_error(expected, predicted)))

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

    print("Classification report for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(expected, predicted)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

    print("\nMean Absolut Error: %s"
          % (metrics.mean_absolute_error(expected, predicted)))

    print("Mean Squared Error: %s"
          % (metrics.mean_squared_error(expected, predicted)))

LinearSVC('l2', 1.0, 15000, 1e-4)
input("\n--- Pulsar tecla para continuar ---\n")



# ----------------------------------------------------------------------------------------- #
print('\n\n\nkNN VECINO MAS CERCANO')
# Creamos el clasificador y lo asignamos a una variable
classifier = neighbors.KNeighborsClassifier(n_neighbors=1)

# Entrenamos nuestro clasificador con los datos de entrenamiento
classifier.fit(trainX_optdigits, trainY_optdigits.ravel())

# Creamos dos variables: valor esperado y el que predecimos
# gracias al vector de entrenamiento
expected = testY_optdigits
predicted = classifier.predict(testX_optdigits)

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

print("\nMean Absolut Error: %s"
      % (metrics.mean_absolute_error(expected, predicted)))

print("Mean Squared Error: %s"
      % (metrics.mean_squared_error(expected, predicted)))

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
print('\n\n\nREGRESION LINEAL')
# Create a classifier: a support vector classifier
regressor = linear_model.LinearRegression()

# We learn the digits on the first half of the digits
regressor.fit(trainX_airfoil, trainY_airfoil.ravel())

# Now predict the value of the digit on the second half:
expected = testY_airfoil
predicted = regressor.predict(testX_airfoil)

print("Mean Absolut Error: %s"
      % (metrics.mean_absolute_error(expected, predicted)))

print("Mean Squared Error: %s"
      % (metrics.mean_squared_error(expected, predicted)))


# ----------------------------------------------------------------------------------------- #
print('\n\n\nGRADIENTE DESCENDENTE ESTOCASTICO')
# Create a classifier: a support vector classifier
regressor = linear_model.SGDRegressor(penalty='l2', max_iter=15000, alpha=0.001, tol=1e-3)

# We learn the digits on the first half of the digits
regressor.fit(trainX_airfoil, trainY_airfoil.ravel())

# Now predict the value of the digit on the second half:
expected = testY_airfoil
predicted = regressor.predict(testX_airfoil)

print("Mean Absolut Error: %s"
      % (metrics.mean_absolute_error(expected, predicted)))

print("Mean Squared Error: %s"
      % (metrics.mean_squared_error(expected, predicted)))



# ----------------------------------------------------------------------------------------- #
print('\n\n\nSUPPORT VECTOR MACHINE (SVM)')
# Create a classifier: a support vector classifier
regressor = svm.LinearSVR(C=1.0, max_iter=15000, epsilon=0)

# We learn the digits on the first half of the digits
regressor.fit(trainX_airfoil, trainY_airfoil.ravel())

# Now predict the value of the digit on the second half:
expected = testY_airfoil
predicted = regressor.predict(testX_airfoil)

print("Mean Absolut Error: %s"
      % (metrics.mean_absolute_error(expected, predicted)))

print("Mean Squared Error: %s"
      % (metrics.mean_squared_error(expected, predicted)))



# ----------------------------------------------------------------------------------------- #
print('\n\n\nkNN VECINO MAS CERCANO')
# Create a classifier: a support vector classifier
regressor = neighbors.KNeighborsRegressor(n_neighbors=1)

# We learn the digits on the first half of the digits
regressor.fit(trainX_airfoil, trainY_airfoil.ravel())

# Now predict the value of the digit on the second half:
expected = testY_airfoil
predicted = regressor.predict(testX_airfoil)


print("Mean Absolut Error: %s"
      % (metrics.mean_absolute_error(expected, predicted)))

print("Mean Squared Error: %s"
      % (metrics.mean_squared_error(expected, predicted)))




# ----------------------------------------------------------------------------------------- #
#############################################################################################
########################### MODIFICACIONES DE METODOS Y PARAMETROS ##########################
#############################################################################################
# ----------------------------------------------------------------------------------------- #



# ----------------------------------------------------------------------------------------- #
print('\n\n\nSUPPORT VECTOR MACHINE (SVM)')
# Creamos el clasificador y lo asignamos a una variable
classifier = svm.LinearSVC(penalty='l2', C=1.0, max_iter=15000, tol=1e-4)
results = cross_val_score(regressor, X_airfoil, y_airfoil.ravel(), cv=10)

# Entrenamos nuestro clasificador con los datos de entrenamiento
classifier.fit(trainX_optdigits, trainY_optdigits.ravel())

# Creamos dos variables: valor esperado y el que predecimos
# gracias al vector de entrenamiento
expected = testY_optdigits
predicted = classifier.predict(testX_optdigits)

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

print("\nMean Absolut Error: %s"
      % (metrics.mean_absolute_error(expected, predicted)))

print("Mean Squared Error: %s"
      % (metrics.mean_squared_error(expected, predicted)))

print("\nMean: ", abs(results.mean()))
print("Desviation: ", np.std(results))


# ----------------------------------------------------------------------------------------- #
LinearSVC('l2', 1.0, 15000, 1e-4)
LinearSVC('l1', 1.0, 15000, 1e-4)
LinearSVC('l2', 0.1, 15000, 1e-4)
LinearSVC('l1', 0.1, 15000, 1e-4)


