"""
TRABAJO 3
Nombre Estudiante: Jose Maria Sanchez Guerrero
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn import svm, metrics, datasets, linear_model
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt


# Fijamos la semilla
np.random.seed(1)


#############################################################################################
##################### OPTICAL RECOGNITION OF HANDWRITTEN DIGITS DATA SET ####################
#############################################################################################
df = pd.read_csv('datos/optdigits.tra')
trainX_optdigits = np.array(df)[:,:-1]
trainY_optdigits = np.array(df)[:,-1:]

df = pd.read_csv('datos/optdigits.tes')
testX_optdigits = np.array(df)[:,:-1]
testY_optdigits = np.array(df)[:,-1:]


# trainX_optdigits = preprocessing.normalize(trainX_optdigits)
# trainX_optdigits = PCA(n_components=trainX_optdigits.shape[1]).fit_transform(trainX_optdigits)



print('\n\n\nPERCEPTRON')
# Create a classifier: a support vector classifier
classifier = linear_model.Perceptron(max_iter=1000, tol=1e-3)

# We learn the digits on the first half of the digits
classifier.fit(trainX_optdigits, trainY_optdigits.ravel())

# Now predict the value of the digit on the second half:
expected = testY_optdigits
predicted = classifier.predict(testX_optdigits)

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))



print('\n\n\nGRADIENTE DESCENDENTE ESTOCASTICO')
# Create a classifier: a support vector classifier
classifier = linear_model.SGDClassifier(max_iter=1000, tol=1e-3)

# We learn the digits on the first half of the digits
classifier.fit(trainX_optdigits, trainY_optdigits.ravel())

# Now predict the value of the digit on the second half:
expected = testY_optdigits
predicted = classifier.predict(testX_optdigits)

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))



# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma=0.001, kernel='linear')

# We learn the digits on the first half of the digits
classifier.fit(trainX_optdigits, trainY_optdigits.ravel())

# Now predict the value of the digit on the second half:
expected = testY_optdigits
predicted = classifier.predict(testX_optdigits)

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))


# # Airfoil Self-Noise Data Set
# data = np.loadtxt('datos/airfoil_self_noise.dat')
#
# skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=1)
# train_airfoil = []
# test_airfoil = []
#
# x = data[:,:data.shape[1]-1]
# y = data[:,data.shape[1]-1]
#
# for train_index, test_index in skf.split(x, y):
#    x_train, y_train = x[train_index], y[train_index]
#    x_test, y_test   = x[test_index], y[test_index]
#
#    train_airfoil.append([x_train, y_train])
#    test_airfoil.append([x_test, y_test])
#
#
#
# print('\n\n\nREGRESION LINEAL')
# # Create a classifier: a support vector classifier
# classifier = linear_model.LinearRegression()
#
# # We learn the digits on the first half of the digits
# classifier.fit(trainX_optdigits, trainY_optdigits)
#
# # Now predict the value of the digit on the second half:
# expected = testY_optdigits
# predicted = classifier.predict(testX_optdigits)
#
# print("Classification report for classifier %s:\n%s\n"
#       % (classifier, metrics.classification_report(expected, predicted)))
# print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
#
#
#
# print('\n\n\nREGRESION LOGISTICA')
# # Create a classifier: a support vector classifier
# classifier = linear_model.LogisticRegression(max_iter=1000, tol=1e-3)
#
# # We learn the digits on the first half of the digits
# classifier.fit(trainX_optdigits, trainY_optdigits)
#
# # Now predict the value of the digit on the second half:
# expected = testY_optdigits
# predicted = classifier.predict(testX_optdigits)
#
# print("Classification report for classifier %s:\n%s\n"
#       % (classifier, metrics.classification_report(expected, predicted)))
# print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
#

