# -*- coding: utf-8 -*-

import numpy as np

m = np.arange(0,6).reshape(2,3)

print('Mostrar la primera fila')
print(m[0,:])

print('Mostrar las columnas pares')
print(m[:,::2])

print('Mostrar la esquina inferior derecha')
print(m[-1,-1])



m[m<3]=0
print('Todos los elementos menores a 3 son 0 ahora')
print(m)


