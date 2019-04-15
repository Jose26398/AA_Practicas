# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 14:10:05 2019

@author: PabloMesejo
"""

def contador(max):
    n=0
    while n < max:
          yield n 
          n=n+1 
          
contad = contador(10)
for i in contad:
    print("valor: "+str(i))

for i in contad:
    print("valor: "+str(i))






# https://pythontips.com/2013/09/29/the-python-yield-keyword-explained/
    # https://es.stackoverflow.com/questions/6048/cu%C3%A1l-es-el-funcionamiento-de-yield-en-python
