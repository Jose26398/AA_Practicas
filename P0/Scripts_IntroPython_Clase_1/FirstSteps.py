# -*- coding: utf-8 -*-

#Variables
entero1 = 5 #Int
entero2 = 505 #Int
flotante = 50.5 #Float
boolean_t = True #Boolean
boolean_f = False #Boolean
string1 = 'String1' #String
string2 = 'String2' #String


#Operaciones aritméticas
suma = entero1 + flotante
resta = entero1 - flotante
producto = entero1 * flotante
print(producto)
division = entero1 / flotante
print(division)
division_entera = entero2 // entero1
print(division_entera)
resto = entero2 % entero1
print(resto)


#Operaciones lógicas
igual = entero2==(500 + 5)
no_igual = entero1 != suma
mayor = entero2 > entero1 #>=
menor = entero1 < entero2 # <=
and_logico = igual and mayor
or_logico = igual or no_igual

#Cambiar tipos
entero2flotante = float(entero1)
flotante2entero = int(flotante)
astring = str(entero2)
abool = bool(entero1)

#Strings
formatear = 'String con entero %d, flotante %f y string %s' % (entero1, 
                                                               flotante, 
                                                               string1)
concatenar = string1 + str(entero1)

#Mostrar por pantalla
print('Dos de los strings:', string1, string2)
print('String y entero:', concatenar, entero1)


########################################################

#Declarar tupla
tupla = (5, 't1', True, 0.5)
print(tupla)

#Declarar lista
lista = [5, 't1', True, 0.5]
print(lista)

#Obtener tamaño lista y tupla
l_tupla = len(tupla)
l_lista = len(lista)
print(l_tupla)
print(l_lista)

print(tupla[2])
print(lista[2])

lista[2] = 1000
print(lista)


#Añadir elemento
lista.append(False) #Al final
print(lista)
position = 1
lista.insert(position, 't21') #En una posición concreta
print(lista)

#Eliminar elemento
lista.remove('t1') #Buscando
print(lista)
lista.pop() #Al final
print(lista)
lista.pop(1) #En la posición 1
print(lista)

#Concatenar
lista2 = ['a', 'b', 'c']
lista_combinada = lista + lista2 #Pone lista2 al final de lista
print(lista_combinada)

#Copiar
lista_copia = lista.copy()
print(lista_copia)


#Declarar
diccionario = {'a': 1, 'b': 2.0}

#Añadir elemento
diccionario['c'] = False

#Mostrar por pantalla
print(diccionario)

#Eliminar elemento
del diccionario['c']
print(diccionario)

#Keys
print(diccionario.keys())

#Values
print(diccionario.values())



###################################

#Condicional
if condicion:
    #Hacer algo
elif otra_condicion:
    #Hacer algo
else:
    #Hacer algo

#Bucle for
for i in range(inicio, fin, paso):
    #Hacer algo   
    print(i)
    
for elemento in lista:
    #Hacer algo

#Bucle while
while condicion:
    #Hacer algo


####################################
    
def funcion(a, b=1):
    c = a + b
    
    #return c #Opcional

c = funcion(1,2) #O funcion(a=1, b=2)
print(c)
c_def = funcion(1)
print(c_def)


def funcion2(a, b=1):
    a = a * b
    c = a + b
    
    return [a,c]
    
a = 1;
c = 10;
a = funcion2(a,2) #O funcion(a=1, b=2)
print(a)
print(c)


####################################

some_guy = 'Fred'
first_names = []
first_names.append(some_guy)
print(first_names)

another_list_of_names = first_names
another_list_of_names.append('George')
some_guy = 'Bill'

print(some_guy, first_names, another_list_of_names)

another_list_of_names2 = first_names.copy()
another_list_of_names2.append('Pablo')

print(some_guy, first_names, another_list_of_names, another_list_of_names2)

####################################
    
def func(a, b):
    a = 'new-value'        # a and b are local names
    b = b + 1              # assigned to new objects
    return 1.0, 2.0            # return new values

x, y = 'old-value', 99
x, y = func(x, y)
print(x, y)
    
    
def spam(eggs):
        eggs.append(1)
        eggs = [2, 3]
        print(eggs)

ham = [0]
spam(ham)
print(ham)



def foo(bar):
    bar.append(42)
    print(bar)
    
answer_list = []
foo(answer_list)
print(answer_list)    


def foo(bar):
    bar = 'new value'
    print (bar)
    # >> 'new value'

answer_list = 3
foo(answer_list)
print(answer_list)
# >> 'old value'



################################
class Clase():
    def __init__(self, a):
        self.a = a
        
    def llamar(self, b):
        return self.a*b
    
class Clase2(Clase):
    def __init__(self, a, b=2.0):
        super().__init__(a)
        self.b=b
        
    def llamar(self, c):
        return self.a*self.b*c
    
    def __call__(self, c):
        return self.llamar(c)
    
c = 3
clase2 = Clase2(a = 1)
d = clase2.llamar(c)
print(d)



class Animal(object):
    def __init__(self, species, age): # Constructor `a = Animal(‘bird’, 10)`
        self.species = species # Refer to instance with `self`
        self.age = age # All instance variables are public
    
    def isPerson(self): # Invoked with `a.isPerson()`
        return self.species == 'Homo Sapiens'

    def ageOneYear(self):
        self.age += 1

class Dog(Animal): # Inherits Animal’s methods
    def ageOneYear(self): # Override for dog years
        self.age += 7

a = Dog(species = 'labrador', age = 6)
a.isPerson()
a.ageOneYear()
print(a.age)
    
    
