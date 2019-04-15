# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 11:06:06 2019

@author: Pablo
"""

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

c = 1
clase2 = Clase2(a=1)
d = clase2.llamar(c)
print(d)

###############################################################


class Car(object):
	"""
		blueprint for car
	"""

	def __init__(self, model, color, company, speed_limit):
		self.color = color
		self.company = company
		self.speed_limit = speed_limit
		self.model = model

	def start(self):
		print('started')

	def stop(self):
		print('stopped')

	def accelarate(self):
		print('accelarating...')
		'accelarator functionality here'

	def change_gear(self, gear_type):
		print('gear changed')
		'gear related functionality here'
        
maruthi_suzuki = Car("ertiga", "black", "suzuki", 60)
audi = Car("A6", "red", "audi", 80)

###############################################################


class Rectangle:
   def __init__(self, length, breadth, unit_cost=0):
       self.length = length
       self.breadth = breadth
       self.unit_cost = unit_cost
   
   def get_perimeter(self):
       return 2 * (self.length + self.breadth)
   
   def get_area(self):
       return self.length * self.breadth
   
   def calculate_cost(self):
       area = self.get_area()
       return area * self.unit_cost
# breadth = 120 cm, length = 160 cm, 1 cm^2 = Rs 2000
r = Rectangle(160, 120, 2000)
print("Area of Rectangle: %s cm^2" % (r.get_area()))
print("Cost of rectangular field: Rs. %s " %(r.calculate_cost()))

###############################################################

class Restaurant(object):
    bankrupt = False
    def open_branch(self):
        if not self.bankrupt:
            print("branch opened")
            
            
x = Restaurant()
x.bankrupt
Restaurant().bankrupt

y = Restaurant()
y.bankrupt = True
y.bankrupt
x.bankrupt


class Restaurant(object):
    bankrupt = False
    def open_branch(this):
        if not this.bankrupt:
            print("branch opened")
            
x = Restaurant()
x.bankrupt
Restaurant().bankrupt

y = Restaurant()
y.bankrupt = True
y.bankrupt
x.bankrupt


###############################################################


class Point(object):
    def __init__(self,x = 0,y = 0):
        self.x = x
        self.y = y

    def distance(self):
        """Find distance from origin"""
        return (self.x**2 + self.y**2) ** 0.5
    
p1 = Point(6,8)

p1.distance()

Point.distance(p1)

type(Point.distance)

type(p1.distance)



###############################################################


class A(object):

    @staticmethod
    def stat_meth():
        print("Look no self was passed")
        
a = A()

a.stat_meth()

type(A.stat_meth)

type(a.stat_meth)