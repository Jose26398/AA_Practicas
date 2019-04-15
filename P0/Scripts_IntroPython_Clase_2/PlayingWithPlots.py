"""
===============================
Legend using pre-defined labels
===============================

Defining legend labels with plots.
"""


import numpy as np
import matplotlib.pyplot as plt

# Make some fake data.
x = np.linspace(-10, 10, 100)
y1 = np.exp(x)
y2 = np.log(x)

# Create plots with pre-defined labels.
fig, ax = plt.subplots()
ax.plot(x, y1, 'k--', label='y = exp(x)')
ax.plot(x, y2, 'k:', label='y = ln(x)')

legend = ax.legend(loc='lower right', shadow=True, fontsize='x-large')

ax.set_ylim((-5, 5))

# Put a nicer background color on the legend.
legend.get_frame().set_facecolor('C0')
plt.grid()
plt.show()

#############################################################################


max_val = 5.
t = np.arange(0., max_val+0.5, 0.5)
plt.plot(t, t, 'r-', label='linear')
plt.plot(t, t**2, 'b--', label='quadratic')
plt.plot(t, t**3, 'g-.', label='cubic')
plt.plot(t, 2**t, 'y:', label='exponential')

plt.xlabel('X axis')
plt.ylabel('Y axis')

plt.title('Several lines together')
plt.legend()
plt.axis([0, max_val, 0, 2**max_val])
plt.show()

#############################################################################

max_val = 5.
t = np.arange(0., max_val+0.5, 0.5)
ax = plt.subplot('211') #Crear dos figuras (2 filas): una encima de la otra (1 col)
ax.set_title('Linear')
ax.plot(t, t, 'r-')
ax.set_ylabel('Y axis')
ax.axis([0, max_val, 0, max_val])

ax = plt.subplot('212') #Crear segunda figura
ax.set_title('Quadratic')
ax.plot(t, t**2, 'b--')
ax.set_ylabel('Y axis')
ax.axis([0, max_val, 0, max_val**2])

plt.tight_layout() #Dejar espacio entre figuras
plt.show()
