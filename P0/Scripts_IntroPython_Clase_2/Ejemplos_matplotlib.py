# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

y = [4, 1, 2, 5, 8.7]
x = range(1, len(y)+1)

plt.plot(x, y)
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.title('First test')
plt.show()



import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)
plt.plot(x, y)
plt.show()