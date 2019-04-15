# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import numpy as np

mu, sigma = 0.5, 0.1
s = np.random.normal(mu, sigma, 1000)

# Create the bins and histogram
count, bins, ignored = plt.hist(s, 20, density=True)



################################

import numpy as np
#Draw samples from the distribution:
n, p = 10, .5  # number of trials, probability of each trial
s = np.random.binomial(n, p, 100)
# result of flipping a coin 10 times, tested 25 times.
s

#What is the probability of getting 3 or less heads
sum(s <= 3)/100
# answer = 0.17, or 17%.

######################################%
import numpy as np
np.random.randint(1,5)

type(np.random.randint(5))

np.random.randint(1,5, size=(3,2))


np.random.choice(5, 3)
#This is equivalent to np.random.randint(0,5,3)

np.random.choice(5, 3, p=[0.1, 0, 0.3, 0.6, 0])


np.random.choice(5, 3, replace=False, p=[0, 0, 0.4, 0.6, 0])