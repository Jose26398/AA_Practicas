# -*- coding: utf-8 -*-

# https://pandas.pydata.org/pandas-docs/version/0.23.4/generated/pandas.DataFrame.html 
import pandas as pd
import numpy as np

d1 = {'col1': [1, 2], 'col2': [3, 4]}
d1

df = pd.DataFrame(data=d1)
df

df.dtypes

d2 = np.random.randint(low=0, high=10, size=(5, 5))
df2 = pd.DataFrame(d2, columns=['a', 'b', 'c', 'd', 'e'])
df2