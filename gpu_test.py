# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 14:25:42 2020

@author: dolor
"""

from numba import guvectorize, cuda, int64
import numpy as np

a = np.arange(5)

@guvectorize([(int64[:], int64, int64[:])], '(n),()->(n)', target='cuda')
def g(x, y, res):
    for i in range(x.shape[0]):
        res[i] = x[i] + y

print(g(a, 2, a))