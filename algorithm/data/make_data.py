#coding: utf-8

import numpy as np

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


x = np.linspace(-3,3,120)
y = gaussian(x, 0,1)

for idx in range(len(x)):
    print "%s,%s" %(x[idx], y[idx] + np.random.rand()*0.1)
