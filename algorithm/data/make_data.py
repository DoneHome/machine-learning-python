#coding: utf-8

import numpy as np

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def gen_lwr_data():
    x = np.linspace(-3,3,120)
    y = gaussian(x, 0,1)

    for idx in range(len(x)):
        print "%s,%s" %(x[idx], y[idx] + np.random.rand()*0.1)

def gen_gda_data():
    """
    multivariate gaussian
    """
    mu0 = np.array([1, 2])
    sigma0 = np.array([[1,0],[0,1]])
    x0 = np.random.multivariate_normal(mu0, sigma0, 50)

    for idx in range(x0.shape[0]):
        print "%s,%s,%s" %(x0[idx][0], x0[idx][1], 0)

    mu1 = np.array([7, 8])
    sigma1 = np.array([[1,0],[0,1]])
    x1 = np.random.multivariate_normal(mu1, sigma1, 50)

    for idx in range(x1.shape[0]):
        print "%s,%s,%s" %(x1[idx][0], x1[idx][1], 1)

if __name__ == "__main__":
    gen_gda_data()
