# coding:utf-8
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = np.genfromtxt("./gaussian_discriminant_analysis.csv", dtype=np.float64, delimiter=",")
fig = plt.figure()
#ax = Axes3D(fig)

class GDA():
    """
    Gaussian Discriminant Analysis
    """

    def __init__(self):
        self.phi = None
        self.mu0 = None
        self.mu1 = None
        self.sigma = None

    def train(self, x, y):
        """
        """
        self.phi = np.mean(y)
        self.mu0 = np.mean(x[y[:,0]==0], axis=0)
        self.mu1 = np.mean(x[y[:,0]==1], axis=0)

        n_x = x[y[:,0]==0] - self.mu0
        p_x = x[y[:,0]==1] - self.mu1

        self.sigma = ((n_x.T).dot(n_x) + (p_x.T).dot(p_x))/x.shape[0]

if __name__ == "__main__":
    trainX = data[:,:2]
    trainY = data[:,-1:]

    gda = GDA()
    gda.train(trainX, trainY)

    pos_data = data[data[:,-1]==1]
    neg_data = data[data[:,-1]==0]
    plt.scatter(pos_data[:,0], pos_data[:,1], color="red")
    plt.scatter(neg_data[:,0], neg_data[:,1], color="blue")

    phi = gda.phi
    sigma = gda.sigma
    inv_sigma = np.linalg.inv(sigma)
    mu0 = gda.mu0.reshape(-1, 1)
    mu1 = gda.mu1.reshape(-1, 1)

    """
    Draw GDA Decision Boundary
    """
    k = (2 * inv_sigma).dot(mu1-mu0)
    b = (mu1.T).dot(inv_sigma).dot(mu1) - (mu0.T).dot(inv_sigma).dot(mu0) + np.log(phi) - np.log(1-phi)

    x1 = np.arange(-1, 10, 1)
    x2 = (b[0][0] - k[0][0]*x1)/k[1][0]

    """
    Draw contour
    """
    xgrid1 = np.arange(-2, 4, 0.1)
    ygrid1 = np.arange(-1, 6, 0.1)
    xgrid0 = np.arange(4, 10, 0.1)
    ygrid0 = np.arange(5, 12, 0.1)
    X1, Y1 = np.meshgrid(xgrid1, ygrid1)
    X0, Y0 = np.meshgrid(xgrid0, ygrid0)

    Z1 = np.exp(-0.35*((X1-1)**2+0.85*(Y1-2)**2)) # Compute function values on the grid
    Z0 = np.exp(-0.35*((X0-7)**2+0.85*(Y0-8)**2))
    CS1 = plt.contour(X1,Y1,Z1)
    CS0 = plt.contour(X0,Y0,Z0)

    plt.plot(x1,x2)
    plt.show()

