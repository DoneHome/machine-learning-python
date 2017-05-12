# coding: utf-8

import sys
import numpy as np
import matplotlib.pyplot as plt

"""
线性回归
"""

data = np.genfromtxt("./data/liner_regression_data.csv", dtype=np.float64, delimiter=",")
plt.figure(1)
plt.scatter(data[:,0],data[:,1])

#plt.show(block=False)

def hypothesis_func(w, x):
    return w[0]*x[0]+w[1]*x[1]

def evaluate_gradient(w, x, y):
    return (hypothesis_func(w, x) - y) * x

def BGD(x, y, w, l_r=0.0001, num_iters=100):
    """
    Batch Gradient Descent
    x: 输入值
    y: 目标值
    w: 权值
    l_r: 学习速率
    num_iters: 迭代次数
    """
    num_train = x.shape[0]

    for i in range(num_iters):
        """
        notice: 
           num_train一定要在grad这一行进行相除, 如果在w更新时再除以的话，会无法收敛到最优解, 直观表现, 梯度不趋向0
        """
        grad = np.sum((x.dot(w).reshape(-1, 1) - y) * x, axis = 0)/num_train
        w -= l_r * grad
    return w

def SGD(x, y, w, l_r=0.0001, num_iters=20):
    """
    Stochastic Gradient Descent
    如何解决err数值溢出, 加log? 其实我怀疑是l_r选取过大，导致发散
    """
    num_train = x.shape[0]
    err = sys.maxint
    _iter = 0
    eps = 1e-1

    while err > eps and _iter < num_iters:
        _iter += 1

        mask = np.random.choice(num_train, 1, replace=False)
        x_train = x[mask][0] #这里只有一个样本, 所以直接取出来了
        y_train = y[mask][0]

        grad = evaluate_gradient(w, x_train, y_train)
        w -= l_r * grad

        err = np.sum(np.square(y - x.dot(w).reshape(-1, 1)))
    return w

def MBGD(x, y, w, l_r = 0.00001, batch=10, num_iters=10000):
    """
    Mini-batch Gradient Descent
    """
    num_train = x.shape[0]
    err = sys.maxint
    _iter = 0
    eps = 1e-1

    while err > eps and _iter < num_iters:
        _iter += 1
        mask = np.random.choice(num_train, batch, replace=False)
        x_train = x[mask]
        y_train = y[mask]

        grad_mat = np.sum((x_train.dot(w.T).reshape(-1, 1) - y_train) * x_train, axis = 0)
        w -= l_r * grad_mat

        err = np.sum(np.square(y - x.dot(w).reshape(-1, 1)))
    return w


if __name__ == "__main__":
    x = data[:, 0].reshape(-1, 1)
    y = data[:, 1].reshape(-1, 1)

    input_x = np.hstack([np.ones((x.shape[0], 1)), x])
    input_y = y

    W = np.random.randn(input_x.shape[1]) * 0.01

    #W = BGD(input_x, input_y, W)
    W = SGD(input_x, input_y, W)
    #W = MBGD(input_x, input_y, W)

    x = np.linspace(0, 100, 10)
    plt.plot(x, W[0]+W[1]*x, color="red")
    plt.show()


