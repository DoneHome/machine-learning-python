# coding:utf-8
import sys
import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt("./data/gaussian_discriminant_analysis.csv", dtype=np.float64, delimiter=",")
fig = plt.figure()

def plot_estimator(estimator, X, y):
    """
    这个函数的作用是基于分类器，对预测结果与原始标签进行可视化。
    """

    estimator.fit(X, y)
    # 确定网格最大最小值作为边界
    x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
    y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
    # 产生网格节点
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
            np.linspace(y_min, y_max, 100))
    # 基于分离器，对网格节点做预测
    Z = estimator.predict(np.c_[xx.ravel(), yy.ravel()])

    # 对预测结果上色
    Z = Z.reshape(xx.shape)
    pl.figure()
    pl.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # 同时对原始训练样本上色
    pl.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
    pl.axis('tight')
    pl.axis('off')
    pl.tight_layout()

class SVM():
    """
    Support Vector Machine
    """

    def __init__(self, C=1, kernel="rbf", num_iters=100):
        self.kernel = kernel # rbf or polynomial(for non-linear classifier) , linear (for linear classifier)
        self.alpha = None # Lagrange multiplier, greater or equal to zero
        self.C = C
        self.zeta = 0

        self.num_iters = num_iters

    def hypothesis_func(self, x):
        """ h = sign(w.T * x + bias)
        """
        pass

    def evaluate_kernel(self):
        if self.kernel == "rbf":
            pass
        elif self.kernel == "polynomial":
            pass
        elif self.kernel == "linear":
            pass
        else:
            raise Exception("%s don't exist" % opt)


    def train(self, X, Y, kernelOpt="rbf"):
        """
        """
        num_train = X.shape[0]
        self.alpha = np.zeros((num_train, 1))
        print self.alpha


if __name__ == "__main__":
    trainX = data[:,:2]
    trainY = data[:,-1:]

    clf = SVM(C=1, kernel="rbf", num_iters=100)
    clf.train(trainX, trainY)

    pos_data = data[data[:,-1]==1]
    neg_data = data[data[:,-1]==0]
    plt.scatter(pos_data[:,0], pos_data[:,1], color="red")
    plt.scatter(neg_data[:,0], neg_data[:,1], color="blue")

    # 绘制支持向量
    #pl.scatter(svc.support_vectors_[:, 0], svc.support_vectors_[:, 1], s=80, facecolors='none', zorder=10)
    #plt.show()



