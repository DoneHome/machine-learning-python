# coding:utf-8
import sys
import numpy as np

from dataProvider import DataProvider
from tree import CART

class RF():
    """
    Random Forest
    """

    def __init__(self, n_estimators=100, max_depth=10, min_samples_split=2, min_samples_leaf=1, subsample=0.7, criterion='friedman_mse'):
        """
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = 0.7
        self.criterion = criterion
        #self.min_impurity_split = 1e-7

        self.trees = []
        self.provider = None

    def bootstrap(self, data):
        """
        # 随机森林里，除了随机选取样本外，还可以随机选取feature，这样才能保证每棵子树之间具有多样性
        #TODO: 随机性测试
        """
        num_samples = data.shape[0]
        num_subs = int(num_samples * self.subsample)

        mask = np.random.choice(num_samples, num_subs, replace = True)
        return data[mask]

    def train(self, data):
        """
        """
        #self.provider = DataProvider(data, self.subsample)

        for i in range(self.n_estimators):
            sub_samples = self.bootstrap(data)
            cart = CART(sub_samples, self.max_depth, self.min_samples_split, self.min_samples_leaf)
            tree = cart.createTree(sub_samples, depth=0)
            print tree.show()
            #self.trees.append(tree)
            return

    def predict(self, data):
        """
        """
        pass

if __name__ == "__main__":
    source_data = np.genfromtxt("./gbdt_data.csv", dtype=str, delimiter=",")

    rf = RF(n_estimators=100, max_depth=10, min_samples_split=2)
    rf.train(source_data)
