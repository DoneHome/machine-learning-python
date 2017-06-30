# coding:utf-8

import sys
import numpy as np
from dataProvider import DataProvider

class Tree():

    def __init__(self):
        """
        """
        self.name = None
        self.val = None
        self.left_node = None
        self.right_node = None

class CART():
    """
    here only handle classification
    """

    def __init__(self, data, max_depth, min_samples_split, min_samples_leaf):
        """
        """
        self.provider = DataProvider(data)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_features_nums = int(1+np.sqrt(self.provider.num_features))
        self.tree = None

    def prune(self):
        pass

    def calcImpurity(self, dataList, D_size):
        """
        For classification:
            Gini(D, A) = |D1|/|D|*Gini(D1) + |D2|/|D|*Gini(D2)
        """
        gini_score = 0.0

        for data in dataList:
            _nums = data.shape[0] * 1.0
            if _nums == 0:
                continue
            factor_1 = _nums/D_size
            factor_2 = 0.0
            for label in self.provider.labels:
                prob = data[:, -1].tolist().count(label)/_nums
                factor_2 += prob * (1.0 - prob)
            gini_score += factor_1 * factor_2
        return gini_score

    def splitDataSet(self, dataSet):
        """
        """
        selectFeature = None
        selectFeatureVal = None
        selectLeftData = None
        selectRightData = None
        min_score = sys.maxint

        num_samples = dataSet.shape[0]

        # select partial features randomly
        selected_features = np.random.choice(self.provider.num_features, self.random_features_nums, replace = False)

        #for idx in range(self.provider.num_features):
        for idx in selected_features:
            leftData = None
            rightData = None
            for feature_val in self.provider.col_features[idx]:
                if self.provider.col_type[idx] == "numerical":
                    leftData = dataSet[np.nonzero(np.array(dataSet[:, idx], dtype=np.float64) <= float(feature_val))[0]]
                    rightData = dataSet[np.nonzero(np.array(dataSet[:, idx], dtype=np.float64) > float(feature_val))[0]]
                elif self.provider.col_type[idx] == "categorical":
                    leftData = dataSet[np.nonzero(dataSet[:, idx] == feature_val)[0]]
                    rightData = dataSet[np.nonzero(dataSet[:, idx] != feature_val)[0]]

                gini_score = self.calcImpurity([leftData, rightData], num_samples)
                if gini_score < min_score:
                    min_score = gini_score
                    selectFeature = idx
                    selectFeatureVal = feature_val
                    selectLeftData = leftData
                    selectRightData = rightData
        #print selectFeature, selectFeatureVal, min_score
        return selectFeature, selectFeatureVal, selectLeftData, selectRightData

    def getLeafNodelValue(self, data):
        """
        Get majority of labels
        """
        labels = data[:, -1].tolist()
        return max(labels, key=labels.count)

    def recursiveGrowth(self, tree, depth, leftData, rightData):
        """
        """

        if leftData.shape[0] == 0 or rightData.shape[0] == 0:
            #print "stop recursive growth: no split data"
            if leftData.shape[0] == 0:
                merge_data = rightData
            else:
                merge_data = leftData
            tree.left_node = self.getLeafNodelValue(merge_data)
            tree.right_node = self.getLeafNodelValue(merge_data)
            return

        if depth >= self.max_depth:
            #print "stop recursive growth:tree depth exceed max_depth:%s" %self.max_depth
            tree.left_node = self.getLeafNodelValue(leftData)
            tree.right_node = self.getLeafNodelValue(rightData)
            return

        if leftData.shape[0] < self.min_samples_split:
            #print "left tree stop recursive growth: the split data below min_samples_split:%s" %self.min_samples_split
            tree.left_node = self.getLeafNodelValue(leftData)
            return
        else:
            #print "left tree growth"
            featureName, featureValue, selectLeftData, selectRightData = self.splitDataSet(leftData)
            tree.left_node = Tree()
            tree.left_node.name = featureName
            tree.left_node.val = featureValue
            self.recursiveGrowth(tree.left_node, depth+1, selectLeftData, selectRightData)

        if rightData.shape[0] < self.min_samples_split:
            #print "right tree stop recursive growth: the split data below min_samples_split:%s" %self.min_samples_split
            tree.right_node = self.getLeafNodelValue(rightData)
            return
        else:
            #print "right tree growth"
            featureName, featureValue, selectLeftData, selectRightData = self.splitDataSet(rightData)
            tree.right_node= Tree()
            tree.right_node.name = featureName
            tree.right_node.val = featureValue
            self.recursiveGrowth(tree.right_node, depth+1, selectLeftData, selectRightData)

    def createTree(self, dataSet, depth):
        """
        """
        featureName, featureValue, selectLeftData, selectRightData = self.splitDataSet(dataSet)
        root_tree = Tree()
        root_tree.name = featureName
        root_tree.val = featureValue
        self.recursiveGrowth(root_tree, depth, selectLeftData, selectRightData)

        self.tree = root_tree
        return root_tree

    def predict(self, data):
        """
        """
        idx = self.tree.name
        col_type = self.provider.col_type[idx]

        if self.provider.col_type[idx] = "numerical":
            if data[idx] <= self.tree.val:
                if isinstance(self.left_node, Tree):
                    return self.predict(self.left_node, data)
                else:
                    return self.left_node
            else:
                if isinstance(self.right_node, Tree):
                    return self.predict(self.right_node, data)
                else:
                    return self.left_node
        elif self.provider.col_type[idx] = "categorical":
            if data[idx] == self.tree.val:
                if isinstance(self.left_node, Tree):
                    return self.predict(self.left_node, data)
                else:
                    return self.left_node
            else:
                if isinstance(self.right_node, Tree):
                    return self.predict(self.right_node, data)
                else:
                    return self.left_node

if __name__ == "__main__":
    source_data = np.genfromtxt("./gbdt_data.csv", dtype=str, delimiter=",")
    tree = CART(source_data)

