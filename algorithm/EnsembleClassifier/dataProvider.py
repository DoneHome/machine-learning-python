#coding: utf-8

import sys
import numpy as np

class DataProvider():

    def __init__(self, source_data):
        #self.data = source_data
        self.num_samples = source_data.shape[0]
        self.num_features = source_data.shape[1]-1

        self.col_type = {} #
        self.col_features = {} #
        self.labels = set()
        self.get_col_content(source_data)

        #self.validation_data = None
        #self.train_data = None
        #self._SPLIT_NAME = {"train": 0.8, "validation":0.2}
        #self.split_data(source_data)

    def get_col_content(self, source_data):
        """
        """
        for i in range(self.num_features):
            try:
                _ = float(source_data[:, i][0])
                self.col_type[i] = "numerical"
            except ValueError, e:
                self.col_type[i] = "categorical"

            self.col_features[i] = set(source_data[:, i])
            self.labels = set(source_data[:, -1])

    def splitData(self, source_data):
        """
        Don't use in RF
        """
        num_samples, num_feature = source_data.shape
        np.random.shuffle(source_data)

        num_training = int(num_samples * self._SPLIT_NAME['train'])
        num_validation = num_samples - num_training

        mask = range(num_training)
        self.train_data = source_data[mask]

        mask = range(num_training, num_samples)
        self.validation_data = source_data[mask]

