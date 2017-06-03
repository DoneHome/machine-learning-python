# coding:utf-8
import sys
import numpy as np


class SMO():
    """
    """

    def __init__(self, alpha, kernel_matrix, C, num_iters, eps):
        """
        """
        self.alpha = alpha
        self.kernel_matrix = kernel_matrix
        self.C = C
        self.eps = eps

        self.num_iters = num_iters

    def select_pairs_alpha(self):
        """
        select alpha_i, alpha_j from vector alpha
        """
        pass

    def optimize(self):
        """
        _iter: the iteration numbers that no any alpha changed
        """

        _iter = 0
        while _iter < self.num_iters:
            # update alpha through support vectors that 0 < alpha_i < C
            nonBoundidx = np.nonzero((self.alpha > 0) * (self.alpha < self.C))[0]
            pass

            # update alpha through all training samples
            pass


