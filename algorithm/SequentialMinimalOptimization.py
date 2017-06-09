# coding:utf-8

import sys
import numpy as np

class SMO():
    """
    Reference:
        《统计学习方法》
        《Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines》
         http://www.wengweitao.com/zhi-chi-xiang-liang-ji-smoxu-lie-zui-xiao-zui-you-hua-suan-fa.html
         http://www.cnblogs.com/jerrylead/archive/2011/03/18/1988419.html
    """

    def __init__(self, alpha, bias, kernel_matrix, C, num_train, num_iters, tol, Y):
        """
        """
        self.alpha = alpha
        self.bias = bias
        self.kernel_matrix = kernel_matrix
        self.C = C
        self.tol = tol
        self.E_val = np.zeros((num_train))
        self.num_train = num_train
        self.num_iters = num_iters
        self.Y = Y

    def calc_decision_error(self, i):
        """
        error = g(xi) - yi
        """
        g_xi= ((self.alpha * self.Y).T).dot(self.kernel_matrix[:,i]) + self.bias
        error = float(g_xi - self.Y[i])
        return error

    def is_violate_KKT(self, i, E_i):
        """
        1) alpha == 0, yi * g(xi) >= 1
        2) 0 < alpha < C, yi * g(xi) == 1
        3) alpha = C, yi * g(xi) <= 1
        """
        if (self.Y[i] * E_i < -self.tol and self.alpha[i] < self.C):
            return True
        elif (self.Y[i] * E_i > self.tol and self.alpha[i] > 0):
            return True
        else:
            return False

    def select_alpha_J(self, i):
        """
        """
        E_i = self.E_val[i]

        max_diff = 0
        j = -1
        E_j = 0

        for k in range(self.num_train):
            if k == i:
                continue
            E_k = self.calc_decision_error(k)
            diff = abs(E_i - E_k)
            if diff > max_diff:
                max_diff = diff
                j = k
                E_j = E_k

        self.E_val[j] = E_k
        return j

    def select_pair_alphas(self, candidateIdx):
        """
        1) choose alpha_i that against kkt
        2) choose alpha_j that maxmize |Ei-Ej|
        """
        for i in candidateIdx:
            if not self.is_satisfy_KKT(i):
                j = self.select_alpha_J(i)
                return [i, j]
        return None

    def calc_alpha_bound(self, i, j):
        """
        """
        alpha_i = self.alpha[i]
        alpha_j = self.alpha[j]

        if self.Y[i] == self.Y[j]:
            L = max(0, alpha_j + alpha_i - self.C)
            H = min(self.C, alpha_j + alpha_i)
        else:
            L = max(0, alpha_j - alpha_i)
            H = min(self.C, self.C + alpha_j - alpha_i)
        return L, H

    def calc_eta(self, i, j):
        """
        """
        eta = self.kernel_matrix[i][i] + self.kernel_matrix[j][j] - 2*self.kernel_matrix[i][j]
        return eta

    def update_alpha_i(self, i, j, alpha_j_new):
        """
        """
        alpha_i_new = self.alpha[i] + self.Y[i]*self.Y[j]*(self.alpha[j] - alpha_j_new)
        return alpha_i_new

    def update_alpha_j(self, i, j , eta):
        """
        """
        alpha_j_new = self.alpha[j] + self.Y[j]*(self.E_val[i] - self.E_val[j])*1.0/eta
        return alpha_j_new

    def update_E_val(self, i, j):
        """
        """
        E_i = self.calc_decision_error(i)
        E_j = self.calc_decision_error(j)

        self.E_val[i] = E_i
        self.E_val[j] = E_j
        return

    def clip_alpha(self, alpha, L, H):
        """
        """
        if alpha > H:
            alpha = H
        elif alpha < L:
            alpha = L
        return alpha

    def update_bias(self, i, j, alpha_i_new, alpha_j_new):
        """
        """
        b1_new = -self.E_val[i] - self.Y[i]*self.kernel_matrix[i][i]*(alpha_i_new - self.alpha[i]) - self.Y[j]*self.kernel_matrix[j][i]*(alpha_j_new - self.alpha[j]) + self.bias
        b2_new = -self.E_val[j] - self.Y[i]*self.kernel_matrix[i][j]*(alpha_i_new - self.alpha[i]) - self.Y[j]*self.kernel_matrix[j][j]*(alpha_j_new - self.alpha[j]) + self.bias

        if alpha_i_new > 0 or alpha_i_new < self.C:
            b_new = b1_new
        elif alpha_j_new > 0 or alpha_j_new < self.C:
            b_new = b2_new
        else:
            b_new = (b1_new + b2_new)*1.0/2
        return b_new

    def takeStep(self, i, j):
        """
        """
        if i == j:
            return 0

        # step 1: calculate L and H
        L, H = self.calc_alpha_bound(i, j)
        if L == H:
            return 0
        # step 2: calculate eta
        eta = self.calc_eta(i, j)
        if eta <= 0:
            return 0
        # step 3: update alpha_j
        alpha_j_new = self.update_alpha_j(i, j, eta)
        # step 4: clip alpha_j
        alpha_j_new = self.clip_alpha(alpha_j_new, L, H)
        # step 5: check the effectiveness of alpha_j
        if abs(alpha_j_new - self.alpha[j]) < 0.0001:
            return 0
        # step 6: update alpha_i
        alpha_i_new = self.update_alpha_i(i, j, alpha_j_new)
        # step 7: update bias
        self.bias = self.update_bias(i, j, alpha_i_new, alpha_j_new)
        # step 8: store new alpha_i, alpha_j
        self.alpha[i] = alpha_i_new
        self.alpha[j] = alpha_j_new
        # step 9: update E_val
        self.update_E_val(i, j)

        return 1

    def examineExample(self, i):
        """
        """
        E_i = self.calc_decision_error(i)
        self.E_val[i] = E_i
        #print "%s\t%s" %(i, E_i)

        if self.is_violate_KKT(i, E_i):
            #print "viloate KKT:%s" %i
            nonBoundJdx = np.nonzero((self.alpha > 0) * (self.alpha < self.C))[0]
            #
            if len(nonBoundJdx) > 1:
                #print "enter loop 1"
                j = self.select_alpha_J(i)
                if self.takeStep(i, j):
                    return 1

            # loop over all non-zero and non-C alpha, starting at a random point
            for j in nonBoundJdx:
                #print "enter loop 2"
                if self.takeStep(i, j):
                    return 1

            # loop over all possible i1, starting at a random point
            for j in range(self.num_train):
                #print "enter loop 3"
                if self.takeStep(i, j):
                    return 1
        return 0

    def optimize(self):
        """
        """
        _iter = 0

        examineAll = True
        numChanged = 0

        while _iter < self.num_iters and (numChanged > 0 or examineAll):
            numChanged = 0

            if examineAll:
                # update alpha through all training samples
                for i in range(self.num_train):
                    numChanged += self.examineExample(i)
                #print "iter:%s\texamineAll\tnumChanged:%s" %(_iter, numChanged)
            else:
                # update alpha through samples that 0 < alpha_i < C
                nonBoundIdx = np.nonzero((self.alpha > 0) * (self.alpha < self.C))[0]
                for i in nonBoundIdx:
                    numChanged += self.examineExample(i)
                #print "iter:%s\tnumChanged\tnumChanged:%s" %(_iter, numChanged)

            if examineAll:
                examineAll = False
            elif numChanged == 0:
                examineAll = True

            _iter += 1
        return self.alpha, self.bias

