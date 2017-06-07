# coding:utf-8

import sys
import numpy as np

class SMO():
    """
    """

    def __init__(self, alpha, bias, kernel_matrix, C, num_train, num_iters, eps, Y):
        """
        """
        self.alpha = alpha
        self.bias = bias
        self.kernel_matrix = kernel_matrix
        self.C = C
        self.eps = eps
        self.E_val = []
        self.num_train = num_train
        self.num_iters = num_iters
        self.Y = Y

    def evaluate_decision_error(self):
        """
        Ei = g(xi) - yi
        """
        for i in range(self.num_train):
            error = 0.0
            for j in range(self.num_train):
                error += self.alpha[j] * self.Y[j] * self.kernel_matrix[j][i]
            error = error + self.bias - self.Y[i]
            self.E_val.append(error)

    def is_satisfy_KKT(self, idx):
        """
        1) alpha == 0, yi * g(xi) >= 1
        2) 0 < alpha < C, yi * g(xi) == 1
        3) alpha = C, yi * g(xi) <= 1
        """
        if (self.Y[idx] * self.E_val[idx] < -self.eps and self.alpha[idx] < self.C):
            return False
        elif (self.Y[idx] * self.E_val[idx] > self.eps and self.alpha[idx] > 0):
            return False
        else:
            return True

    def select_alpha_J(self, i):
        """
        """
        Ei = self.E_val[i]
        if Ei >=0:
            Ej = min(self.E_val)
        else:
            Ej = max(self.E_val)

        j = self.E_val.index(Ej)
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

    def calc_alpha_bound(self, alphas_list):
        """
        """
        i, j = alphas_list
        alpha_i = self.alpha[i]
        alpha_j = self.alpha[j]

        if self.Y[i] == self.Y[j]:
            L = max(0, alpha_j + alpha_i - self.C)
            H = min(self.C, alpha_j + alpha_i)
        else:
            L = max(0, alpha_j - alpha_i)
            H = min(self.C, self.C + alpha_j - alpha_i)
        return L, H

    def calc_eta(self, alphas_list):
        """
        """
        i, j = alphas_list
        eta = self.kernel_matrix[i][i] + self.kernel_matrix[j][j] - 2*self.kernel_matrix[i][j]
        return eta

    def update_alpha_i(self, alphas_list, alpha_j_new):
        """
        """
        i, j = alphas_list
        alpha_i_new = self.alpha[i] + self.Y[i]*self.Y[j]*(self.alpha[j] - alpha_j_new)
        return alpha_i_new

    def update_alpha_j(self, alphas_list, eta):
        """
        """
        i, j = alphas_list
        alpha_j_new = self.alpha[j] + self.Y[j]*(self.E_val[i] - self.E_val[j])*1.0/eta
        return alpha_j_new

    def clip_alpha(self, alpha, L, H):
        """
        """
        if alpha > H:
            alpha = H
        elif alpha < L:
            alpha = L
        return alpha

    def update_bias(self, alphas_list, alpha_i_new, alpha_j_new):
        """
        """
        i, j = alphas_list

        b1_new = -self.E_val[i] - self.Y[i]*self.kernel_matrix[i][i]*(alpha_i_new - self.alpha[i]) - self.Y[j]*self.kernel_matrix[j][i]*(alpha_j_new - self.alpha[j]) + self.bias
        b2_new = -self.E_val[j] - self.Y[i]*self.kernel_matrix[i][j]*(alpha_i_new - self.alpha[i]) - self.Y[j]*self.kernel_matrix[j][j]*(alpha_j_new - self.alpha[j]) + self.bias

        if alpha_i_new > 0 or alpha_i_new < self.C:
            b_new = b1_new
        elif alpha_j_new > 0 or alpha_j_new < self.C:
            b_new = b2_new
        else:
            b_new = (b1_new + b2_new)*1.0/2
        return b_new

    def solve(self, candidateIdx):
        """
        step 1: select alpha_i and alpha_j
        step 2: calculate L and H
        step 3: calculate eta
        step 4: update alpha_j
        step 5: clip alpha_j
        step 6: update alpha_i
        step 7: check the effectiveness of alpha_j
        step 8: update bias
        step 9: update E_val
        ...
        """
        # step 1
        alphas_list = self.select_pair_alphas(candidateIdx)
        if alphas_list:
            pass
        # step 2
        L, H = self.calc_alpha_bound(alphas_list)
        # step 3
        eta = self.calc_eta(alphas_list)
        # step 4
        alpha_j_new = self.update_alpha_j(alphas_list, eta)
        # step 5
        alpha_j_clip = self.clip_alpha(alpha_j_new, L, H)
        # step 6
        alpha_i_new = self.update_alpha_i(alphas_list, alpha_j_new)
        # step 7
        pass
        # step 8
        bias_new = self.update_bias(alphas_list, alpha_i_new, alpha_j_new)
        # step 9
        pass

    def optimize(self):
        """
        """
        _iter = 0
        self.evaluate_decision_error()

        entireSet = True

        while _iter < self.num_iters:
            if entireSet:
                # update alpha through all training samples
                candidateIdx = np.array(range(self.num_train))
                flag = self.solve(candidateIdx)
            else:
                # update alpha through samples that 0 < alpha_i < C
                candidateIdx = np.nonzero((self.alpha > 0) * (self.alpha < self.C))[0]
                flag = self.solve(candidateIdx)

            if entireSet:
                entireSet = False
            elif flag == False:
                entireSet = True

            _iter += 1
            return

