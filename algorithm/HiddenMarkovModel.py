# coding: utf-8

import numpy as np

class HMM(object):
    """
    Inputs have status dimension N, and there are M observations

    Inputs:
        A: A numpy array of shape (N, N) containing state transition probability
        B: A numpy array of shape (N, M) containing output probability
       Pi: A numpy array of shape (1, N) containing initial probability
      Obs: A numpy array of shape (1, N) containing observation sequence
    """

    def __init__(self, Ann, Bnm, Pn, On):
        self.A = Ann
        self.B = Bnm
        self.Pi = Pn
        self.Obs = On

        self.N = self.A.shape[0]
        self.M = self.B.shape[1]

    def forward(self):
        """
        Args:
        """
        T = self.Obs.shape[0]
        alpha = np.zeros((T, self.N))
        alpha[0, :] = self.Pi * self.B[:, self.Obs[0]]

        for t in xrange(1, T):
            for i in xrange(self.N):
                alpha[t][i] = np.sum(alpha[t-1, :] * self.A[:, i]) * self.B[i][self.Obs[t]]

        prob = np.sum(alpha[-1, :])
        return prob

    def viterbi(self):
        """
        Args:
        """
        T = self.Obs.shape[0]
        I = np.zeros(T)
        delta = np.zeros((T, self.N))
        psi = np.zeros((T, self.N))
        delta[0, :] = self.Pi * self.B[:, self.Obs[0]]

        for t in xrange(1, T):
            for i in xrange(self.N):
                delta[t][i] = np.max(delta[t-1, :] * self.A[:, i]) * self.B[i][self.Obs[t]]
                psi[t][i] = np.argmax(delta[t-1, :] * self.A[:, i]) + 1

        I[T-1] = np.argmax(delta[-1, :]) + 1

        for t in range(T-2, -1, -1):
            I[t] = psi[t+1][I[t+1]-1]

        return I


if __name__ == "__main__":
    Ann = np.array([[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]])
    Bnm = np.array([[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]])
    Pn = np.array([0.2, 0.4, 0.4])
    Obs = np.array([0, 1, 0])

    h = HMM(Ann, Bnm, Pn, Obs)
    #h.forward()
    h.viterbi()
