"""
Short Text Topic Modeling via SeaNMF
"""
import time
import numpy as np
from numpy.linalg import norm
from tweemo.tmo.utils import calculate_PMI, co_occur, Jar
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score  # nmi & ari


class SeaNMFL1(object):
    def __init__(
            self,
            A, S,
            IW1=[], IW2=[], IH=[],
            alpha=1.0, beta=0.1, n_topic=10, max_iter=100, max_err=1e-3,
            rand_init=True, fix_seed=False):
        """
        0.5*||A-WH^T||_F^2+0.5*alpha*||S-WW_c^T||_F^2+0.5*beta*||W||_1^2
        W = W1
        Wc = W2
        """
        self.component_ = None
        if fix_seed:
            np.random.seed(0)

        self.A = A
        self.S = S
        self.IW1, self.IW2, self.IH = IW1, IW2, IH

        self.n_row = A.shape[0]
        self.n_col = A.shape[1]

        self.n_topic = n_topic
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        self.B = np.ones([self.n_topic, 1])
        self.max_err = max_err

        self.rand_init = rand_init

    def _fit_transform(self):
        if self.rand_init:
            self.nmf_init_rand()
        else:
            self.nmf_init(self.IW1, self.IW2, self.IH)
        self.nmf_iter()

        return self.W1, self.W2, self.H

    def fit_transform(self, all_decomposition: bool = False):
        W, Wc, H = self._fit_transform()
        self.component_ = H
        return Jar(W=W, Wc=Wc) if all_decomposition is False else Jar(W=W, Wc=Wc, H=H)

    def transform(self):
        W, Wc, _ = self._fit_transform()
        return W, Wc

    def nmf_init_rand(self):
        self.W1 = np.random.random((self.n_row, self.n_topic))
        self.W2 = np.random.random((self.n_row, self.n_topic))
        self.H = np.random.random((self.n_col, self.n_topic))

        for k in range(self.n_topic):
            self.W1[:, k] /= norm(self.W1[:, k])
            self.W2[:, k] /= norm(self.W2[:, k])

    def nmf_init(self, IW1, IW2, IH):
        self.W1 = IW1
        self.W2 = IW2
        self.H = IH

        for k in range(self.n_topic):
            self.W1[:, k] /= norm(self.W1[:, k])
            self.W2[:, k] /= norm(self.W2[:, k])

    def nmf_iter(self):
        loss_old = 1e20
        print('loop begin')
        start_time = time.time()
        for i in range(self.max_iter):
            self.nmf_solver()
            loss = self.nmf_loss()
            if loss_old - loss < self.max_err:
                break
            loss_old = loss
            end_time = time.time()
            print('Step={}, Loss={}, Time={}s'.format(i, loss, end_time - start_time))

    def nmf_solver(self):
        """
        using BCD framework
        """
        epss = 1e-20
        # Update W1
        AH = np.dot(self.A, self.H)
        SW2 = np.dot(self.S, self.W2)
        HtH = np.dot(self.H.T, self.H)
        W2tW2 = np.dot(self.W2.T, self.W2)
        W11 = self.W1.dot(self.B)

        for k in range(self.n_topic):
            num0 = HtH[k, k] * self.W1[:, k] + self.alpha * W2tW2[k, k] * self.W1[:, k]
            num1 = AH[:, k] + self.alpha * SW2[:, k]
            num2 = np.dot(self.W1, HtH[:, k]) + self.alpha * np.dot(self.W1, W2tW2[:, k]) + self.beta * W11[0]
            self.W1[:, k] = num0 + num1 - num2
            self.W1[:, k] = np.maximum(self.W1[:, k], epss)  # project > 0
            self.W1[:, k] /= norm(self.W1[:, k]) + epss  # normalize
        # Update W2
        W1tW1 = self.W1.T.dot(self.W1)
        StW1 = np.dot(self.S, self.W1)
        for k in range(self.n_topic):
            self.W2[:, k] = self.W2[:, k] + StW1[:, k] - np.dot(self.W2, W1tW1[:, k])
            self.W2[:, k] = np.maximum(self.W2[:, k], epss)
        # Update H
        AtW1 = np.dot(self.A.T, self.W1)
        for k in range(self.n_topic):
            self.H[:, k] = self.H[:, k] + AtW1[:, k] - np.dot(self.H, W1tW1[:, k])
            self.H[:, k] = np.maximum(self.H[:, k], epss)

    def get_topics(self) -> np.ndarray:
        return np.argmax(self.component_, axis=1).reshape(-1)

    def nmf_loss(self):
        """
        Calculate loss
        """
        loss = norm(self.A - np.dot(self.W1, np.transpose(self.H)), 'fro') ** 2 / 2.0
        if self.alpha > 0:
            loss += self.alpha * norm(np.dot(self.W1, np.transpose(self.W2)) - self.S, 'fro') ** 2 / 2.0
        if self.beta > 0:
            loss += self.beta * norm(self.W1, 1) ** 2 / 2.0

        return loss

    def get_lowrank_matrix(self):
        return self.W1, self.W2, self.H

    def get_nmi(self, labels_true) -> float:
        return normalized_mutual_info_score(labels_true, self.get_topics())

    def get_ari(self, labels_true) -> float:
        return adjusted_rand_score(labels_true, self.get_topics())

    def get_pmi_scores(self, n_topKeyword: int) -> Jar:
        PMI_arr = []
        # calculate co-occur matrix
        S = co_occur(self.A)
        for k in range(self.n_topic):
            topKeywordsIndex = self.W1[:, k].argsort()[::-1][:n_topKeyword]
            PMI_arr.append(calculate_PMI(S, topKeywordsIndex))
        index = np.argsort(PMI_arr)
        weights_by_topics = {k: [] for k in index}
        for k in index:
            weights_topic_k = np.argsort(self.W1[:, k])[::-1][:n_topKeyword]
            for i_w in weights_topic_k:
                weights_by_topics[k].append((i_w, self.W1[i_w, k]))
        return Jar(PMI_arr=PMI_arr, weights_by_topics=weights_by_topics)

    def save_format(self, W1file='W.txt', W2file='Wc.txt', Hfile='H.txt'):
        np.savetxt(W1file, self.W1)
        np.savetxt(W2file, self.W2)
        np.savetxt(Hfile, self.H)
