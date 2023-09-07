"""
Topic Modeling via NMF
"""

import time
import numpy as np
from numpy.linalg import norm
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score  # nmi & ari
from tweemo.tmo.utils import Jar, calculate_PMI, co_occur


class NMF(object):
    def __init__(
            self,
            A, IW=[], IH=[],
            n_topic=10, max_iter=100, max_err=1e-3,
            rand_init=True):
        """
        A = WH^T
        """
        self.H = None
        self.W = None
        self.component_ = None

        self.A = A
        self.n_row = A.shape[0]
        self.n_col = A.shape[1]

        self.n_topic = n_topic
        self.max_iter = max_iter
        self.max_err = max_err
        self.rand_init = rand_init

        self.IW, self.IH = IW, IH
        self.obj = []

    def _fit_transform(self):
        if self.rand_init:
            self.nmf_init_rand()
        else:
            self.nmf_init(self.IW, self.IH)
        self.nmf_iter()

        return self.W, self.H

    def fit_transform(self, all_decomposition: bool = False):
        W, H = self._fit_transform()
        self.component_ = H
        return Jar(W=W) if all_decomposition == False else Jar(W=W, H=H)

    def transform(self):
        W, _ = self._fit_transform()
        return Jar(W=W)

    def nmf_init_rand(self):
        self.W = np.random.random((self.n_row, self.n_topic))
        self.H = np.random.random((self.n_col, self.n_topic))

        for k in range(self.n_topic):
            self.W[:, k] /= norm(self.W[:, k])

    def nmf_init(self, IW, IH):
        self.W = IW
        self.H = IH

        for k in range(self.n_topic):
            self.W[:, k] /= norm(self.W[:, k])

    def nmf_iter(self):
        loss_old = 1e20
        print('loop begin')
        start_time = time.time()
        for i in range(self.max_iter):
            self.nmf_solver()
            loss = self.nmf_loss()
            self.obj.append(loss)

            if loss_old - loss < self.max_err:
                break
            loss_old = loss
            end_time = time.time()
            print('Step={}, Loss={}, Time={}s'.format(i, loss, end_time - start_time))
        print('loop end')

    def nmf_solver(self):
        """
        regular NMF without constraint.
        Block Coordinate Decent
        """
        epss = 1e-20

        HtH = self.H.T.dot(self.H)
        AH = self.A.dot(self.H)
        for k in range(self.n_topic):
            tmpW = self.W[:, k] * HtH[k, k] + AH[:, k] - np.dot(self.W, HtH[:, k])
            self.W[:, k] = np.maximum(tmpW, epss)
            self.W[:, k] /= norm(self.W[:, k]) + epss

        WtW = self.W.T.dot(self.W)
        AtW = self.A.T.dot(self.W)
        for k in range(self.n_topic):
            self.H[:, k] = self.H[:, k] * WtW[k, k] + AtW[:, k] - np.dot(self.H, WtW[:, k])
            self.H[:, k] = np.maximum(self.H[:, k], epss)

    def nmf_loss(self):
        loss = norm(self.A - np.dot(self.W, np.transpose(self.H)), 'fro') ** 2 / 2.0
        return loss

    def get_topics(self) -> np.ndarray:
        return np.argmax(self.component_, axis=1).reshape(-1)

    def get_loss(self):
        return np.array(self.obj)

    def get_lowrank_matrix(self):
        return self.W, self.H

    def get_nmi(self, labels_true) -> float:
        return normalized_mutual_info_score(labels_true, self.get_topics())

    def get_ari(self, labels_true) -> float:
        return adjusted_rand_score(labels_true, self.get_topics())

    def get_pmi_scores(self, n_topKeyword: int) -> Jar:
        PMI_arr = []
        # calculate co-occur matrix
        S = co_occur(self.A)
        for k in range(self.n_topic):
            topKeywordsIndex = self.W[:, k].argsort()[::-1][:n_topKeyword]
            PMI_arr.append(calculate_PMI(S, topKeywordsIndex))
        index = np.argsort(PMI_arr)
        weights_by_topics = {k: [] for k in index}
        for k in index:
            weights_topic_k = np.argsort(self.W[:, k])[::-1][:n_topKeyword]
            for i_w in weights_topic_k:
                weights_by_topics[k].append((i_w, self.W[i_w, k]))
        return Jar(PMI_arr=PMI_arr, weights_by_topics=weights_by_topics)

    def save_format(self, Wfile='W.txt', Hfile='H.txt'):
        np.savetxt(Wfile, self.W)
        np.savetxt(Hfile, self.H)
