from tweemo.tmo.utils import read_docs, read_vocab
import numpy as np


class Corpus(object):
    def __int__(
            self,
            corpus_file,
            vocab_file,
    ):
        self.docs = read_docs(corpus_file)
        self.vocab = read_vocab(vocab_file)
        self.n_docs = len(self.docs)
        self.n_terms = len(self.vocab)
        print('n_docs={}, n_terms={}'.format(self.n_docs, self.n_terms))

    def _read_term_doc_matrix(self):
        print('read term doc matrix')
        self.dt_mat = np.zeros([self.n_terms, self.n_docs])
        for k in range(self.n_docs):
            for j in self.docs[k]:
                self.dt_mat[j, k] += 1.0
        print('term doc matrix done')
        print('-' * 50)

    def _co_occur_matrix(self):
        print('calculate co-occurance matrix')
        self.co_occur_mat = np.zeros([self.n_terms, self.n_terms])
        for itm in self.docs:
            for kk in itm:
                for jj in itm:
                    self.co_occur_mat[int(kk), int(jj)] += 1.0
        print('co-occur done')
        print('-' * 50)

    def ppmi(self):
        print('calculate PPMI')
        D1 = np.sum(self.dt_mat)
        SS = D1 * self.dt_mat
        for k in range(self.n_terms):
            SS[k] /= np.sum(self.dt_mat[k])
        for k in range(self.n_terms):
            SS[:, k] /= np.sum(self.dt_mat[:, k])
        self.dt_mat = []  # release memory
        SS[SS == 0] = 1.0
        SS = np.log(SS)
        SS[SS < 0.0] = 0.0
        print('PPMI done')
        print('-' * 50)
