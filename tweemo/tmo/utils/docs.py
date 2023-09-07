import re
import numpy as np


def read_docs(file_name):
    print('read documents')
    print('-' * 50)
    docs = []
    fp = open(file_name, 'r')
    for line in fp:
        arr = re.split('\s', line[:-1])
        arr = filter(None, arr)
        arr = [int(idx) for idx in arr]
        docs.append(arr)
    fp.close()

    return docs


def read_vocab(file_name):
    print('read vocabulary')
    print('-' * 50)
    vocab = []
    fp = open(file_name, 'r')
    for line in fp:
        arr = re.split('\s', line[:-1])
        vocab.append(arr[0])
    fp.close()

    return vocab


def co_occur(A: np.ndarray) -> np.ndarray:
    """
    :param A: term document matrix
    :return: co occurences matrix
    """
    n_terms = A.shape[0]
    SS_mat = np.zeros([n_terms, n_terms])
    for index_doc_j in range(A.shape[1]):
        doc_j = A[:, index_doc_j].reshape(-1)
        set_doc_j = list(np.where(doc_j > 0))[0]
        for i in set_doc_j:
            for j in set_doc_j:
                if i != j:
                    SS_mat[i, j] += 1
    return SS_mat


def calculate_PMI(
        AA: np.ndarray,
        topKeywordsIndex
):
    """
    Reference:
    Short and Sparse Text Topic Modeling via Self-Aggregation
    """
    D1 = np.sum(AA)
    n_tp = len(topKeywordsIndex)
    PMI = []
    for index1 in topKeywordsIndex:
        for index2 in topKeywordsIndex:
            if index2 < index1:
                if AA[index1, index2] == 0:
                    PMI.append(0.0)
                else:
                    C1 = np.sum(AA[index1])
                    C2 = np.sum(AA[index2])
                    PMI.append(np.log(AA[index1, index2] * D1 / C1 / C2))
    avg_PMI = 2.0 * np.sum(PMI) / float(n_tp) / (float(n_tp) - 1.0)

    return avg_PMI
