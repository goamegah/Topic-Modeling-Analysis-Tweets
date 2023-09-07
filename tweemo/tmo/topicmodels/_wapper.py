from tweemo.tmo.utils import Jar
import numpy as np
from tweemo.tmo.topicmodels import NMF
from tweemo.tmo.topicmodels import SeaNMFL1
from nltk.lm.vocabulary import Vocabulary


class WrapperModel:
    def __init__(self,
                 model_name: str,
                 mapping: dict[str, int],
                 requirements: dict[str, object]):
        self.model = globals()[model_name](**requirements)  # load model
        self.model_name = model_name
        self.mapping_words = mapping

    def fit_transform(self,
                      all_decomposition: bool = False) -> Jar:
        return self.model.fit_transform(all_decomposition=all_decomposition)

    def get_topics(self) -> np.ndarray:
        return self.model.get_topics()

    def get_pmi_scores(self,
                       n_topKeyword: int) -> Jar:
        jar = self.model.get_pmi_scores(n_topKeyword)
        PMI_arr, weights_by_topics = jar.PMI_arr, jar.weights_by_topics
        mapping_inv = {value: key for key, value in zip(self.mapping_words.keys(),
                                                        self.mapping_words.values())}
        for k in range(len(weights_by_topics.keys())):
            for i in range(len(weights_by_topics[k])):
                tup = weights_by_topics[k][i]
                i_w, weight = tup[0], tup[1]
                weights_by_topics[k][i] = mapping_inv[i_w], weight

        return Jar(PMI_arr=PMI_arr, weights_by_topics=weights_by_topics)

    # metrics
    def get_nmi(self, labels_true) -> float:
        return self.model.get_nmi(labels_true)

    def get_ari(self, labels_true) -> float:
        return self.model.get_ari(labels_true)
