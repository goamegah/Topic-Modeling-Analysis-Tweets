import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
from tweemo.tmo.dimred.globals import DEFAULT_PARAMS
from tweemo.tmo.dimred.InterfaceDeep import InterfaceDeep


class ReductionEmbedding:
    def __init__(self, X: np.ndarray):
        """
        :param X: n (n:number of sentences) rows time d (d: features dimension of sentence)
        """
        self.X = X

    def create_rmodel(self,
                      method="PCA",
                      params=DEFAULT_PARAMS["PCA"]) -> object:
        """
        :param method: name of the model for reduction
        :param params: parameters of the model for reduction
        :return: model for reduction
        """
        if method in {"PCA", "TSNE", "UMAP"}:
            model_class = globals()[method]
            model = model_class(**params)
            model.fit(self.X)
            return model
        if method == "AE":
            id_ = InterfaceDeep(**params["id"]) \
                if len(params["id"].keys()) > 0 \
                else InterfaceDeep()
            id_.fit(self.X, **params["train"])
            return id_

    def create_reduced_matrix(self,
                              method="PCA",
                              params=DEFAULT_PARAMS["PCA"]) -> np.ndarray:
        """
        :param method: name of the model for reduction
        :param params: parameters of the model for reduction
        :return: Matrix created by the model for the reduction of the dimension
        """
        model = self.create_rmodel(method=method, params=params)
        return model.fit_transform(self.X)
