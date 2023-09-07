import re
import zlib
from typing import Tuple, Any

import numpy.random as random

import numpy as np
import pandas as pd

import copy

from nltk.lm import Vocabulary
from wordcloud import WordCloud
import PIL

# data
from tweemo.db import IOSQL, Serializer, LocalCache

# model
from tweemo.tmo.topicmodels.globals import HYPERPARAMS
from tweemo.tmo.topicmodels import WrapperModel

# processing
from tweemo.tmo.processing import Processing

# dimred
from tweemo.tmo.dimred import ReductionEmbedding
from tweemo.tmo.dimred.globals import DEFAULT_PARAMS


class Controller:
    def __init__(
            self,
            app_name,
            dbname="tweets_db"
    ):
        np.random.seed(0)
        dbname = dbname
        self.app_name = app_name
        self.io = IOSQL(dbname)
        self.cache = LocalCache()
        self.wrapper = None
        self.df = pd.DataFrame()

    def get_table(self,
                  tb_name: str,
                  columns: list[str] = ["*"]) -> pd.DataFrame:
        return self.io.read_table(tb_name, columns)

    def get_df(self,
               tb_name: str,
               nrows_classes: dict[str, int],
               columns: list[str] = ["*"],
               lang="en") -> pd.DataFrame:

        df = pd.DataFrame(
            self.get_table(tb_name, columns=columns),
            columns=columns
        )

        [df[c].astype(str, copy=False) for c in df.columns]
        df = df.loc[df["lang"] == lang]

        for class_ in nrows_classes.keys():
            if nrows_classes[class_] == "all":
                nrows_classes[class_] = df.loc[df["class"] == class_].shape[0]

        classes = list(df["class"].unique())
        rows_dict = {class_name: None for class_name in classes}

        for class_name in classes:
            df_sub = df.loc[df["class"] == class_name]
            rows_dict[class_name] = random.choice(
                df_sub.index.values,
                size=min(df_sub.shape[0], nrows_classes[class_name])
            )

        # ---#
        df["text_tokenized"] = \
            df["text_tokenized"].apply(lambda tweet: " ".join(re.findall(r"[a-zA-Z]+", tweet)))

        return pd.concat(
            [df.loc[rows_dict[class_name]]
             for class_name in list(df["class"].unique())]
        )

    def get_vocabulary(
            self,
            df: pd.DataFrame,
            min_count=1
    ):
        words = Processing.list_words_from_sentences(list(df["text_tokenized"]))
        return list(Processing.create_vocab(words, min_count=min_count))

    def get_tweets_count(
            self,
            df: pd.DataFrame
    ) -> pd.DataFrame:
        return df["text_tokenized"].apply(lambda tweet: len(tweet.split()))

    def get_token_count(
            self,
            df: pd.DataFrame,
            pos=15
    ) -> list:
        dt_mat, voca_map = Processing.get_dt_matrix(df["text_tokenized"])
        voca_map = {value: key for key, value in zip(voca_map.keys(), voca_map.values())}
        tokens_count = np.sum(dt_mat, axis=0)
        tokens_dict = {voca_map[i]: tokens_count[0, i] for i in range(dt_mat.shape[1])}
        return list(dict(sorted(tokens_dict.items(), key=lambda t: t[1], reverse=True)).items())[
               :pos]

    def get_classes(
            self,
            s: pd.Series
    ) -> list:
        return list(s.unique())

    def get_modeles_name(self):
        return \
            {
                "SeaNMF": "SeaNMFL1",
                "NMF": "NMF"
            }

    def get_reduc_method(self):
        return \
            {
                "PCA": "PCA",
                "t-SNE": "TSNE",
                "Auto-encodeur": "AE",
                "UMAP": "UMAP"
            }

    def get_requirements(
            self,
            model_name: str,
            docs: list[str],
            mapping: dict,
            hyperparams: dict
    ) -> dict:
        if model_name == "NMF":
            A = Processing.create_td_matrix(docs=docs, vocab=mapping)
            requirements = {"A": A}
            requirements.update(hyperparams)
            return requirements
        elif model_name == "SeaNMFL1":
            A = Processing.create_td_matrix(docs=docs, vocab=mapping)
            S = Processing.create_log_co_occurences(docs, mapping)
            requirements = {"A": A, "S": S}
            requirements.update(hyperparams)
            return requirements
        else:
            raise Exception(f"Model not recognized : {model_name}")

    def load_model(
            self,
            df: pd.DataFrame,
            model_name: str,
            hyperparams: dict = None,
            min_count=1
    ) -> Vocabulary:
        assert min_count >= 1
        if not hyperparams:
            hyperparams = HYPERPARAMS[model_name]
        docs = list(df["text_tokenized"])
        vocab = Processing.create_vocab(Processing.list_words_from_sentences(docs),
                                        min_count=min_count)
        vocab_nltk = vocab
        docs = [" ".join(list(s)) for s in
                vocab.lookup([docs[i].split() for i in range(len(docs))])]
        vocab = [word for word in sorted(vocab) if
                 min_count > 1 or (word != '<UNK>' and min_count == 1)]
        mapping = {word: i for i, word in enumerate(vocab)}
        requirements = self.get_requirements(model_name, docs, mapping, hyperparams)
        self.wrapper = WrapperModel(model_name, mapping, requirements)
        return vocab_nltk

    def get_pmi_scores(self,
                       n_topKeyword) -> tuple[Any, Any]:
        jar = self.wrapper.get_pmi_scores(n_topKeyword)
        return jar.PMI_arr, jar.weights_by_topics

    def get_hyperparams(
            self,
            model_name: str,
            number_topics: int
    ) -> dict:
        hyperparams = copy.deepcopy(HYPERPARAMS[model_name])
        hyperparams["n_topic"] = number_topics
        return hyperparams

    def get_dict_freq_word(
            self,
            topic: int
    ) -> dict:
        topics = self.wrapper.get_topics()
        A = self.wrapper.model.A
        tweets_topic = [i for i in range(A.shape[1]) if topics[i] == topic]
        if len(tweets_topic) == 0:  # case if topic don't have tweets assigned to him
            return {}
        A_sub = A[:, tweets_topic]
        mapping_i_to_w = {i_w: w for w, i_w in zip(self.wrapper.mapping_words.keys(),
                                                   self.wrapper.mapping_words.values())}

        dict_freq = {mapping_i_to_w[i]: np.sum(A_sub[i, :], keepdims=False) \
                     for i in mapping_i_to_w.keys()}
        return dict_freq

    def plot_wc(
            self,
            dict_freq: dict[str, int],
            width=1201,
            height=200
    ) -> PIL.Image.Image:
        wc = WordCloud(width=width, height=height)
        wc.fit_words(dict_freq)
        return wc.to_image()

    def get_tweets_embeddings_reduc(
            self,
            params
    ) -> pd.DataFrame:
        method = params[0]
        hash_params = str(hash("".join([str(params[i]) for i in range(len(params))])))
        if self.cache.get(hash_params) == None:
            # Jar
            orig_embeddings = self.wrapper.fit_transform(all_decomposition=True).H
            rd = ReductionEmbedding(orig_embeddings)
            reduc_embeddings = rd.create_reduced_matrix(method,
                                                        DEFAULT_PARAMS[method])
            self.cache.set(hash_params,
                           zlib.compress(Serializer.dumps(reduc_embeddings)))
        else:
            obj_bytes = zlib.decompress(self.cache.get(hash_params))
            reduc_embeddings = Serializer.loads(obj_bytes)
        df = pd.DataFrame(reduc_embeddings, columns=["0", "1"])
        df["topic_class"] = self.wrapper.get_topics()
        df["topic_class"] = df["topic_class"].astype(str)
        return df

    def verify_model_loaded(
            self,
            hash_true_load_model: int,
            params: list
    ) -> bool:
        hash_params = str(hash("".join([str(params[i]) for i in range(len(params))])))
        return hash_params == hash_true_load_model

    def get_data_table(
            self,
            max_words=51
    ) -> pd.DataFrame:

        if (self.df == None).sum().sum() == 0:
            table = self.df.copy()
            table["text_tokenized"] = table["text_tokenized"].apply(lambda tweet: tweet[:max_words])
            return table
        else:
            return None

    def if_exists(
            self,
            params: list
    ):
        hash_params = str(hash("".join([str(params[i]) for i in range(len(params))])))
        return self.cache.get(hash_params)

    def set_model(
            self,
            params: list
    ):
        hash_params = str(hash("".join([str(params[i]) for i in range(len(params))])))
        self.cache.set(
            hash_params,
            zlib.compress(Serializer.dumps(self.wrapper))
        )

    def get_model(
            self,
            params: list
    ):
        hash_params = str(hash("".join([str(params[i]) for i in range(len(params))])))
        obj_bytes = zlib.decompress(self.cache.get(hash_params))
        self.wrapper = Serializer.loads(obj_bytes)

    def get_metrics(self):
        labels_true = self.df["class"]
        df = pd.DataFrame()
        df["NMI"] = [self.wrapper.get_nmi(labels_true)]
        df["ARI"] = [self.wrapper.get_ari(labels_true)]
        df = df.set_index(pd.Index(["Metrics"]))
        return df
