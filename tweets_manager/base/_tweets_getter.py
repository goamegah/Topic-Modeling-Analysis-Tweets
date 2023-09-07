import json
import os

from ._base import Getter
from ._custom_log import CustomLog
import tweepy


class TweetsGetter(Getter):
    def __init__(
            self,
            topic: str,
            name: str = "TweetsGetterDefault",
            max_results: int = 10,
            custom_logger: CustomLog = CustomLog()
    ):
        """
        :param name: name of the getter
        :param topics: list of topic that we want to retrieve tweets refer to this topics
        :param custom_logger: logger of the getter
        :param max_results: number of tweets received (>=10)

        :var data: tweets content (json dictionnary)
        """
        super().__init__(name=name)
        self.credential = os.getenv('POETRY_TWEETSMODELING_TWEETS_BEARER_TOKEN', None)
        self.topic = topic
        assert max_results >= 10
        self.max_results = max_results
        self.data = None
        self.custom_logger = custom_logger
        self.get_data()

    def _get_api_oauth2(self):
        """
        :return: return api object produced by tweepy to use Twitter api with OAuth 2.0 (with bearer token)
        """
        auth2 = tweepy.Client(bearer_token=self.credential) \
            if self.credential is not None \
            else None

        return auth2

    def get_data(self):
        """
        :return: retrieve data from Twitter API
        """
        client = self._get_api_oauth2()

        try:
            tweets_list = client.search_recent_tweets(
                query=f'{self.topic} -is:retweet',
                tweet_fields=["lang", "created_at", "id", "text",
                              "author_id", "context_annotations"],
                max_results=self.max_results
            ).data

            self.data = [
                {
                    "id": tweet.id,
                    "text": tweet.text,
                    "lang": tweet.lang,
                    "created_at": tweet.created_at,
                    "author_id": tweet.author_id,
                    "context_annotations": tweet.context_annotations
                } for tweet in tweets_list
            ]

            self.custom_logger.logger.info(
                json.dumps(
                    self.get_logs(),
                    default=str
                )
            )
        except Exception as e:
            self.custom_logger.logger.error(
                json.dumps(
                    self.get_logs(e),
                    default=str
                )
            )

    def get_logs(self, e=None):
        """
        :return: return logs for the step of getting data
        (exemple: number of bytes retrieved ...)
        """
        if e is None:
            return \
                {
                    "Topic": self.topic,
                    "Nombre de tweets recus": len(self.data)
                }
        else:
            return \
                {
                    "Topic": self.topic,
                    "Nombre de tweets recus": 0,
                    "Exception": str(e)
                }
