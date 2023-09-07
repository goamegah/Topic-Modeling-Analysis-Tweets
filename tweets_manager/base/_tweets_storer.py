from typing import List

from ._base import Storer
from ._tweets_getter import TweetsGetter
import json


class TweetsStorer(Storer):
    def __init__(
            self,
            getter,
            method="local-disk",
            name="TweetsStorerDefault",
            file_name="default.json"
    ):
        """
        :param getter: getter object that it get tweets (object)
        :param method: method (string) to use for store data
        :param name: name of the storer instance
        """
        assert isinstance(getter, TweetsGetter)
        self.getter = getter
        super().__init__(
            data=self.getter.data,
            method=method,
            name=name
        )
        self.file_name = file_name
        self.n_topic = len(file_name)

    def store(self, mode="a"):
        assert mode == "a" or mode == "w"

        if self.method == "local-disk":
            try:
                with open(self.file_name, mode) as f:
                    f.write(json.dumps(self.getter.data, default=str))
                    f.write('\n')
            except Exception as e:
                self.getter.custom_logger.logger.error(
                    json.dumps(
                        {self.name: str(e)},
                        default=str
                    )
                )
