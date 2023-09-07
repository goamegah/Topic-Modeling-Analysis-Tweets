import os


class LocalCache:
    def __init__(
            self,
            dir_cache="/home/godwin/cache_tweets"
    ):
        self.dir_cache = dir_cache

    def set(
            self,
            key: str,
            value: bytes
    ):
        with open(f"{self.dir_cache}/{key}.data", "wb") as f:
            f.write(value)

    def get(self, key: str):
        try:
            with open(f"{self.dir_cache}/{key}.data", "rb") as f:
                data = f.read()
        except FileNotFoundError as e:
            return None
        return data
