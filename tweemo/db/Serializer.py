import os
import pickle


class Serializer:
    def __init__(self):
        pass

    @classmethod
    def dumps(cls, obj: object) -> str:
        return pickle.dumps(obj)

    @classmethod
    def dump(cls, obj: object, out_file="default.data") -> str:
        with open(out_file, "wb") as outfile:
            return pickle.dump(obj, outfile)

    @classmethod
    def load(cls, in_file="default.data") -> object:
        with open(in_file, "rb") as infile:
            obj = pickle.load(infile)
        os.remove(in_file)
        return obj

    @classmethod
    def loads(cls, obj_bytes: bytes) -> object:
        return pickle.loads(obj_bytes)
