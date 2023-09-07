import logging


class Getter:
    def __init__(self, data_source: str = None, name="GetterDefault"):
        """
        :param data_source: source of data who allow to get data
        :param name: name of the getter
        """
        self.data_source = data_source
        self.name = name

    def get_data(self):
        """
        :return: retrieve data from source without processing
        """
        raise NotImplementedError(f"get_data from {self.name}:{self} should be implemented !")

    def get_logs(self):
        """
        :return: return logs for the step of getting data (exemple: number of bytes retrieved ...)
                could be have very different/type for a specific data source
        """
        raise NotImplementedError(f"get_logs from {self.name}:{self} should be implemented !")


class Logger:
    def __init__(self, level: str):
        """
        :param level: hierarchy for logs (DEBUG < INFO < WARNING < ERROR < CRITICAL)
        """
        self.logger = logging.getLogger()
        self.level = level
        assert level in \
               {
                   logging.DEBUG,
                   logging.INFO,
                   logging.WARNING,
                   logging.ERROR,
                   logging.CRITICAL
               }


class Storer:
    """
        this class represent an object which take a data with a specific type,
         and store it with method pass in constructor of the class
         default method for storing is store in local disk.
    """

    def __init__(
            self,
            data: object,
            method="local_disk",
            name="StorerDefault"
    ):
        """
        :param data: store data thanks to method
        :param method: method (string) to use for store data
        :param name: name of the storer instance
        """
        self.data = data
        self.method = method
        self.name = name

    def store(self):
        raise NotImplementedError(f"store method from {self.name}:{self}should be implemented !")
