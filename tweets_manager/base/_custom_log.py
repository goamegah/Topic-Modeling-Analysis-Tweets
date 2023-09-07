import logging
from ._base import Logger


class CustomLog(Logger):
    def __init__(
            self,
            level: str = logging.DEBUG,
            file_name: str = "logfile.log"
    ):
        """
        :param level: hierarchy for logs (DEBUG < INFO < WARNING < ERROR < CRITICAL)
        :param file_name: name of file which are store
        """
        super().__init__(level)
        self.formatter = logging.Formatter('%(asctime)s~%(levelname)s~%(message)s~module:%(module)s')
        console_handler = self.init_StreamHandler()
        file_handler = self.init_FileHandler(file_name)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        self.logger.setLevel(level)

    def init_StreamHandler(self):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.level)
        console_handler.setFormatter(self.formatter)
        return console_handler

    def init_FileHandler(self, file_name):
        file_handler = logging.FileHandler(file_name)
        file_handler.setFormatter(self.formatter)
        file_handler.setLevel(self.level)
        return file_handler
