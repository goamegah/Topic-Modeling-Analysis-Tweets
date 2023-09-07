# parser
import os
import argparse
from base import CustomLog, TweetsGetter, TweetsStorer
import logging

root = os.getenv('POETRY_TWEETSMODELING_STORAGE_ROOT_PATH', './tweets')

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default=f'{root}/data', help='output data dir')
parser.add_argument('--log_file', default=f'{root}/logs/tweets_log.log', help='output logs data dir')
parser.add_argument('--data_file', default=f'{root}/data/tweets.json', help='output logs data dir')
parser.add_argument('--storage_method', default='local-disk', help='where store data collected')
parser.add_argument('--topic', default='geology', help='data topics')
parser.add_argument('--tweets_max_results', type=int, default=100, help='maximum tweets result')
args = parser.parse_args()


def main():
    custom_logger = CustomLog(
        level=logging.INFO,
        file_name=args.log_file
    )

    fetcher = TweetsGetter(
        args.topic,
        custom_logger=custom_logger,
        max_results=args.tweets_max_results
    )

    saver = TweetsStorer(
        getter=fetcher,
        method="local-disk",
        file_name=args.data_file
    )

    saver.store()


if __name__ == '__main__':
    main()
