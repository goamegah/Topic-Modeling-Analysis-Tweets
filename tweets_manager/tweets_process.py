import os
import argparse
import json
from preprocessing import Text

ROOT_PATH = os.getenv(
    'POETRY_TWEETSMODELING_PROJECT_ROOT_PATH',
    None
)
TWEETS_PATH = f'{ROOT_PATH}/tweets_storage/data' if ROOT_PATH is not None else '../tweets_storage' \
                                                                               '/data'

input_file_name = f'{TWEETS_PATH}/tweets.json'
output_file_name = f'{TWEETS_PATH}/tweets.txt'

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', default=input_file_name, help='intput file name (json)')
parser.add_argument('--output_file', default=output_file_name, help='output file name (txt)')
parser.add_argument('--topic', default="tweets", help='topic about tweets')
parser.add_argument('--output_file_ext', default="txt", help='output file extension')
args = parser.parse_args()

# instantiate
txt = Text()  # need to think about that

if args.output_file_ext == "txt":
    with open(args.input_file, "r") as file:
        for line in file:
            try:  # try something good
                tweets100 = json.loads(line)
                for tweet in tweets100:
                    if tweet["lang"] == "en":  # only english text
                        tweet_text = tweet["text"]
                        tweet_text_processed = txt.process_document(tweet_text)
                        with open(str(args.output_file) +
                                  f'_{args.topic}.{args.output_file_ext}', "a") as f:
                            f.write(tweet_text_processed)
                            f.write("\n")  # back to new line
            except json.decoder.JSONDecodeError as e:
                print(e)
else:  # not forget add <class> field
    pass
