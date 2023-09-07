#!/bin/sh
. ./setenv.sh

# récupérer la variable POETRY_TWEETSMODELING_STORAGE_ROOT_PATH si la variable existe
PROJECT_DIR=${POETRY_TWEETSMODELING_PROJECT_ROOT_PATH:-1}

# repertoire où stocker les tweets.
# Si inexistant, alors le créer
TWEETS_STORAGE_DIR=${PROJECT_DIR}/tweets_storage
[ ! -d "$TWEETS_STORAGE_DIR" ] && mkdir -p "$TWEETS_STORAGE_DIR" # create dir if it doesn't exist

# scripts_dir="$PROJECT_DIR/tweets_manager"
data_dir="$TWEETS_STORAGE_DIR/data"
logs_dir="$TWEETS_STORAGE_DIR/logs"

[ ! -d "$data_dir" ] && mkdir -p "$data_dir"
[ ! -d "$logs_dir" ] && mkdir -p "$logs_dir"

# définir les params du script de collect des tweets
topics="movies politics sports" # you can add another topic separated by space
storage_method="local-disk"
for t in $topics; do
  data_file="$data_dir/tweets_$t.json"
  log_file="$logs_dir/tweets_$t.log"

  # gérer l'exécution avec poetry
  poetry run python3 "$PROJECT_DIR/tweets_manager/tweets_collect.py" --data_dir "$data_dir" \
      --log_file "$log_file" \
      --data_file "$data_file" \
      --storage_method $storage_method \
      --topic "$t" \
      --tweets_max_results 100
done

