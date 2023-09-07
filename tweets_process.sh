#!/bin/sh

# en premier lieu spécifier le chemin vers le dossier du projets
# dans la variable POETRY_TWEETSMODELING_STORAGE_ROOT_PATH
# $ export POETRY_TWEETSMODELING_STORAGE_ROOT_PATH="path/to/project/dir"

# récupérer la variable POETRY_TWEETSMODELING_STORAGE_ROOT_PATH si la variable existe
PROJECT_DIR=${POETRY_TWEETSMODELING_PROJECT_ROOT_PATH:-1}

# repertoire où stocker les tweets.
# Si inexistant, alors le créer
TWEETS_STORAGE_DIR="${PROJECT_DIR}/tweets_storage"
[ ! -d "$TWEETS_STORAGE_DIR" ] && mkdir -p "$TWEETS_STORAGE_DIR" # create dir if it doesn't exist

# scripts_dir="$PROJECT_DIR/tweets_manager"
data_dir="$TWEETS_STORAGE_DIR/data"
#logs_dir="$TWEETS_STORAGE_DIR/logs"

[ ! -d "$data_dir" ] && mkdir -p "$data_dir"
#[ ! -d "$logs_dir" ] && mkdir -p "$logs_dir"

topics="movies politics sports"   # Must be the same as tweets_collect file. You can that some separated by space

for t in $topics; do
  input_file="$data_dir/tweets_$t.json"
  output_file="$data_dir/tweets_$t"
  output_file_ext="txt" # change it to csv if you want

  # gérer l'exécution avec poetry :)
  poetry run python3 "$PROJECT_DIR/tweets_manager/tweets_process.py" --input_file "$input_file" \
      --output_file "$output_file" \
      --topic "$t" \
      --output_file_ext "$output_file_ext"
done