#!/bin/sh

# everything start here !

# General information
#********************

# specify value of all of poetry env vars you want to use
# for security comment line below if you are sure not to set path
export POETRY_TWEETSMODELING_PROJECT_ROOT_PATH='/path/to/project/tweets-modeling' # root path of project
export POETRY_TWEETSMODELING_TWEETS_STORAGE_PATH='/path/to/project/tweets-modeling/tweets_storage' # root path of project

# Specific information: NOT SET CODE BELOW BEFORE READING IMPORTANT NOTE
#*********************

export POETRY_TWEETSMODELING_TWEETS_BEARER_TOKEN='mysecretkey'
# export POETRY_TWEETSMODELING_TWEETS_BEARER_TOKEN='mysecretkey'

# IMPORTANT Note:
# ***************

# For security, its highly recommend to set tweets tokens
# in Poetry env or whatever env you like to use. Do not specify it in this <file>.
# Especially we set > export POETRY_TWEETSMODELING_TWEETS_BEARER_TOKEN='mysecretkey'
# where, <mysecretkey> match to your credential information

# twitter key is used by "base/_tweets_getter" file in other to connect to twitter client api
# feel free to check this code attentively before set any credential
