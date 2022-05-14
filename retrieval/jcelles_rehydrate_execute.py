from jcelles_rehydrate_module import rehydrate
from twarc import Twarc2, expansions
import pandas as pd

# Replace your bearer token below
your_bearer_token = ""
if not your_bearer_token:
    print("No bearer token supplied")

# List of Tweet IDs you want to lookup
df = pd.read_csv('2020-04-03_clean-dataset.tsv',sep='\t')
df['tweet_id'] = df['tweet_id'].astype(str)

############################
# FUNCTION PARAMETERS
############################

client = Twarc2(bearer_token = your_bearer_token)
outfile_name = 'rehydrated_tweets.json'
tweet_ids = list(df['tweet_id'])[0:5] #subset to 5 for the time being

############################
# REHYDRATE
############################
rehydrate(client, tweet_ids, outfile_name)
