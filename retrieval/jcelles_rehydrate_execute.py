from jcelles_rehydrate_module import rehydrate
from twarc import Twarc2, expansions
import pandas as pd
import wget
from pathlib import Path
from config import client

# set directories
home_path = Path(__file__).parent.parent
tweet_ids_path = home_path.joinpath("data/dailies/tweet_ids")
rehydrated_path = home_path.joinpath("data/dailies/rehydrated")

############################
# GET DATA
############################
target_date = "2021-01-20"
dataset_URL = "".join(["https://github.com/thepanacealab/covid19_twitter/blob/master/dailies/", target_date, "/", target_date, "_clean-dataset.tsv.gz?raw=true"])

#Download the dataset (compressed in a GZ format)
#!wget dataset_URL -O clean-dataset.tsv.gz
outfile_tsv = str(tweet_ids_path) +'/clean-' + target_date + '.tsv.gz'
wget.download(dataset_URL, out = outfile_tsv)

# gets list of Tweet IDs to lookup; filter to English tweets only
df = pd.read_csv(outfile_tsv, sep='\t')
if "lang" in df.columns:
    df = df[df.lang == 'en']
df['tweet_id'] = df['tweet_id'].astype(str)

############################
# FUNCTION PARAMETERS
############################

# client = Twarc2(bearer_token = your_bearer_token)
outfile_name = str(rehydrated_path) + '/rehydrated-' + target_date + '.json'
tweet_ids = list(df['tweet_id'])[0:5] #subset to 5 for the time being

############################
# REHYDRATE
############################
rehydrate(client, tweet_ids, outfile_name)
