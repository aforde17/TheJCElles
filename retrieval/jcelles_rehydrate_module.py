from twarc import Twarc2, expansions
import pandas as pd
import json

def rehydrate(client, tweet_ids, outfile_name):
    '''
    Rehydrate tweets

    Input (tweet_ids): List of Tweet IDs
    Output: A json file of hydrated tweets
    '''
    # The tweet_lookup function allows
    lookup = client.tweet_lookup(tweet_ids = tweet_ids)
    for page in lookup:
        # The Twitter API v2 returns the Tweet information and the user, media etc.  separately
        # so we use expansions.flatten to get all the information in a single JSON
        result = expansions.flatten(page)
    
    with open(outfile_name, 'w') as fout:
        json.dump(result , fout)
