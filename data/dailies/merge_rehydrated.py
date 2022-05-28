import pandas as pd
import os
from zipfile import ZipFile
from pathlib import Path
from datetime import datetime
from langdetect import detect

def clean_non_english(df):
    '''
    Cleans removes all non english tweets from an input data frame
    '''
    cleaned = []
    for _, row in df.iterrows():
        try:
            lang = detect(row['text'])
            if lang == 'en':
                cleaned.append(row)
        except:
            cleaned.append(row)
    return pd.DataFrame(cleaned)


home_path = Path(os.getcwd())
tweets_path = home_path.joinpath("rehydrated")

#Assumes all the jsons are in a zipped folder called full_rehydrated_tweets within 
# the dailies folder
zipped = home_path.joinpath("full_rehydrated_tweets.zip")

# Extracting the jsons
with ZipFile(zipped, 'r') as zip_ref:
    zip_ref.extractall(tweets_path)


# looping over all files in the directory
files = Path(tweets_path).glob('*')

list_of_dfs = []
for file in files:
    df = pd.read_json(file)
    # Pulling out date rather than datetime
    df['date'] = pd.to_datetime(df['created_at']).dt.date
    df = df[['date', 'text']]
    list_of_dfs.append(df)

# Concatnating the list of subsetted dfs
int_df = pd.concat(list_of_dfs)

mask_keywords =  ["mask", "wearamask", "masking", "masked", "unmask", "unmasked", "unmasking",
    "anti-mask", "maskon", "maskoff", "N95", "face cover", "face covering", "face covered", "mouth cover", 
    "mouth covering", "mouth covered", "nose cover", "nose covering", "nose covered", "cloth covering", 
    "cover your face", "coveryourface", "facemask", "face diaper", "n95", "n-95", "kn95", "kn-95", "respirator"]
 
mask_key = "|".join(mask_keywords)

full_df = int_df[int_df['text'].str.contains(mask_key)==True]

full_df = clean_non_english(full_df)

full_df.to_csv("full_tweets.csv", index = False)



