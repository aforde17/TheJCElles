import pandas as pd
from zipfile import ZipFile
from pathlib import Path


home_path = Path()
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
full_df = pd.concat(list_of_dfs)

full_df.to_csv("full_tweets.csv")