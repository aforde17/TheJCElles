import pandas as pd
import os
from pathlib import Path
from datetime import date


home_path = Path(os.getcwd())
tweets_path = home_path.joinpath("rehydrated")

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