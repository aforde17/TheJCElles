'''
Testing out working with the tweets from the panacea lab

- Akila Forde
'''

import gzip
import shutil
import os
import wget
import csv
import linecache
from shutil import copyfile
import ipywidgets as widgets
import numpy as np
import pandas as pd

dataset_URL = "https://github.com/thepanacealab/covid19_twitter/blob/master/dailies/2020-07-14/2020-07-14_clean-dataset.tsv.gz?raw=true" #@param {type:"string"}


#Downloads the dataset (compressed in a GZ format)
#!wget dataset_URL -O clean-dataset.tsv.gz
wget.download(dataset_URL, out='clean-dataset.tsv.gz')

#Unzips the dataset and gets the TSV dataset
with gzip.open('clean-dataset.tsv.gz', 'rb') as f_in:
    with open('clean-dataset.tsv', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

#Deletes the compressed GZ file
os.unlink("clean-dataset.tsv.gz")

#Gets all possible languages from the dataset
df = pd.read_csv('clean-dataset.tsv',sep="\t")

#Filter for only english language tweets
if "lang" in df.columns:
    df = df[df.lang == 'en']

df.to_csv('clean-dataset.tsv',sep="\t")












