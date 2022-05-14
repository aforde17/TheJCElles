import pandas as pd
from pathlib import Path
import re
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download("stopwords")

stopwords = set(nltk.corpus.stopwords.words("english"))

home_path = Path(__file__).parent.parent
data_path = home_path.joinpath("data/")
df = pd.read_csv(data_path.joinpath("tweets_subset.csv"))

#Filter tweets by "mask"
df = df.loc[df.text.str.contains('mask',case=False)]

stemmer = WordNetLemmatizer()

def pre_process(text, stopwords):
    text = str(text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z]",' ',text)
    text = text.lower()
    text = text.split()
    text = [stemmer.lemmatize(word) for word in text]
    text_without_sw = [word for word in text if word not in stopwords]
    return text_without_sw

document = df[["text"]]

processed_data = []
for row in document:
    text = document["text"].values
    tokens = pre_process(text, stopwords)
    processed_data.append(tokens)

print(processed_data)




