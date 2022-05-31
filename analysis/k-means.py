from cProfile import label
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk
import string
from pathlib import Path
from sklearn.cluster import KMeans 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize, StandardScaler, scale
from sklearn.metrics import pairwise_distances
import re
import nltk
from nltk.stem import WordNetLemmatizer
import gensim 
from gensim import corpora, models 
import matplotlib.pyplot as plt

nltk.download("stopwords")
nltk.download('wordnet')
nltk.download('omw-1.4')

stopwords = set(nltk.corpus.stopwords.words("english"))

stemmer = WordNetLemmatizer()

def pre_process(text, stopwords):
    text = str(text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z]",' ',text)
    text = text.lower()
    text = text.split()
    text = [stemmer.lemmatize(word) for word in text if len(word) > 3]
    text_without_sw = [word for word in text if word not in stopwords]
    text_without_sw = " ".join(text_without_sw)
    return text_without_sw


# polarity score on x
# domain on y?

home_path = Path(__file__).parent.parent
data_path = home_path.joinpath("data/dailies/")


# tweets = pd.DataFrame(pd.read_json(data_path.joinpath("rehydrated-2022-01-13.json")))

# df = pd.read_pickle(data_path.joinpath("full_tweets_clean.pkl"))
df = pd.read_csv(data_path.joinpath("full_tweets.csv"))
df = pd.DataFrame(df)

df = df[["date", "text"]]

keyword = ["mask", "wearamask", "masking", "N95", "face cover", "face covering", "face covered", "mouth cover", "mouth covering",
"mouth covered", "nose cover", "nose covering", "nose covered", "cover your face", "coveryourface"]


filtered_tweets = []

for tweet in df.itertuples():
    for word in keyword:
        if word in tweet.text:
            filtered_tweets.append(tweet.text)
        break
print("Done filtering")

token = []
for tweet_text in filtered_tweets:
    tokens = pre_process(tweet_text, stopwords)
    token.append(tokens)
print("Done selecting tweet text")


# # Testing VADER algorithm
# test_tweets = filtered_tweets[:10]

# analyzer = SentimentIntensityAnalyzer()


# polarity = []
# for sentence in filtered_tweets:
#     vs = analyzer.polarity_scores(sentence)
#     polarity.append(vs)


tf_idf_vectorizor = TfidfVectorizer(stop_words = 'english')
sklearn_pca = PCA(n_components=2)
scaler = StandardScaler()


features = tf_idf_vectorizor.fit_transform(token).toarray()
print("Done making features")

features_scaled = scale(features.T)
print("Done scaling")
features_std = scaler.fit_transform(features_scaled)
print("Done transforming")

Y_sklearn = sklearn_pca.fit_transform(features)
print("Done with Y_sklearn")

model = KMeans(n_clusters=3, init='k-means++', max_iter=100, n_init=1)
fitted = model.fit(features)
print("Done with fitting")
predict = model.predict(features)
print("Done fitting and predicting")

print(model.labels_)
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(Y_sklearn[:, 0], Y_sklearn[:, 1],c=predict ,s=50, cmap='viridis') # Plotting scatter plot 
ax.legend()
centers2 = fitted.cluster_centers_ # It will give best possible coordinates of cluster center after fitting k-means
ax.scatter(centers2[:, 0], centers2[:, 1],c='black', s=300, alpha=0.6)
plt.show()




