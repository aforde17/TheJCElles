from cProfile import label
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pathlib import Path
from sklearn.cluster import KMeans, MiniBatchKMeans
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import nltk
import nltk
from collections import Counter
nltk.download('words')

english_vocab = set(w.lower() for w in nltk.corpus.words.words())
keywords = ["mask", "wearamask", "masking", 
            "masked", "unmask", "unmasked",
            "unmasking", "anti-mask", "maskon", 
            "maskoff", "N95", "face cover", "face covering", 
            "face covered", "mouth cover", "mouth covering", 
            "mouth covered", "nose cover", "nose covering", "nose covered", 
            "cloth covering", "cover your face", "coveryourface", "facemask", 
            "face diaper", "n95", "n-95", "kn95", "kn-95", "respirator","covid19", "wear", "people", "face", "do",
            "coronavirus", "covid", "sars"]

home_path = Path(__file__).parent.parent
data_path = home_path.joinpath("data/dailies/")
fig_path = home_path.joinpath("plots/")
cleaning = home_path.joinpath("cleaning/")


def elbow_method(Y_sklearn):
    """
    This is the function used to get optimal number of clusters in order to feed to the k-means clustering algorithm.
    """

    number_clusters = range(1, 7)  # Range of possible clusters that can be generated
    kmeans = [KMeans(n_clusters=i, max_iter = 600) for i in number_clusters] # Getting no. of clusters 

    score = [kmeans[i].fit(Y_sklearn).score(Y_sklearn) for i in range(len(kmeans))] # Getting score corresponding to each cluster.
    score = [i*-1 for i in score] # Getting list of positive scores.
    
    plt.plot(number_clusters, score)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Score')
    plt.title('Elbow Method')
    plt.show()
# tweets = pd.DataFrame(pd.read_json(data_path.joinpath("rehydrated-2022-01-13.json")))

df = pd.read_pickle(data_path.joinpath("full_tweets_clean.pkl"))

df = pd.DataFrame(df)
df['text'] = df['text'].str.join(" ")




df_copy = df.copy()
print("Making tf-idf vector")
tf_idf_vectorizor = TfidfVectorizer(stop_words = 'english')
print("Initializing dimensionality reduction")
truncate = TruncatedSVD()
print("Making features")
features = tf_idf_vectorizor.fit_transform(df_copy['text'])
print("Done making features")
print("Initializing kmeans")
model = KMeans(n_clusters=3, init='k-means++', max_iter=100, n_init=1)
# # was using scaler.fit_transform(scale(features.T)) as input for Y_sklearn
# print("Reducing dimensions for tf-idf vector")
Y_sklearn = truncate.fit_transform(features)
elbow_method(Y_sklearn)

# print("Fitting reduced tf-idf")
# fit = model.fit(features)
# print("Predicting clusters from tf-idf")
# predict = model.predict(features)
# df_copy['cluster'] = model.labels_
# df_copy['text'] = df_copy['text'].str.split()

# # for n = 3
# d1 = {}
# d2 = {}
# d3 = {}
# for row in df_copy[['text', 'cluster']].itertuples():
#     for word in row.text:
#         if word not in keywords:
#             if row.cluster == 0:
#                 if word not in d2.keys() and word not in d3.keys():
#                     if word in english_vocab:
#                         d1[word] = d1.get(word, 0) + 1
#             if row.cluster == 1:
#                 if word not in d1.keys() and word not in d3.keys():
#                     if word in english_vocab:
#                         d2[word] = d2.get(word, 0) + 1
#             if row.cluster == 2:
#                 if word not in d1.keys() and word not in d2.keys():
#                     if word in english_vocab:
#                         d3[word] = d3.get(word, 0) + 1
# common_cluster_one = Counter(d1).most_common(20)
# common_cluster_two = Counter(d2).most_common(20)
# common_cluster_three = Counter(d3).most_common(20)
# for idx, (word, _) in enumerate(common_cluster_one):
#     common_cluster_one[idx] = word
# for idx, (word, _) in enumerate(common_cluster_two):
#     common_cluster_two[idx] = word
# for idx, (word, _) in enumerate(common_cluster_three):
#     common_cluster_three[idx] = word

# def models(n_clusters, df):
#     copy = df.copy()
#     d = {}
#     for row in df_copy[['text', 'cluster']].itertuples():
#         for word in row.text:
#             if word not in keywords:
#                 d[row.cluster] = d.get(row.cluster, {})
#                 lst = [y for x, y in d.items() if x != row.cluster]
#                 d[row.cluster][word] = d[row.cluster].get(word, 0) + 1

#                 if row.cluster == 0:
#                     if word not in d2.keys() and word not in d3.keys():
#                         if word in english_vocab:
#                             d1[word] = d1.get(word, 0) + 1
#                 if row.cluster == 1:
#                     if word not in d1.keys() and word not in d3.keys():
#                         if word in english_vocab:
#                             d2[word] = d2.get(word, 0) + 1
#                 if row.cluster == 2:
#                     if word not in d1.keys() and word not in d2.keys():
#                         if word in english_vocab:
#                             d3[word] = d3.get(word, 0) + 1
#     common_cluster_one = Counter(d1).most_common(20)
#     common_cluster_two = Counter(d2).most_common(20)
#     common_cluster_three = Counter(d3).most_common(20)
#     for idx, (word, _) in enumerate(common_cluster_one):
#         common_cluster_one[idx] = word
#     for idx, (word, _) in enumerate(common_cluster_two):
#         common_cluster_two[idx] = word
#     for idx, (word, _) in enumerate(common_cluster_three):
#         common_cluster_three[idx] = word
