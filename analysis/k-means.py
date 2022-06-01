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


home_path = Path(__file__).parent.parent
data_path = home_path.joinpath("data/dailies/")
fig_path = home_path.joinpath("plots/")
cleaning = home_path.joinpath("cleaning/")


# tweets = pd.DataFrame(pd.read_json(data_path.joinpath("rehydrated-2022-01-13.json")))

df = pd.read_pickle(data_path.joinpath("full_tweets_clean.pkl"))

df = pd.DataFrame(df)
df['text'] = df['text'].str.join(" ")
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


print("Making tf-idf vector")
tf_idf_vectorizor = TfidfVectorizer(stop_words = 'english')
print("Initializing dimensionality reduction")
truncate = TruncatedSVD()

print("Making features")
features = tf_idf_vectorizor.fit_transform(filtered_tweets)
print("Done making features")



print("Initializing kmeans")
model = KMeans(n_clusters=3, init='k-means++', max_iter=100, n_init=1)

# # was using scaler.fit_transform(scale(features.T)) as input for Y_sklearn
print("Reducing dimensions for tf-idf vector")
Y_sklearn = truncate.fit_transform(features)


# word_positions = {v: k for k, v in tf_idf_vectorizor.vocabulary_.items()}
# print(word_positions)

print("Fitting reduced tf-idf")
fitted = model.fit(Y_sklearn)
print("Predicting clusters from tf-idf")
predict = model.predict(Y_sklearn)

# testing 
# # dist_words = sorted(v for k, v in word_positions.items())
# # print(dist_words)
# for cluster in set(predict):
#     tfidf = features[predict == cluster]
#     tfidf[tfidf > 0] = 1

#     df = pd.DataFrame(tfidf)
#     print(df)

print(model.labels_)
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(Y_sklearn[:, 0], Y_sklearn[:, 1],c=predict ,s=50, cmap='viridis') # Plotting scatter plot 
ax.legend(predict)

centers2 = fitted.cluster_centers_ # It will give best possible coordinates of cluster center after fitting k-means

ax.scatter(centers2[:, 0], centers2[:, 1],c='black', s=300, alpha=0.6)
plt.savefig(fig_path.joinpath("cluster-unscaled.png"))




