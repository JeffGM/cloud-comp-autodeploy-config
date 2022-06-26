import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
from joblib import dump, load
from io import BytesIO

raw_training_data = pd.read_csv("./dataset/training.csv", on_bad_lines='skip', sep=';')
raw_test_data = pd.read_csv("./dataset/test.csv", on_bad_lines='skip', sep=';')

train_tweets = raw_training_data[["text"]]
train_is_tweet_american = raw_training_data[["country_code"]].apply(lambda x: x == "US").astype(int)

test_tweets = list(raw_training_data[["text"]].values.ravel())
test_is_tweet_american = list(raw_training_data[["country_code"]].apply(lambda x: x == "US").astype(int).values.ravel())

tweet_clf = Pipeline([
     ('vect', CountVectorizer()),
     ('tfidf', TfidfTransformer()),
     ('clf', MultinomialNB()),
])

tweet_clf.fit(train_tweets.values.ravel(), train_is_tweet_american.values.ravel())


dump(tweet_clf, 'american_tweet_recognizer_model.joblib')
clf = load('american_tweet_recognizer_model.joblib')

predicted = clf.predict(test_tweets)

print(np.mean(predicted == test_is_tweet_american))
print("test")
