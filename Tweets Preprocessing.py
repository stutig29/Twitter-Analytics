import pandas as pd
from textblob import TextBlob
from nltk.stem import PorterStemmer
from textblob import Word
import numpy as np
from nltk.corpus import stopwords
import ast
import re
import csv 

stop = stopwords.words('english')
train = pd.read_csv('uber2.csv',names=["Date&Time","tweet"])
train['actual_tweet']= train['tweet']
# initial b removal
train['tweet'] = train['tweet'].apply(ast.literal_eval).str.decode("utf-8")
#print(train['tweet'])
#RT@ removal
train['tweet'] = train['tweet'].apply(lambda x: re.compile('\#').sub('', re.compile('RT @|rt @').sub('@', x, count=1).strip()))
#print(train['tweet'])

#hastags,@,username removal
train['tweet'] = train['tweet'].apply(lambda x: " ".join(re.sub("b^\s+|(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)","",x).split()))
#print(train['tweet'])
train['actual_tweet']= train['tweet']
#common words removal
freq = pd.Series(' '.join(train['tweet']).split()).value_counts()[:10]
print(freq)
freq=list(freq.index)
train['tweet'] = train['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
#print(train['tweet'].head())


#rare words removal
freq = pd.Series(' '.join(train['tweet']).split()).value_counts()[-10:]
freq = list(freq.index)
train['tweet'] = train['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
#print(train['tweet'].head())

train['tweet'] = train['tweet'].apply(lambda x: " ".join(x.lower() for x in x.split()))
#print(train['tweet'].head())

train['tweet'] = train['tweet'].str.replace('[^\w\s]','')
#print(train['tweet'].head())

train['tweet'] = train['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

#spelling correction
train['tweet'][:5].apply(lambda x: __builtins__.str(TextBlob(x).correct()))
#print(train['tweet'].head())

#stemming
st = PorterStemmer()
train['tweet'][:5].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))
#lemmatisation
train['tweet'] = train['tweet'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
#print(train['tweet'].head())
#n-gram 
TextBlob(train['tweet'][0]).ngrams(2)
#TF
tf1 = (train['tweet'][1:2]).apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0).reset_index()
tf1.columns = ['words','tf']
#print(tf1)
#IDF
for i,word in enumerate(tf1['words']):
  tf1.loc[i, 'idf'] = np.log(train.shape[0]/(len(train[train['tweet'].str.contains(word)])))
#print(tf1)
  #TFIDF
tf1['tfidf'] = tf1['tf'] * tf1['idf']
#print(tf1)
def get_tweet_sentiment(tweet):
  '''
  Get sentiment value of the tweet text
  It can be either positive, negative or neutral
  '''
  # create TextBlob object of the passed tweet text
  blob = TextBlob(tweet)
 
  # get sentiment
  if blob.sentiment.polarity > 0:
    sentiment = 'positive'
  elif blob.sentiment.polarity < 0:
    sentiment = 'negative'
  else:
    sentiment = 'neutral'
 
  return sentiment

train['sentiment'] = train['tweet'].apply(lambda x: get_tweet_sentiment(x))
#train = train[train.sentiment != 'neutral']
#print(train[['tweet','sentiment']].head(3))
train[['actual_tweet','sentiment']].to_csv('uber.csv', encoding='utf-8', index=False)