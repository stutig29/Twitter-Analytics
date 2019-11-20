import numpy as np
import pandas as pd 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
import re
from sklearn.metrics import accuracy_score
np.random.seed(20)

data = pd.read_csv('senti_mumbai.csv')
data = data[['actual_tweet','sentiment']]
data = data[data.sentiment != "neutral"]
data['actual_tweet'] = data['actual_tweet'].apply(lambda x: x.lower())
data['actual_tweet'] = data['actual_tweet'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))

print(data[ data['sentiment'] == 'positive'].size)
print(data[ data['sentiment'] == 'negative'].size)
#print(data)

for idx,row in data.iterrows():
    row[0] = row[0].replace('rt',' ')
    
max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(data['actual_tweet'].values)
X = tokenizer.texts_to_sequences(data['actual_tweet'].values)
X = pad_sequences(X)
embed_dim = 128
lstm_out = 196

Y = pd.get_dummies(data['sentiment']).values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.33, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

model = Sequential()
model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2,return_sequences=True))

model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2,return_sequences=True))

model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2,return_sequences=True))

model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())

#print(Y_train[1])
#print(X_train[1])
batch_size = 32
model.fit(X_train, Y_train, epochs = 10, batch_size=batch_size, verbose = 2)

validation_size = 20

X_validate = X_test[-validation_size:]
Y_validate = Y_test[-validation_size:]
X_test = X_test[:-validation_size]
Y_test = Y_test[:-validation_size]
score,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)
print("score: %.2f" % (score))
print("acc: %.2f" % (acc))
from sklearn.metrics import precision_score, recall_score
Y_test_ = np.argmax(Y_test, axis = 1)
Y_pred = model.predict_classes(X_test)
print("precision: %.2f"%(precision_score(Y_pred, Y_test_)))
print("recall: %.2f"%(recall_score(Y_pred, Y_test_)))

'''pos_cnt, neg_cnt, pos_correct, neg_correct = 0, 0, 0, 0
for x in range(len(X_validate)):
    
    result = model.predict(X_validate[x].reshape(1,X_test.shape[1]),batch_size=1,verbose = 2)[0]
   
    if np.argmax(result) == np.argmax(Y_validate[x]):
        if np.argmax(Y_validate[x]) == 0:
            neg_correct += 1
        else:
            pos_correct += 1
       
    if np.argmax(Y_validate[x]) == 0:
        neg_cnt += 1
    else:
        pos_cnt += 1

print("POSITIVE TWEETS:",pos_cnt)
print("NEGATIVE TWEETS:",neg_cnt)

print("pos_acc", pos_correct/pos_cnt*100, "%")
print("neg_acc", neg_correct/neg_cnt*100, "%")'''

'''twt = ['service is good']
twt = tokenizer.texts_to_sequences(twt)
twt = pad_sequences(twt, maxlen=26, dtype='int32', value=0)
print(twt)
sentiment = model.predict(twt,batch_size=1,verbose = 2)[0]
if(np.argmax(sentiment) == 0):
    print("negative")
elif (np.argmax(sentiment) == 1):
    print("positive")'''