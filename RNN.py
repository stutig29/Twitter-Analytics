import numpy as np
import pandas as pd 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense,SimpleRNN
from sklearn.model_selection import train_test_split
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score

def vanilla_rnn():
    model = Sequential()
    model.add(SimpleRNN(50, input_shape = (24,1)))
    model.add(Dense(46))
    model.add(Dense(2,activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return model

np.random.seed(20)

data = pd.read_csv('sentiments_elections.csv')
data = data[['tweet','sentiment']]
data = data[data.sentiment != "neutral"]
data['tweet'] = data['tweet'].apply(lambda x: x.lower())

print(data[ data['sentiment'] == 'positive'].size)
print(data[ data['sentiment'] == 'negative'].size)
#print(data)
    
max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(data['tweet'].values)
X = tokenizer.texts_to_sequences(data['tweet'].values)
X = pad_sequences(X)
embed_dim = 128
units=196

Y = pd.get_dummies(data['sentiment']).values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.33, random_state = 2)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

X_train = np.array(X_train).reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = np.array(X_test).reshape((X_test.shape[0], X_test.shape[1], 1))


model = KerasClassifier(build_fn = vanilla_rnn, epochs = 10, batch_size = 27, verbose = 1)

model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
Y_test_ = np.argmax(Y_test, axis = 1)
print("acc: %.2f"%(accuracy_score(Y_pred, Y_test_)))
print("precision: %.2f"%(precision_score(Y_pred, Y_test_)))
print("recall: %.2f"%(recall_score(Y_pred, Y_test_)))




