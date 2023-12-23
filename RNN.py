import re
import numpy as np
import pandas as pd
from keras.src.callbacks import EarlyStopping
from nltk import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, SpatialDropout1D
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D
from tensorflow.python.keras.layers import Flatten
import matplotlib.pyplot as plt
from Preprocessing_Functions import*
from keras.layers import Dropout, SimpleRNN
from keras.models import load_model

def plot_training_validation(history):
   
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()


    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

data = pd.read_csv('Dataset/train.csv')

X, Y = preprocessing(data)

training_portion = 0.8
train_size = int(len(X) * training_portion)

X_train = X[0:train_size]
Y_train = Y[0:train_size]

validation_articles = X[train_size:]
validation_labels = Y[train_size:]

encoder = LabelEncoder()
Y_train = encoder.fit_transform(Y_train)
validation_labels = encoder.transform(validation_labels)

print(np.unique(Y_train))

model = Sequential()
model.add(Embedding(input_dim=44999, output_dim=16, input_length=50))
model.add(Dropout(0.5))
model.add(SimpleRNN(units=16, return_sequences=True,dropout=0.5, recurrent_dropout=0.1))
model.add(Dropout(0.5))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(3, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'] )


early = EarlyStopping(monitor='val_loss' , patience=3,restore_best_weights=True)

history = model.fit(X_train, Y_train, epochs=10, validation_data=(validation_articles, validation_labels ) , callbacks=[early])




val_acc = history.history['val_accuracy'][-1]
print(f"Final Validation Accuracy: {val_acc}")
plot_training_validation(history)

test_data = pd.read_csv('Dataset/test _no_label.csv')


test_X, test_ids = preprocess_test(test_data)

class_mapping = {idx: label for idx, label in enumerate(encoder.classes_)}

predictions = model.predict(test_X)

print(predictions)

predicted_sentiments = []
for pred in predictions:
    if pred[0] > pred[1] and pred[0] > pred[2]:
        predicted_sentiments.append(-1)
    elif pred[2] > pred[1] and pred[2] > pred[0]:
        predicted_sentiments.append(1)
    elif pred[1] > pred[0] and pred[1] > pred[2]:
        predicted_sentiments.append(0)


submission_df = pd.DataFrame({'ID': test_ids, 'rating': predicted_sentiments})

submission_df = submission_df.head(1000)


submission_df.to_csv('submission_RNN.csv', index=False)

model.save('RNN_Model.h5')
loaded_model = load_model('RNN_Model.h5')
predictions = loaded_model.predict(test_X)




