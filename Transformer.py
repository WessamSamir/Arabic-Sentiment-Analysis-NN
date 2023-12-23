import re
import pandas as pd
from nltk import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential 
from keras.layers import Embedding, Dense, Flatten, Dropout, Attention, LayerNormalization,Input
from keras.optimizers import Adam
import tensorflow as tf
from transformers import TFBertModel, BertTokenizer
from Preprocessing_Functions import *
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model

def positional_encoding(max_len, d_model):
    positions = np.arange(max_len)[:, np.newaxis]
    angles = 1 / np.power(10000, (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / np.float32(d_model))
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])
    pos_encoding = angles[np.newaxis, ...]
    return pos_encoding

def transformer_encoding(inputs, d_model, num_heads, dropout=0.1):
    attn = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
    ffn = tf.keras.Sequential([Dense(64, activation="relu"), Dense(d_model),])
    layernorm1 = LayerNormalization(epsilon=1e-6)
    layernorm2 = LayerNormalization(epsilon=1e-6)
    dropout1 = Dropout(dropout)
    dropout2 = Dropout(dropout)
    
    attn_output = attn(query=inputs, value=inputs)
    attn_output = dropout1(attn_output)
    out1 = layernorm1(inputs + attn_output)
    ffn_output = ffn(out1)
    ffn_output = dropout2(ffn_output)
    transformer_output = layernorm2(out1 + ffn_output)
    return transformer_output

inputs = Input(shape=(50,))
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

embedding = Embedding(input_dim=44999, output_dim=128, input_length=50)(inputs)

pos_encoding = positional_encoding(50, 128)
pos_encoding_layer = tf.constant(pos_encoding, dtype=tf.float32)
positional_encoded = embedding + pos_encoding_layer


transformer_output = transformer_encoding(positional_encoded, d_model=128, num_heads=4, dropout=0.3)

flatten = Flatten()(transformer_output)
dense1 = Dense(128, activation='relu')(flatten)
dense2 = Dense(64, activation='relu')(dense1)
outputs = Dense(3, activation='softmax')(dense2)


model = tf.keras.Model(inputs=inputs, outputs=outputs)

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
print(Y_train)
print(validation_labels)

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
 
history = model.fit(X_train, Y_train, epochs=7, batch_size= 128, validation_data=(validation_articles, validation_labels))
 

test_data = pd.read_csv('Dataset/test _no_label.csv')
X_test, test_ids = preprocess_test(test_data)
 
predictions = model.predict(X_test)
plot_training_validation(history)
predicted_sentiments = []
for pred in predictions:
    if pred[0] > pred[1] and pred[0] > pred[2]:
        predicted_sentiments.append(-1)
    elif pred[2] > pred[1] and pred[2] > pred[0]:
        predicted_sentiments.append(1)
    else:
        predicted_sentiments.append(0)
 
submission_df = pd.DataFrame({'ID': test_ids, 'rating': predicted_sentiments})
submission_df = submission_df.head(1000)
submission_df.to_csv('Transformer_submission1.csv', index=False)
print("Passs-------------")
model.save('Transformer_model.h5')
loaded_model = load_model('Transformer_model.h5')
predictions = loaded_model.predict(X_test)