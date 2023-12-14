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
from keras.layers import LSTM, Dense, Embedding, SpatialDropout1D

# Functions for text preprocessing
def remove_punctuation(text):
    punc_pattern = r'[^\w\s]'
    text = re.sub(punc_pattern, '', text)
    # remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text

def preprocessing(data):
    X = data['review_description']
    Y = data['rating']
    stop_words = set()
    stop_words.update(set(stopwords.words('arabic')))
    stop_words.update(set(stopwords.words('english')))
    lemmatizer = WordNetLemmatizer()
    X = X.apply(remove_punctuation)
    X = X.apply(lambda x: " ".join(x for x in word_tokenize(x)))
    X = X.apply(lambda x: " ".join(x for x in word_tokenize(x) if x not in stop_words))
    X = X.apply(lambda x: " ".join([lemmatizer.lemmatize(word) for word in word_tokenize(x)]))

    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(X)
    sequences = tokenizer.texts_to_sequences(X)

    return pad_sequences(sequences, maxlen=100), Y

# Loading and preprocessing the training data
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

model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128, input_length=100))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(128, dropout=0.4, recurrent_dropout=0.4, return_sequences=True))
model.add(LSTM(64, dropout=0.4, recurrent_dropout=0.4))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


model.fit(X_train, Y_train, epochs=20, batch_size=128, validation_data=(validation_articles, validation_labels))

# Load and preprocess the test data
test_data = pd.read_csv('Dataset/test _no_label.csv')

# Preprocess the test data without 'rating' column
def preprocess_test(data):
    X_test = data['review_description']
    stop_words = set(stopwords.words('arabic'))
    stop_words.update(set(stopwords.words('english')))
    lemmatizer = WordNetLemmatizer()
    X_test = X_test.apply(remove_punctuation)
    X_test = X_test.apply(lambda x: " ".join(x for x in word_tokenize(x)))
    X_test = X_test.apply(lambda x: " ".join(x for x in word_tokenize(x) if x not in stop_words))
    X_test = X_test.apply(lambda x: " ".join([lemmatizer.lemmatize(word) for word in word_tokenize(x)]))

    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(X_test)
    sequences = tokenizer.texts_to_sequences(X_test)

    return pad_sequences(sequences, maxlen=100), data['ID']

test_X, test_ids = preprocess_test(test_data)

# Make predictions on the test data
class_mapping = {idx: label for idx, label in enumerate(encoder.classes_)}

# Make predictions on the test data
predictions = model.predict(test_X)
predicted_labels = [class_mapping[pred.argmax()] for pred in predictions]

# Map predicted labels to sentiment values: -1, 0, 1
sentiment_mapping = {0: -1, 1: 0, 2: 1}

# Apply sentiment mapping to predicted labels
predicted_sentiments = [sentiment_mapping.get(label, -1) for label in predicted_labels]


# Create submission DataFrame
submission_df = pd.DataFrame({'ID': test_ids, 'rating': predicted_sentiments})

# Ensure exactly 1000 lines in the submission file
submission_df = submission_df.head(1000)

# Save DataFrame to a CSV file
submission_df.to_csv('submission.csv', index=False)

