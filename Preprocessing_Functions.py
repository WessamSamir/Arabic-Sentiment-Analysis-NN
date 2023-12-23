import re
import pandas as pd
from nltk import WordNetLemmatizer
from nltk import stem
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import re



def remove_punctuation(text):
    punc_pattern = r'(?<=\w)([^\w\s] |_) +(?=\w)'
    text = re.sub(punc_pattern, '', text)
   
    text = re.sub(r'\s+', ' ', text)
    return text


def remove_numbers(text):
    return re.sub(r'\d+', ' ', text)


def remove_unusual_sequences(text):
    return re.sub(r'(.)\1+', r'\1', text)



def remove_non_arabic(text):
   
    arabic_pattern = re.compile('[\u0600-\u06FF]+')

    words = word_tokenize(text)

    arabic_words = [word for word in words if arabic_pattern.fullmatch(word)]

    result_text = ' '.join(arabic_words)

    return result_text

def remove_emojis(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F700-\U0001F77F"  # alchemical symbols
                               u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                               u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                               u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                               u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                               u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                               u"\U00002702-\U000027B0"  # Dingbats
                               u"\U000024C2-\U0001F251" 
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def preprocessing(data):

    X = data['review_description'].fillna('')
    Y = data['rating']
    stop_words = set()
    stop_words.update(set(stopwords.words('arabic')))
    stop_words.update(set(stopwords.words('english')))
    lemmatizer = WordNetLemmatizer()
    X = X.apply(remove_punctuation)

    X = X.apply(remove_numbers)

    X = X.apply(remove_unusual_sequences)

    X = X.apply(remove_non_arabic)

    X = X.apply(remove_emojis)
    
    X = X.str.strip()
    non_empty_indices = X[X != ''].index
    X = X.loc[non_empty_indices]
    Y = Y.loc[non_empty_indices]

    X = X.apply(lambda x: " ".join(x for x in word_tokenize(x)))
    X = X.apply(lambda x: " ".join(x for x in word_tokenize(x) if x not in stop_words))
    X = X.apply(lambda x: " ".join([lemmatizer.lemmatize(word) for word in word_tokenize(x)]))

    tokenizer = Tokenizer(num_words=44999)
    tokenizer.fit_on_texts(X)
    sequences = tokenizer.texts_to_sequences(X)
    return pad_sequences(sequences, maxlen=50), Y



def preprocess_test(data):
    X_test = data['review_description']
    stop_words = set(stopwords.words('arabic'))
    stop_words.update(set(stopwords.words('english')))
    lemmatizer = WordNetLemmatizer()
    X_test = X_test.apply(remove_punctuation)

    X_test = X_test.apply(remove_numbers)

    X_test = X_test.apply(remove_unusual_sequences)

    X_test = X_test.apply(remove_non_arabic)

    X_test = X_test.apply(remove_emojis)



    X_test = X_test.apply(lambda x: " ".join(x for x in word_tokenize(x)))
    X_test = X_test.apply(lambda x: " ".join(x for x in word_tokenize(x) if x not in stop_words))
    X_test = X_test.apply(lambda x: " ".join([lemmatizer.lemmatize(word) for word in word_tokenize(x)]))

    tokenizer = Tokenizer(num_words=44999)
    tokenizer.fit_on_texts(X_test)
    sequences = tokenizer.texts_to_sequences(X_test)

    return pad_sequences(sequences, maxlen=50), data['ID']