import re
import pandas as pd
from nltk import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder


def remove_punctuation(text):
    punc_pattern = r'[^\w\s]'
    text = re.sub(punc_pattern, '', text)
    # remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text


def preprocessing(data):

    stop_words = set()
    stop_words.update(set(stopwords.words('arabic')))
    stop_words.update(set(stopwords.words('english')))
    lemmatizer = WordNetLemmatizer()
    data['review_description'] = data['review_description'].apply(remove_punctuation)
    data['review_description'] = data['review_description'].apply(lambda x: " ".join(x for x in word_tokenize(x)))
    data['review_description'] = data['review_description'].apply(
        lambda x: " ".join(x for x in word_tokenize(x) if x not in stop_words))
    data['review_description'] = data['review_description'].apply(
        lambda x: " ".join([lemmatizer.lemmatize(word) for word in word_tokenize(x)]))

    encoder = LabelEncoder()
    fit_tokens = encoder.fit_transform(data['review_description'])
    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
    tfidf_vect.fit(data['review_description'])
    tfidf_values = tfidf_vect.transform(data['review_description'])

    return data


data = pd.read_csv('train.csv')
preprocessing(data)
