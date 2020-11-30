import string
import re
import pandas as pd
import nltk
from nltk import pos_tag
from nltk import sent_tokenize
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
nltk.download('wordnet')


def simple_parse(filename):
    df = pd.read_csv(filename)
    df.haikus = df.haikus.apply(lambda x: x.lower())
    df.haikus = df.haikus.apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
    df.haikus = df.haikus.apply(lambda x: x.replace('\n', ' '))
    return df.haikus


def preprocess_text(sampleText):
    sampleText = re.sub(r'[^\w\s]',' ',sampleText)
    # sampleText.translate(str.maketrans('', '', string.punctuation))
    tokenizer = TweetTokenizer()
    #stop_words = set(stopwords.words('english')) 
    lemmatizer = WordNetLemmatizer()

    tokens = tokenizer.tokenize(sampleText)
    lemmas = []
    for word in tokens:
        #if word.isalnum() and not word in stop_words:
        if word.isalnum():
            word = word.lower()
            word = lemmatizer.lemmatize(word, pos = 'v')
            lemmas.append(word)
    tokens = lemmas
    while "" in tokens:
        tokens.remove("")
    while " " in tokens:
        tokens.remove(" ")
    while "\n" in tokens:
        tokens.remove("\n")
    while "\n\n" in tokens:
        tokens.remove("\n\n")
    return tokens
