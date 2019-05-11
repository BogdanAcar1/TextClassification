from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import string
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))


def process_tokens(tokens, stemming = False, lemmatization = False):
    tokens = [w.lower() for w in tokens]
    words = [word for word in tokens if word.isalpha() and word not in stop_words]
    if stemming == True:
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in words]
    elif lemmatization == True:
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)
