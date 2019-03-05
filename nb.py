from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

stop_words = set(stopwords.words('english'))

def lemmatize(word):
	return WordNetLemmatizer().lemmatize(word)

def normalize(word):
	return lemmatize(word.translate(str.maketrans('', '', string.punctuation)).lower())

def word_filter(word):
	return word not in stop_words and word.isalpha()

def process_data(data):
	"""
	(text, topics) = data
	words = word_tokenize(text)
	words = [normalize(word) for word in words]
	words = [word for word in words if word_filter(word)]
	"""
	counts = CountVectorizer().fit_transform(data)
	tfidf = TfidfTransformer().fit_transform(counts)
	#print(counts)
	print(counts.shape)

def makeNB():
	"""
	Best parameters determined using grid search
	Best score 0.829362
	clf__alpha: 0.01
	tfidf__use_idf: True
	vect__ngram_range: (1, 2)
	"""
	nb_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
	nb_clf.set_params(clf__alpha = 0.01, tfidf__use_idf = True, vect__ngram_range = (1, 2))
	return nb_clf