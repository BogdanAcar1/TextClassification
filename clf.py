from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

stop_words = set(stopwords.words('english'))

def make_nb():
	"""
	Best parameters determined using grid search
	Best score 0.829362
	clf__alpha: 0.01
	tfidf__use_idf: True
	vect__ngram_range: (1, 2)
	"""
	nb_clf = Pipeline([('vect', CountVectorizer(stop_words = 'english')),
					   ('tfidf', TfidfTransformer()),
					   ('clf', OneVsRestClassifier(MultinomialNB()))])
	#nb_clf.set_params(clf__alpha = 0.01, tfidf__use_idf = True, vect__ngram_range = (1, 2))
	return nb_clf

def make_svm():
	svm_clf = text_clf = Pipeline([('vect', CountVectorizer(stop_words = 'english')),
		                           ('tfidf', TfidfTransformer()),
     							   ('clf', OneVsRestClassifier(SVC(kernel = 'linear')))])
	#svm_clf.set_params(clf__alpha = 0.01, tfidf__use_idf = True, vect__ngram_range = (1, 1))
	return svm_clf
