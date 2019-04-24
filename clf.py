from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.model_selection import GridSearchCV

mn_nb_params = {
	'vect__ngram_range': [(1, 1), (1, 2)],
}

linear_svm_params = {
	'vect__ngram_range': [(1, 1), (1, 2)],
	'tfidf__use_idf': (True, False),
	'clf__estimator__C': [0.001, 0.01, 0.1, 1, 10]
}

knn_params = {
	'vect__ngram_range': [(1, 1), (1, 2)],
	'tfidf__use_idf': (True, False),
	'clf__estimator__n_neighbors': [3, 5, 7, 9],
}

def find_best_params(clf, parameters, train_data, train_targets):
	gs_clf = GridSearchCV(clf, parameters, cv = 5, iid = False, n_jobs = -1)
	gs_clf = gs_clf.fit(train_data, train_targets)
	print("Best score %f" % gs_clf.best_score_)
	for param_name in sorted(parameters.keys()):
		print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

def make_multinomial_nb():
	mn_nb_clf = Pipeline([('vect', CountVectorizer(stop_words = 'english', ngram_range = (1 ,2))),
					   ('clf', OneVsRestClassifier(MultinomialNB(fit_prior = True, class_prior = None)))])
	return mn_nb_clf

def make_linear_svm():
	linear_svm_clf = Pipeline([('vect', CountVectorizer(stop_words = 'english')),
                        ('tfidf', TfidfTransformer()),
   					   	('clf', OneVsRestClassifier(LinearSVC(C = 0.01), n_jobs = 1))])
	return linear_svm_clf

def make_knn():
	knn_clf = Pipeline([('vect', CountVectorizer(stop_words = 'english')),
                        ('tfidf', TfidfTransformer()),
   					   	('clf', OneVsRestClassifier(KNeighborsClassifier(n_neighbors = 7, n_jobs = 1)))])
	return knn_clf

def make_rocchio():
	return OneVsRestClassifier(NearestCentroid())
