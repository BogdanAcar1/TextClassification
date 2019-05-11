from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

import reuters_helper as rh

def geometric_prog(a, r, n):
	return [a * r ** (n - 1) for n in range (1, n + 1)]
"""
Best score 0.894259
vect__ngram_range: (1, 2)
"""
mn_nb_params = {
	'vect__ngram_range': [(1, 1), (1, 2)],
}

"""
Best score 0.939571
clf__estimator__C: 3.20361328125
"""
linear_svm_params = {
	'clf__estimator__C': geometric_prog(2 ** (-3), 1.5, 9),
}

"""
Best score 0.879974
clf__estimator__n_neighbors: 11
"""
knn_params = {
	'clf__estimator__n_neighbors': [1, 3, 5, 7, 9, 11],
}

def find_best_params(clf, parameters, train_data, train_targets):
	gs_clf = GridSearchCV(clf, parameters, cv = 5, iid = False, n_jobs = -1, scoring = 'f1_micro')
	gs_clf = gs_clf.fit(train_data, train_targets)
	print("Best score %f" % gs_clf.best_score_)
	for param_name in sorted(parameters.keys()):
		print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))
	print(clf.named_steps['clf'].estimator.__class__.__name__, gs_clf.cv_results_)

def make_multinomial_nb():
	mn_nb_clf = Pipeline([('vect', CountVectorizer(ngram_range = (1 ,2))),
					   ('clf', OneVsRestClassifier(MultinomialNB(fit_prior = True, class_prior = None)))])
	return mn_nb_clf

def make_linear_svm():
	linear_svm_clf = Pipeline([('vect', CountVectorizer()),
                        ('tfidf', TfidfTransformer()),
   					   	('clf', OneVsRestClassifier(LinearSVC(C = 3.20361328125), n_jobs = 1))])
	return linear_svm_clf

def make_knn():
	knn_clf = Pipeline([('vect', CountVectorizer()),
                        ('tfidf', TfidfTransformer()),
   					   	('clf', OneVsRestClassifier(KNeighborsClassifier(n_neighbors = 11, n_jobs = 1)))])
	return knn_clf

if __name__ == '__main__':
	train_data, train_targets, _, _ = rh.get_top_dataset()
	find_best_params(make_multinomial_nb(), mn_nb_params, train_data, train_targets)
	find_best_params(make_linear_svm(), linear_svm_params, train_data, train_targets)
	find_best_params(make_knn(), knn_params, train_data, train_targets)
