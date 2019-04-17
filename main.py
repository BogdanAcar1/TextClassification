import xml.etree.ElementTree as ET
from os import listdir
from os.path import isfile, join
import re
from clf import *
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn import preprocessing
lb = preprocessing.MultiLabelBinarizer()
import itertools
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
rs = RandomUnderSampler(random_state=0)
from collections import Counter
from pandas import DataFrame


#top_categories = ["acq", "corn", "crude", "earn", "grain", "interest", "money-fx", "ship", "trade", "wheat"]
top_categories = ["earn", "acq", "money-fx", "grain", "crude", "trade", "interest", "ship", "wheat", "corn"]

def read_dataset(dataset_path):
	 files = [join(dataset_path, f) for f in listdir(dataset_path) if isfile(join(dataset_path, f)) and re.compile("reut2-[0-9]{3}.xml").match(f)]
	 return files

def parse_xml(xml_file, train_data, train_targets, test_data, test_targets):
	tree = ET.parse(xml_file)
	root = tree.getroot()
	for article in root.findall("REUTERS"):
		if article.find("TEXT").find("BODY") != None: #and article.get("TOPICS") != None:
			text = article.find("TEXT").find("BODY").text
			topics = [topic.text for topic in article.find("TOPICS").findall("D")]
			labels = [top_categories.index(topic) for topic in topics if topic in top_categories]
			if labels != []:
				if article.get("LEWISSPLIT") == "TRAIN":
					train_data.append(text)
					train_targets.append(labels)
				elif article.get("LEWISSPLIT") == "TEST":
					test_data.append(text)
					test_targets.append(labels)

def build_dataset(dataset_path):
	dataset = read_dataset(dataset_path)
	train_data, train_targets, test_data, test_targets = [], [], [], []
	for xml_file in dataset:
		parse_xml(xml_file, train_data, train_targets, test_data, test_targets)
	###
	pipeline = Pipeline([('vect', CountVectorizer(stop_words = 'english')),('tfidf', TfidfTransformer())])
	train_data = pipeline.fit_transform(train_data)
	train_targets = lb.fit_transform(train_targets)
	test_data = pipeline.transform(test_data)
	test_targets = lb.transform(test_targets)
	###
	return train_data, np.array(train_targets), test_data, np.array(test_targets)

def test_models(clfs, train_data, train_targets, test_data, test_targets):
	metrics = {
		clf.estimator.__class__.__name__: {
	        "precision": [],
	        "recall": [],
	        "fscore": []
	    } for clf in clfs
	}

	for clf in clfs:
		clf.fit(train_data, train_targets)
		predicted = clf.predict(test_data)
		precision, recall, fscore, _ =  precision_recall_fscore_support(test_targets, predicted, average = None, warn_for = tuple())
		micro_avg =  precision_recall_fscore_support(test_targets, predicted, average = "micro", warn_for = tuple())
		#print(micro_avg)
		metrics[clf.estimator.__class__.__name__]["precision"] = precision
		metrics[clf.estimator.__class__.__name__]["recall"] = recall
		metrics[clf.estimator.__class__.__name__]["fscore"] = fscore
		metrics[clf.estimator.__class__.__name__]["micro_avg"] = micro_avg
	return metrics

def display_metrics(metrics):
	clfs = list(metrics.keys())
	fscores = {clf: metrics[clf]["fscore"] for clf in clfs}
	classes = {"classes": top_categories}
	fscores = {**classes, **fscores}
	micro_avg = {"classes": "micro-avg"}
	micro_avg = {**micro_avg, **{clf: metrics[clf]["micro_avg"][2] for clf in clfs}}
	micro_avg = DataFrame(micro_avg, index = [0])
	fscores = DataFrame(fscores).append(micro_avg)
	print(fscores)



def test_clf(clf, train_data, train_targets, test_data, test_targets):
	#train_data, train_targets = rs.fit_resample(train_data, train_targets)
	#print(train_targets.shape)
	clf.fit(train_data, train_targets)
	predicted = clf.predict(test_data)
	#test_targets = lb.fit_transform(test_targets)
	precision, recall, fscore, support =  precision_recall_fscore_support(test_targets, predicted, average = None, warn_for = tuple())
	print(f"class precision: {[float('%.2f' % p) for p in precision]}")
	print(f"class recall   : {[float('%.2f' % p) for p in recall]}")
	print(f"class fscore   : {[float('%.2f' % p) for p in fscore]}")
	print("micro-avg ", precision_recall_fscore_support(test_targets, predicted, average = "micro", warn_for = tuple()))
	print(support)

if __name__ == '__main__':
	models = [make_nb(), make_knn(), make_svm()]
	train_data, train_targets, test_data, test_targets = build_dataset("reuters21578-xml")
	metrics = test_models(models, train_data, train_targets, test_data, test_targets)
	display_metrics(metrics)
	# train_data, train_targets, test_data, test_targets = build_dataset("reuters21578-xml")
	# for i in range(10):
	# 	print(top_categories[i], sum(train_targets[:, i]))
	# print("SVM metrics")
	# test_clf(make_svm(), train_data, train_targets, test_data, test_targets)
	# print("NB metrics")
	# test_clf(make_nb(), train_data, train_targets, test_data, test_targets)
	# print("KNN metrics")
	# test_clf(make_knn(), train_data, train_targets, test_data, test_targets)
