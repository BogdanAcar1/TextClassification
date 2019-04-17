import xml.etree.ElementTree as ET
from os import listdir
from os.path import isfile, join
import re
from clf import *
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn import preprocessing
from pandas import DataFrame

lb = preprocessing.MultiLabelBinarizer()
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
	train_targets = lb.fit_transform(train_targets)
	test_targets = lb.transform(test_targets)
	return train_data, np.array(train_targets), test_data, np.array(test_targets)

def test_models(clfs, train_data, train_targets, test_data, test_targets):
	metrics = {
		clf.named_steps['clf'].estimator.__class__.__name__: {
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
		metrics[clf.named_steps['clf'].estimator.__class__.__name__]["precision"] = precision
		metrics[clf.named_steps['clf'].estimator.__class__.__name__]["recall"] = recall
		metrics[clf.named_steps['clf'].estimator.__class__.__name__]["fscore"] = fscore
		metrics[clf.named_steps['clf'].estimator.__class__.__name__]["micro_avg"] = micro_avg
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

if __name__ == '__main__':
	train_data, train_targets, test_data, test_targets = build_dataset("reuters21578-xml")
	# clf = make_knn() # make_linear_svm()
	# print(clf.named_steps['clf'].estimator.get_params().keys())
	# find_best_params(clf, knn_params, train_data, train_targets)

	models = [make_multinomial_nb(), make_linear_svm(), make_knn()]
	metrics = test_models(models, train_data, train_targets, test_data, test_targets)
	display_metrics(metrics)
