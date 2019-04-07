import xml.etree.ElementTree as ET
from os import listdir
from os.path import isfile, join
import re
from clf import *
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn import preprocessing
lb = preprocessing.MultiLabelBinarizer()
import itertools


top_categories = ["acq", "corn", "crude", "earn", "grain", "interest", "money-fx", "ship", "trade", "wheat"]

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
	return np.array(train_data), np.array(train_targets), np.array(test_data), np.array(test_targets)

def test_clf(clf):
	train_data, train_targets, test_data, test_targets = build_dataset("reuters21578-xml")
	train_targets = lb.fit_transform(train_targets)
	clf.fit(train_data, train_targets)
	predicted = clf.predict(test_data)
	test_targets = lb.fit_transform(test_targets)
	print(precision_recall_fscore_support(test_targets, predicted, average = "micro", warn_for = tuple()))

if __name__ == '__main__':
	print("SVM metrics")
	test_clf(make_svm())
	print("NB metrics")
	test_clf(make_nb())
