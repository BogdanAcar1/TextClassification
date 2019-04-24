import xml.etree.ElementTree as ET
from os import listdir
from os.path import isfile, join
import re
from clf import *
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn import preprocessing
from process import *

top_categories = ["earn", "acq", "money-fx", "grain", "crude", "trade", "interest", "ship", "wheat", "corn"]

CORPUS_PATH = "reuters21578-xml"
CLASSES_PATH = "reuters21578-xml/all-topics-strings.lc.txt"

def read_classes(classes_path):
	classes = []
	with open(classes_path, "r") as classes_in:
		classes = classes_in.readlines()
	classes = [c.strip() for c in classes]
	return classes

def read_dataset(dataset_path):
	 files = [join(dataset_path, f) for f in listdir(dataset_path) if isfile(join(dataset_path, f)) and re.compile("reut2-[0-9]{3}.xml").match(f)]
	 return files

def parse_xml(xml_file, train_data, train_targets, test_data, test_targets):
	tree = ET.parse(xml_file)
	root = tree.getroot()
	for article in root.findall("REUTERS"):
		if article.find("TEXT").find("BODY") != None:
			text = process_text(article.find("TEXT").find("BODY").text, lemmatization = True)
			labels = [topic.text for topic in article.find("TOPICS").findall("D")]
			if labels != []:
				if article.get("LEWISSPLIT") == "TRAIN":
					train_data.append(text)
					train_targets.append(labels)
				elif article.get("LEWISSPLIT") == "TEST":
					test_data.append(text)
					test_targets.append(labels)

def build_dataset(dataset_path):
	lb = preprocessing.MultiLabelBinarizer(read_classes(CLASSES_PATH))
	dataset = read_dataset(dataset_path)
	train_data, train_targets, test_data, test_targets = [], [], [], []
	for xml_file in dataset:
		parse_xml(xml_file, train_data, train_targets, test_data, test_targets)
	train_targets = lb.fit_transform(train_targets)
	test_targets = lb.transform(test_targets)
	return train_data, np.array(train_targets), test_data, np.array(test_targets)

def test_models(clfs, train_data, train_targets, test_data, test_targets):
	reports = {}
	all_labels = read_classes(CLASSES_PATH)
	labels = [all_labels.index(c) for c in top_categories]
	for clf in clfs:
		clf.fit(train_data, train_targets)
		test_predicted = clf.predict(test_data)
		report = classification_report(test_targets, test_predicted, target_names = top_categories, labels = labels)
		print(report)
		report = classification_report(test_targets, test_predicted, output_dict = True, target_names = top_categories, labels = labels)
		reports[clf.named_steps['clf'].estimator.__class__.__name__] = report
	return reports

if __name__ == '__main__':
	train_data, train_targets, test_data, test_targets = build_dataset(CORPUS_PATH)
	models = [make_multinomial_nb(), make_linear_svm(),]# make_knn()]
	metrics = test_models(models, train_data, train_targets, test_data, test_targets)
	print(metrics)
