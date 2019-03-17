import xml.etree.ElementTree as ET
from os import listdir
from os.path import isfile, join
import re
import clf
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

top_categories = ["acq", "corn", "crude", "earn", "grain", "interest", "money-fx", "ship", "trade", "wheat"]

def read_dataset(dataset_path):
	 files = [join(dataset_path, f) for f in listdir(dataset_path) if isfile(join(dataset_path, f)) and re.compile("reut2-[0-9]{3}.xml").match(f)]
	 return files

def parse_xml(xml_file, train_data, train_targets, test_data, test_targets):
	tree = ET.parse(xml_file)
	root = tree.getroot()
	for article in root.findall("REUTERS"):
		if article.find("TEXT").find("BODY") != None and article.get("TOPICS") != None:
			text = article.find("TEXT").find("BODY").text
			topics = [topic.text for topic in article.find("TOPICS").findall("D")]
			labels = [1 if c in topics else 0 for c in top_categories]
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
	for label, category in enumerate(top_categories):
		clf.fit(train_data, train_targets)
		predicted = clf.predict(test_data)
		print(f"Test accuracy : {accuracy_score(test_targets[:, label], predicted)}")

if __name__ == '__main__':
	test_clf(clf.make_nb())
	test_clf(clf.make_svm())
