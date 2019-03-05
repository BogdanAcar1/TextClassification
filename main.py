import xml.etree.ElementTree as ET
from os import listdir
from os.path import isfile, join
import re
import nb
import numpy as np
from sklearn.model_selection import GridSearchCV
parameters = {
	'vect__ngram_range': [(1, 1), (1, 2)],
	'tfidf__use_idf': (True, False),
	'clf__alpha': (1e-2, 1e-3),
}

top_categories = ["acq", "corn", "crude", "earn", "grain", "interest", "money-fx", "ship", "trade", "wheat"]


def read_dataset(dataset_path):
	 files = [join(dataset_path, f) for f in listdir(dataset_path) if isfile(join(dataset_path, f)) and re.compile("reut2-[0-9]{3}.xml").match(f)]
	 return files

def parse_xml(xml_file, train_data, test_data):
	tree = ET.parse(xml_file)
	root = tree.getroot()
	for article in root.findall("REUTERS"):
		if article.find("TEXT").find("BODY") != None and article.get("TOPICS") != None:
			item = (article.find("TEXT").find("BODY").text, [])
			for topic in article.find("TOPICS").findall("D"):				
				if topic.text in top_categories:					
					item[1].append(topic.text)
			if len(item[1]) > 0:
				if article.get("LEWISSPLIT") == "TRAIN":
					train_data.append(item)
				elif article.get("LEWISSPLIT") == "TEST":
					test_data.append(item)

def build_dataset(dataset_path):
	dataset = read_dataset(dataset_path)
	train_data = []
	test_data = []				
	for xml_file in dataset:
		parse_xml(xml_file, train_data, test_data)
	return train_data, test_data

def parse_xml2(xml_file, train_data, train_targets, test_data, test_targets):
	tree = ET.parse(xml_file)
	root = tree.getroot()
	for article in root.findall("REUTERS"):
		if article.find("TEXT").find("BODY") != None and article.get("TOPICS") != None:
			text = article.find("TEXT").find("BODY").text
			for topic in article.find("TOPICS").findall("D"):				
				if topic.text in top_categories:										
					if article.get("LEWISSPLIT") == "TRAIN":
						train_data.append(text)
						train_targets.append(top_categories.index(topic.text))
					elif article.get("LEWISSPLIT") == "TEST":
						test_data.append(text)
						test_targets.append(top_categories.index(topic.text))

def build_dataset2(dataset_path):
	dataset = read_dataset(dataset_path)
	train_data, train_targets, test_data, test_targets = [], [], [], []
	for xml_file in dataset:
		parse_xml2(xml_file, train_data, train_targets, test_data, test_targets)
	return train_data, train_targets, test_data, test_targets

if __name__ == '__main__':
	train_data, train_targets, test_data, test_targets = build_dataset2("reuters21578-xml")	
	nb_clf = nb.makeNB()
	nb_clf.fit(train_data, train_targets)
	predicted = nb_clf.predict(test_data)
	print("Accuracy %f" % np.mean(predicted == test_targets))
	#gs_clf = GridSearchCV(nb_clf, parameters, cv=5, iid=False, n_jobs=-1)
	#gs_clf = gs_clf.fit(train_data, train_targets)
	#print("Best score %f" % gs_clf.best_score_)
	#for param_name in sorted(parameters.keys()):
	#	print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))