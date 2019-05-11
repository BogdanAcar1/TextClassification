from clf import *
import numpy as np
from sklearn.metrics import classification_report
from process import *
from reuters_helper import get_top_dataset, top_categories
import warnings
import json
warnings.simplefilter("ignore")

def test_models(clfs, train_data, train_targets, test_data, test_targets):
	reports = {}
	for clf in clfs:
		clf.fit(train_data, train_targets)
		test_predicted = clf.predict(test_data)
		report = classification_report(test_targets, test_predicted, target_names = top_categories)
		print(f"Test data metrics for classifier {clf.named_steps['clf'].estimator.__class__.__name__}:")
		print(report)
		report = classification_report(test_targets, test_predicted, output_dict = True, target_names = top_categories)
		reports[clf.named_steps['clf'].estimator.__class__.__name__] = report
	return reports

if __name__ == '__main__':
	train_data, train_targets, test_data, test_targets = get_top_dataset()
	models = [make_multinomial_nb(), make_linear_svm(), make_knn()]
	metrics = test_models(models, train_data, train_targets, test_data, test_targets)
	with open("metrics.out", "w") as metrics_out:
		metrics_out.write(json.dumps(metrics, indent = 2));
