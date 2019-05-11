import nltk
from nltk.corpus import reuters
from sklearn import preprocessing
import numpy as np
from process import *

top_categories = ["earn", "acq", "money-fx", "grain", "crude", "trade", "interest", "ship", "wheat", "corn"]
nltk.download('reuters')

def intersect(l1, l2):
    return [x for x in l1 if x in l2]

def get_top_split_set(split):
    ids = [id for id in reuters.fileids(top_categories) if id.startswith(split)]
    data = [process_tokens(reuters.words(id), stemming = True) for id in ids]
    targets = [intersect(reuters.categories(id), top_categories) for id in ids]
    return data, targets

def get_top_dataset():
    lb = preprocessing.MultiLabelBinarizer(top_categories)
    train_data, train_targets = get_top_split_set("train")
    test_data, test_targets = get_top_split_set("test")
    train_targets = lb.fit_transform(train_targets)
    test_targets = lb.transform(test_targets)
    return np.array(train_data), np.array(train_targets), np.array(test_data), np.array(test_targets)
