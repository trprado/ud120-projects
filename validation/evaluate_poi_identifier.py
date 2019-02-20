"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""

import sys
import pickle
import numpy as np
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "rb") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list, sort_keys='../tools/python2_lesson13_keys.pkl')
labels, features = targetFeatureSplit(data)


### it's all yours from here forward!
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier().fit(features_train, labels_train)
pred = clf.predict(features_test)

from sklearn.metrics import accuracy_score

print(f'Accuracy tree: {accuracy_score(pred, labels_test)}')

# Quizz 28
print(f'Poi predictions: {pred.sum()}')

# Quizz 29
print(f'Peoples in test set: {len(features_test)}')

# Quizz 30
print(f'Accuracy if all is no-Poi: {accuracy_score([0 for _ in range(len(pred))], labels_test)}')

# Quizz 31
print(f'True postive: {sum([1 for i, j in zip(pred, labels_test) if i == j and i != 0])}')

# Quizz 32 to 33
from sklearn.metrics import recall_score, precision_score

print(f'Recall score: {recall_score(labels_test, pred)}')
print(f'Precision score: {precision_score(labels_test, pred)}')

# Quizz 34 to 39
from sklearn.metrics import confusion_matrix

predictions =      [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
true_predictions = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]

print(f'Confusion matrix:\n {confusion_matrix(true_predictions, predictions)}')

print(f'True postive: {sum([1 for i, j in zip(true_predictions, predictions) if i == j and i != 0])}')
print(f'True negative: {sum([1 for i, j in zip(true_predictions, predictions) if i == j and i != 1])}')
print(f'False positive: {sum([1 for i, j in zip(true_predictions, predictions) if i != j and i != 1])}')
print(f'False negative: {sum([1 for i, j in zip(true_predictions, predictions) if i != j and i != 0])}')

print(f'Precision score: {precision_score(true_predictions, predictions)}')
print(f'Recall score: {recall_score(true_predictions, predictions)}')


# Results
################################################################################
# Quizz 28
################################################################################
# Poi predictions: 4.0
#
################################################################################
# Quizz 29
################################################################################
# Peoples in test set: 29
#
################################################################################
# Quizz 30
################################################################################
# Accuracy if all is no-Poi: 0.8620689655172413
#
################################################################################
# Quizz 31
################################################################################
# True postive: 0
#
################################################################################
# Quizz 32 to 33
################################################################################
# Recall score: 0.0
# Precision score: 0.0
#
################################################################################
# Quizz 34 to 39
################################################################################
# Confusion matrix:
#  [[9 3]
#  [2 6]]
# True postive: 6
# True negative: 9
# False positive: 3
# False negative: 2
# Precision score: 0.6666666666666666
# Recall score: 0.75
#
################################################################################
# Quizz 40
################################################################################
# Answer: POI
#
################################################################################
# Quizz 41
################################################################################
# Answer: precision/recall
#
################################################################################
# Quizz 42
################################################################################
# Answer: recall/precision
#
################################################################################
# Quizz 43
################################################################################
# Answer: f1-score/low
#