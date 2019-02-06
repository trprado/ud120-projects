#!/usr/bin/python

"""
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn import tree

clf = tree.DecisionTreeClassifier(min_samples_split=40)

t0 = time()
clf.fit(features_train, labels_train)
print("tempo de treinamento:", round(time()-t0, 3), "s")

t0 = time()
pred = clf.predict(features_test)
print("tempo de predição:", round(time()-t0, 3), "s")

from sklearn.metrics import accuracy_score

acc = accuracy_score(pred, labels_test)
print(f'Accuracy: {acc}')

# Quizz 2
print(f'Features: {features_train.shape[1]}')

# Results Quizz 1
#########################################################
# no. of Chris training emails: 7936
# no. of Sara training emails: 7884
# tempo de treinamento: 146.061 s
# tempo de predição: 0.04 s
# Accuracy: 0.9772468714448237
#########################################################

# Results Quizz 2
#########################################################
# no. of Chris training emails: 7936
# no. of Sara training emails: 7884
# tempo de treinamento: 146.061 s
# tempo de predição: 0.04 s
# Accuracy: 0.9772468714448237
# Features: 3785
#########################################################