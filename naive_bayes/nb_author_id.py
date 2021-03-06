"""
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project.

    Use a Naive Bayes Classifier to identify emails by their authors

    authors and labels:
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

from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()

t0 = time()
clf.fit(features_train, labels_train)
print("tempo de treinamento:", round(time()-t0, 3), "s")

t0 = time()
pred = clf.predict(features_test)
print("tempo de predição:", round(time()-t0, 3), "s")

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(labels_test, pred)
print('Acurracia: ', round(accuracy, 4))

# Results
#########################################################
# no. of Chris training emails: 7936
# no. of Sara training emails: 7884
# tempo de treinamento: 2.592 s
# tempo de predição: 0.298 s
# Acurracia:  0.9733
#########################################################