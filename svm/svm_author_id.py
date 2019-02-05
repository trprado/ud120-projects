"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
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
# Quizz 3 to 6 and 8
# features_train = features_train[:len(features_train)//100]
# labels_train = labels_train[:len(labels_train)//100]

from sklearn.svm import SVC

# Quizz 1 to 3
# clf = SVC(kernel='linear')

# Quizz 4
# clf = SVC(kernel='rbf', gamma='auto')

# Quizz 5
# clf = SVC(kernel='rbf', gamma='auto', C=10.)
# clf = SVC(kernel='rbf', gamma='auto', C=100.)
# clf = SVC(kernel='rbf', gamma='auto', C=1000.)
# Quizz 5 to 9
clf = SVC(kernel='rbf', gamma='auto', C=10000.)


t0 = time()
clf.fit(features_train, labels_train)
print("tempo de treinamento:", round(time()-t0, 3), "s")

t0 = time()
pred = clf.predict(features_test)
print("tempo de predição:", round(time()-t0, 3), "s")


from  sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)
print('Accuracy: ', acc)

# Quizz 8
# pos = ', '.join([str(x) for x in pred[[10,26,50]]])
# print('Predition 10o, 26o and 50o:', pos)

# Quizz 9
import numpy as np
unique, counts = np.unique(pred, return_counts=True)
for val, count in zip(unique, counts):
    print(f'Predictions in {"Sara" if val == 0 else "Chris"}: {count}')

# Results Quiz 1
#########################################################
# no. of Chris training emails: 7936
# no. of Sara training emails: 7884
# Accuracy:  0.9840728100113766
#########################################################

# Results Quiz 2
#########################################################
# no. of Chris training emails: 7936
# no. of Sara training emails: 7884
# tempo de treinamento: 446.304 s
# tempo de predição: 41.41 s
# Accuracy:  0.9840728100113766
#########################################################

# Results Quiz 3
#########################################################
# no. of Chris training emails: 7936
# no. of Sara training emails: 7884
# tempo de treinamento: 0.239 s
# tempo de predição: 2.304 s
# Accuracy:  0.8845278725824801
#########################################################

# Results Quiz 4
#########################################################
# gamma='auto'
# no. of Chris training emails: 7936
# no. of Sara training emails: 7884
# tempo de treinamento: 0.264 s
# tempo de predição: 2.668 s
# Accuracy:  0.6160409556313993
#########################################################

# Results Quiz 5
#########################################################
# gamma='auto'

# C=10.
# no. of Chris training emails: 7936
# no. of Sara training emails: 7884
# tempo de treinamento: 0.277 s
# tempo de predição: 2.94 s
# Accuracy:  0.6160409556313993

# C=100.
# no. of Chris training emails: 7936
# no. of Sara training emails: 7884
# tempo de treinamento: 0.269 s
# tempo de predição: 2.661 s
# Accuracy:  0.6160409556313993

# C=1000.
# no. of Chris training emails: 7936
# no. of Sara training emails: 7884
# tempo de treinamento: 0.185 s
# tempo de predição: 1.765 s
# Accuracy:  0.8213879408418657

# C=10000.
# no. of Chris training emails: 7936
# no. of Sara training emails: 7884
# tempo de treinamento: 0.244 s
# tempo de predição: 2.177 s
# Accuracy:  0.8924914675767918
#########################################################

# Results Quiz 6
#########################################################
# gamma='auto'
# C=1.
# no. of Chris training emails: 7936
# no. of Sara training emails: 7884
# tempo de treinamento: 0.264 s
# tempo de predição: 2.668 s
# Accuracy:  0.6160409556313993

# gamma='auto'
# C=10000.
# no. of Chris training emails: 7936
# no. of Sara training emails: 7884
# tempo de treinamento: 0.244 s
# tempo de predição: 2.177 s
# Accuracy:  0.8924914675767918
#########################################################

# Results Quiz 7
#########################################################
# gamma='auto'
# C=10000.
# no. of Chris training emails: 7936
# no. of Sara training emails: 7884
# tempo de treinamento: 305.335 s
# tempo de predição: 27.926 s
# Accuracy:  0.9908987485779295
#########################################################

# Results Quiz 8
#########################################################
# gamma='auto'
# C=10000.
# no. of Chris training emails: 7936
# no. of Sara training emails: 7884
# tempo de treinamento: 0.246 s
# tempo de predição: 2.175 s
# Accuracy:  0.8924914675767918
# Predition 10o, 26o and 50o: 1, 0, 1
#########################################################

# Results Quiz 9
#########################################################
# gamma='auto'
# C=10000.
# no. of Chris training emails: 7936
# no. of Sara training emails: 7884
# tempo de treinamento: 295.269 s
# tempo de predição: 28.14 s
# Accuracy:  0.9908987485779295
# Predictions in Sara: 881
# Predictions in Chris: 877
#########################################################