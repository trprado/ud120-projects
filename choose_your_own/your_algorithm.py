import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the
### visualization code (prettyPicture) to show you the decision boundary

# K nearst neighborhood
# from sklearn.neighbors import KNeighborsClassifier

# clf = KNeighborsClassifier(n_neighbors=1)

# clf.fit(features_train, labels_train)
# pred = clf.predict(features_test)

# from sklearn.metrics import accuracy_score
# acc = accuracy_score(pred, labels_test)

# print(f'KnN Accuracy: {acc}')

# Results
################################################################################
# n_neighbors=3
# KnN Accuracy: 0.936
#
# n_neighbors=2
# KnN Accuracy: 0.928
#
# n_neighbors=1
# KnN Accuracy: 0.94
################################################################################

# Adaboost
# from sklearn.ensemble import AdaBoostClassifier

# clf = AdaBoostClassifier(n_estimators=50, learning_rate=1)

# clf.fit(features_train, labels_train)
# pred = clf.predict(features_test)

# from sklearn.metrics import accuracy_score
# acc = accuracy_score(pred, labels_test)

# print(f'AdaBoost Accuracy: {acc}')

# Results
################################################################################
# n_estimators=50
# learning_rate=1
# AdaBoost Accuracy: 0.924
#
# n_estimators=10
# learning_rate=1
# AdaBoost Accuracy: 0.916
#
# n_estimators=100
# learning_rate=1
# AdaBoost Accuracy: 0.924
#
# Raising values ​​in learning_rate makes acuracy worse
################################################################################

# Random Forest
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=1000, max_features='auto', n_jobs=-1)

clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)

print(f'AdaBoost Accuracy: {acc}')

# Results
################################################################################
# n_estimators=10
# lmax_features='auto'
# AdaBoost Accuracy: 0.916
#
# n_estimators=100
# lmax_features='auto'
# AdaBoost Accuracy: 0.912
#
# n_estimators=1000
# lmax_features='auto'
# AdaBoost Accuracy: 0.92
################################################################################

try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
