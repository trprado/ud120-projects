"""
===================================================
Faces recognition example using eigenfaces and SVMs
===================================================

The dataset used in this example is a preprocessed excerpt of the
"Labeled Faces in the Wild", aka LFW_:

  http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz (233MB)

  .. _LFW: http://vis-www.cs.umass.edu/lfw/

  original source: http://scikit-learn.org/stable/auto_examples/applications/face_recognition.html

"""

print(__doc__)

from time import time
import logging
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


###############################################################################
# Download the data, if not already on disk and load it as numpy arrays
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape
np.random.seed(42)

# for machine learning we use the data directly (as relative pixel
# position info is ignored by this model)
X = lfw_people.data
n_features = X.shape[1]

# the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)


###############################################################################
# Split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

###############################################################################
# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction
# n_components = 150
# Quizz 35
# n_components = 300
# n_components = 600
# Quizz 36
# n_components = 10
# n_components = 15
# n_components = 25
# n_components = 50
# n_components = 100
n_components = 250

print("Extracting the top %d eigenfaces from %d faces" % (n_components, X_train.shape[0]))
t0 = time()
pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X_train)
print("done in %0.3fs" % (time() - t0))

# print(f'Explained variance ration: {pca.explained_variance_ratio_}')
print(f'First principal component: {pca.explained_variance_ratio_[0]}')
print(f'Second principal component: {pca.explained_variance_ratio_[1]}')

eigenfaces = pca.components_.reshape((n_components, h, w))

print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("done in %0.3fs" % (time() - t0))


###############################################################################
# Train a SVM classification model

print("Fitting the classifier to the training set")
t0 = time()
param_grid = {
         'C': [1e3, 5e3, 1e4, 5e4, 1e5],
          'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
          }
# for sklearn version 0.16 or prior, the class_weight parameter value is 'auto'
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid, cv=5)
clf = clf.fit(X_train_pca, y_train)
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)


###############################################################################
# Quantitative evaluation of the model quality on the test set

print("Predicting the people names on the testing set")
t0 = time()
y_pred = clf.predict(X_test_pca)
print("done in %0.3fs" % (time() - t0))

print(classification_report(y_test, y_pred, target_names=target_names))
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))

###############################################################################
# Qualitative evaluation of the predictions using matplotlib

def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


# plot the result of the prediction on a portion of the test set

def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

prediction_titles = [title(y_pred, y_test, target_names, i)
                         for i in range(y_pred.shape[0])]

plot_gallery(X_test, prediction_titles, h, w)

# plot the gallery of the most significative eigenfaces

eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)

plt.show()

# Results
################################################################################
# Quizz 35
################################################################################
# Total dataset size:
# n_samples: 1288
# n_features: 1850
# n_classes: 7
# Extracting the top 150 eigenfaces from 966 faces
# done in 0.313s
# First principal component: 0.19346539676189423
# Second principal component: 0.15116848051548004
# Projecting the input data on the eigenfaces orthonormal basis
# done in 0.024s
# Fitting the classifier to the training set
# done in 97.153s
# Best estimator found by grid search:
# SVC(C=1000.0, cache_size=200, class_weight='balanced', coef0=0.0,
#   decision_function_shape='ovr', degree=3, gamma=0.005, kernel='rbf',
#   max_iter=-1, probability=False, random_state=None, shrinking=True,
#   tol=0.001, verbose=False)
# Predicting the people names on the testing set
# done in 0.082s
#                    precision    recall  f1-score   support
#
#      Ariel Sharon       0.78      0.54      0.64        13
#      Colin Powell       0.83      0.87      0.85        60
#   Donald Rumsfeld       0.94      0.63      0.76        27
#     George W Bush       0.82      0.98      0.89       146
# Gerhard Schroeder       0.95      0.80      0.87        25
#       Hugo Chavez       1.00      0.47      0.64        15
#        Tony Blair       0.97      0.81      0.88        36
#
#         micro avg       0.85      0.85      0.85       322
#         macro avg       0.90      0.73      0.79       322
#      weighted avg       0.87      0.85      0.85       322
#
# [[  7   1   0   5   0   0   0]
#  [  1  52   0   7   0   0   0]
#  [  1   2  17   7   0   0   0]
#  [  0   3   0 143   0   0   0]
#  [  0   1   0   3  20   0   1]
#  [  0   3   0   4   1   7   0]
#  [  0   1   1   5   0   0  29]]
#
# F1: 0.7879877685352235
################################################################################
# Quizz 35
################################################################################
#                    precision    recall  f1-score   support
#
#      Ariel Sharon       0.56      0.69      0.62        13
#      Colin Powell       0.76      0.90      0.82        60
#   Donald Rumsfeld       0.86      0.67      0.75        27
#     George W Bush       0.86      0.90      0.88       146
# Gerhard Schroeder       0.84      0.64      0.73        25
#       Hugo Chavez       0.80      0.53      0.64        15
#        Tony Blair       0.85      0.78      0.81        36
#
#         micro avg       0.82      0.82      0.82       322
#         macro avg       0.79      0.73      0.75       322
#      weighted avg       0.82      0.82      0.82       322
#
#                    precision    recall  f1-score   support
#
#      Ariel Sharon       0.58      0.85      0.69        13
#      Colin Powell       0.66      0.83      0.74        60
#   Donald Rumsfeld       0.50      0.67      0.57        27
#     George W Bush       0.89      0.74      0.81       146
# Gerhard Schroeder       0.61      0.56      0.58        25
#       Hugo Chavez       0.56      0.60      0.58        15
#        Tony Blair       0.74      0.64      0.69        36
#
#         micro avg       0.72      0.72      0.72       322
#         macro avg       0.65      0.70      0.66       322
#      weighted avg       0.75      0.72      0.73       322
#
# Answer : Could go either way
################################################################################
# Quizz 36
################################################################################
# n_components = 10
# F1-score: 0.12
#
# n_components = 15
# F1-score: 0.32
#
# n_components = 25
# F1-score: 0.62
#
# n_components = 50
# F1-score: 0.67
#
# n_components = 100
# F1-score: 0.67
#
# n_components = 200
# F1-score: 0.67
#
# Answer: Better
################################################################################
# Quizz 36
################################################################################
# Answer: Yes, performance starts to drop with many PCs
################################################################################
# Quizz 37
################################################################################
# Answer: Yes, performance starts to drop with many PCs