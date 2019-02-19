import pickle
import numpy as np
np.random.seed(42)


### The words (features) and authors (labels), already largely processed.
### These files should have been created from the previous (Lesson 10)
### mini-project.
words_file = "../text_learning/your_word_data.pkl"
authors_file = "../text_learning/your_email_authors.pkl"
word_data = pickle.load( open(words_file, "rb"))
authors = pickle.load( open(authors_file, "rb") )



### test_size is the percentage of events assigned to the test set (the
### remainder go into training)
### feature matrices changed to dense representations for compatibility with
### classifier functions in versions 0.15.2 and earlier
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(word_data, authors, test_size=0.1, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test).toarray()


### a classic way to overfit is to use a small number
### of data points and a large number of features;
### train on only 150 events to put ourselves in this regime
features_train = features_train[:150].toarray()
labels_train   = labels_train[:150]



### your code goes here
print(f'Train points: {len(features_train)}')

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)

pred = clf.predict(features_test)

from sklearn.metrics import accuracy_score
print(f'Accuracy: {accuracy_score(pred, labels_test)}')

importances = clf.feature_importances_
print(f'Feature importances: {importances[np.where(importances > 0.2)]}')

att_index = np.where(importances > 0.2)[0]
print(f'Position: {att_index}')

for index in att_index:
    print(f'Powerfull word: {vectorizer.get_feature_names()[index]}')

# Results
################################################################################
# Quizz 23
################################################################################
# Answer: Low
################################################################################
# Quizz 24
################################################################################
# Answer: High
################################################################################
# Quizz 25
################################################################################
# Train points: 150
################################################################################
# Quizz 26
################################################################################
# Train points: 150
# Accuracy: 0.9476678043230944
################################################################################
# Quizz 27
################################################################################
# Train points: 150
# Accuracy: 0.9476678043230944
# Feature importances: [0.76470588]
# Position: [33614]
################################################################################
# Quizz 28
################################################################################
# Train points: 150
# Accuracy: 0.9476678043230944
# Feature importances: [0.76470588]
# Position: [33614]
# Powerfull word: sshacklensf
################################################################################
# Quizz 29
################################################################################
# Train points: 150
# Accuracy: 0.9505119453924915
# Feature importances: [0.6666666666666667]
# Position: [14343]
# Powerfull word: cgermannsf
################################################################################
# Quizz 30
################################################################################
# Answer: Yes, there's one new important word
################################################################################
# Quizz 31
################################################################################
# Train points: 150
# Accuracy: 0.8168373151308305
# Feature importances: [0.36363636]
# Position: [21323]
# Powerfull word: houectect