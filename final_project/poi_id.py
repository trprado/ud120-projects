import sys
import pickle
sys.path.append("../tools/")

from tools_fp import select_best_features, remove_outliers, add_feature

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
# Open dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    dataset = pickle.load(data_file)
    dataset.pop('TOTAL')

select_features_list = ['poi', 'salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees', 'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']

selected_features = select_best_features(dataset, select_features_list, nan_perc=0.5)

# Funcão para selecionar as melhores features do dataset.
features_list = ['poi'] + list(selected_features.keys()) # You will need to use more features
print('='*80)
print(f'Selected feature list:')
print('='*80)
print(features_list)
print('-'*80)

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers

# Valores que se encontram no dataset e não são interessantes manter.
data_dict.pop('TOTAL', 0) # Apenas a soma de valores de todas os itens do dataset
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0) # Agencia de viagem que não tem relação direta com funcionarios.
data_dict.pop('LOCKHART EUGENE E', 0) # Pessoa com todos os valores nulos.

# Função para remover as linhas com outliers, não utilizado pois alguns valores
# considerados outliers são na realidade informações de POI's.
# data_dict = remove_outliers(data_dict, features_list)

# Teste usando KBest para verificar se pontuação e p-valor estão ideais nos
# features selecionados previamente.
data = featureFormat(data_dict, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

from sklearn.feature_selection import SelectKBest

features_new = SelectKBest(k='all')
features_new.fit(features, labels)
print()
print('='*80)
print('SelectKBest in selected features:')
print('='*80)
print(f'Best scores: {features_new.scores_.round(4)}')
print(f'P-values: {features_new.pvalues_.round(4)}')
print('-'*80)


### Task 3: Create new feature(s)
# Features usadas para se criar as novas features
# Salary se mostrou um dos melhores pontuadores e total_payment serve para para
# visualizar ganhos gerais.
select_to_new_features = ['salary', 'total_payments']

# Função gera novas features baseados em [x^2, x*y, y^2]
data_dict, features_list = add_feature(data_dict, features_list, select_to_new_features)

print('='*80)
print(f'Final feature list: \n{features_list}')
print(f'Count features: {len(features_list)}')
print('='*80)

# Teste usando KBest para verificar se pontuação e p-valor estão ideais nos
# features gerados previamente.
data = featureFormat(data_dict, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

from sklearn.feature_selection import SelectKBest

features_new = SelectKBest(k='all')
features_new.fit(features, labels)
print()
print('='*80)
print('SelectKBest in selected features:')
print('='*80)
print(f'Best scores: {features_new.scores_.round(4)}')
print(f'P-values: {features_new.pvalues_.round(4)}')
print('-'*80)

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Imports para fazer preprocessamento e decomposição além de criar o pipeline
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

################################################################################
################################################################################
import warnings
# Limpa alguns warnings por divisões por zero causadas dentro dos calculos
warnings.filterwarnings('ignore')

clfs = [
    ('svc', SVC(gamma='scale', random_state=42)),
    ('gnb', GaussianNB()),
    ('dtc', DecisionTreeClassifier(random_state=42)),
    ('knc', KNeighborsClassifier()),
    ('rfc', RandomForestClassifier(n_estimators=20, random_state=42)),
    ('ada', AdaBoostClassifier(random_state=42))
]

# Processos para o pipeline
estimators = [
    ('preproc', MinMaxScaler())
]

# Parâmetros para as validações com GridSearchCV
parameters = [
    {
        'svc__kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
        'svc__C': [1, 5, 10, 20, 50, 100]
    },
    {
        # Espaço das configurações do GNB
    },
    {
        'dtc__criterion': ['gini', 'entropy'],
        'dtc__max_depth': [1, 3, 5, 10, 15, 30],
        'dtc__min_samples_split': [2, 3, 5, 10, 15],
        'dtc__min_samples_leaf': [1, 3, 5, 10, 15],
        'dtc__min_impurity_decrease': [.0, .3, .5, 1]
    },
    {
        'knc__n_neighbors': [3, 5, 10, 15],
        'knc__leaf_size': [5, 10, 15, 30, 50],
        'knc__p': [1, 2, 3, 4]
    },
    {
        'rfc__min_samples_leaf': [2, 3, 4, 5, 10, 15],
        'rfc__max_depth': [3, 5, 10, 15, 30],
        'rfc__min_samples_split': [2, 5, 10, 15]
    },
    {
        'ada__n_estimators': [10, 20, 30, 40, 50, 60, 63, 65, 67, 70, 73, 75, 77, 80, 85, 90, 95, 100, 200, 500],
        'ada__learning_rate': [0.0001, 0.001, 0.1, 0.15, 0.2, 0.3, 0.5, 0.6, 1.0]
    }
]

from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV

# Valida e gera os scores para definir os melhores Classificadores, esta comentado pela demora ao executar todos os comparadores para algoritmos de árvore.
for i in range(0, len(clfs)):
    pipe = Pipeline([
        ('preproc', MinMaxScaler()),
        ('reduce', PCA()),
        clfs[i]])
    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
    clf = GridSearchCV(pipe, param_grid=parameters[i], scoring='f1', cv=sss, refit=True)
    clf.fit(features, labels)
    print()
    print('='*80)
    print(clfs[i])
    print('='*80)
    index = clf.best_index_
    print(f"Fit time mean: {clf.cv_results_['mean_fit_time'][index]:.4f}")
    print(f"F1 score: {clf.best_score_:.4f}")
    print(f'Best parameters: {clf.best_params_}')
    print('-'*80)

    # Gera documentos html com os scores de cada algoritmo.
    import pandas as pd
    pd.DataFrame.from_dict(clf.cv_results_, orient='columns').loc[:, ['params', 'mean_fit_time', 'mean_test_score']].to_html('results/' + clfs[i][0] + 'results.html')
################################################################################
################################################################################

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# ==============================================================================
# Melhor pontuação foi do SVC com pontuação F1 de 0.3022, porém devido a
# precisão e validação estarem a baixo de 0.3 em todos os testes realizados,
# foi descartado.
# Resultado:
# ==============================================================================
# ('svc', SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
#   decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
#   max_iter=-1, probability=False, random_state=42, shrinking=True,
#   tol=0.001, verbose=False))
# ==============================================================================
# Fit time mean: 0.0021
# F1 score: 0.3022
# Best parameters: {'svc__C': 20, 'svc__kernel': 'sigmoid'}
# ------------------------------------------------------------------------------

# ==============================================================================
# RandomForestClassifier teve uma baixa pontuação em Precisão e Validação
# ==============================================================================
# ('rfc', RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#             max_depth=None, max_features='auto', max_leaf_nodes=None,
#             min_impurity_decrease=0.0, min_impurity_split=None,
#             min_samples_leaf=1, min_samples_split=2,
#             min_weight_fraction_leaf=0.0, n_estimators=20, n_jobs=None,
#             oob_score=False, random_state=42, verbose=0, warm_start=False))
# ==============================================================================
# Fit time mean: 0.0261
# F1 score: 0.2507
# Best parameters: {'rfc__max_depth': 5, 'rfc__min_samples_leaf': 2, 'rfc__min_samples_split': 5}
# ------------------------------------------------------------------------------

# ==============================================================================
# AdaBoostClassifier foi o que teve uma das menores pontuações na Acuraica,
# porem foi o que obteve o melhor equilibrio de pontuação entre Precisão e
# Validação.
# ==============================================================================
# ('ada', AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
#           learning_rate=1.0, n_estimators=50, random_state=42))
# ==============================================================================
# Fit time mean: 0.0203
# F1 score: 0.2493
# Best parameters: {'ada__learning_rate': 0.5, 'ada__n_estimators': 10}
# ------------------------------------------------------------------------------

# ==============================================================================
# Os parâmetros encontrados não foram usados pois os mesmos geravam um baixo
# valor de Precisão e Validação, para isso foi realizado em cada um, um novo
# teste que definiria os melhores parâmetros.
# ==============================================================================

import numpy as np

# Transforma lista em array para acesso mais fácil ao retorno de split
labels = np.array(labels)

from sklearn.metrics import accuracy_score, precision_score, recall_score

minmaxs = MinMaxScaler()
features = minmaxs.fit_transform(features, labels)

# Melhores parâmetros encontrados para ter um balanço entre precision e recall:
# clf = AdaBoostClassifier(learning_rate=0.6, n_estimators=67)

sss = StratifiedShuffleSplit(n_splits=100, random_state=42)

# ==============================================================================
# Teste para encontrar os melhores parâmetros para SVC, pode demorar
# um pouco mais devido a quantidade de repetições e comparação entre
# parâmetros.
# ==============================================================================
# parameters = {
#     'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
#     'C': np.arange(start=1, stop=100, step=1)
# }
# temp = SVC()
# gs_cv = GridSearchCV(temp, param_grid=parameters, scoring='f1', cv=sss, refit=True)
# start_time = time.clock()
# gs_cv.fit(features, labels)
# print(f'Tempo execução: {time.clock() - start_time}s')
# print(f'Best precision: {gs_cv.best_score_}')
# print(f'Best params: {gs_cv.best_params_}')
# ==============================================================================
# clf = SVC(kernel='rbf', C=48, gamma='scale')
# ==============================================================================
# Pontuação:
# ==============================================================================
# Accuracy score: 0.8647
# Precision score: 0.1500
# Recall score: 0.0800
# ------------------------------------------------------------------------------

# ==============================================================================
# Teste para encontrar os melhores parâmetros para RandomForest, pode demorar
# um pouco mais devido a quantidade de repetições e comparação entre
# parâmetros.
# ==============================================================================
# parameters = {
#     'max_depth': np.arange(start=1, stop=5, step=1),
#     'min_samples_leaf': np.arange(start=2, stop=10, step=1),
#     'min_samples_split': np.arange(start=2, stop=10, step=1)
# }
# temp = RandomForestClassifier()
# gs_cv = GridSearchCV(temp, param_grid=parameters, scoring='f1', cv=sss, refit=True)
# start_time = time.clock()
# gs_cv.fit(features, labels)
# print(f'Tempo execução: {time.clock() - start_time}s')
# print(f'Best precision: {gs_cv.best_score_}')
# print(f'Best params: {gs_cv.best_params_}')
# ==============================================================================
# clf = RandomForestClassifier(max_depth=4, min_samples_leaf=2,
# min_samples_split=8)
# ==============================================================================
# Pontuação:
# ==============================================================================
# Accuracy score: 0.8580
# Precision score: 0.1983
# Recall score: 0.1350
# ------------------------------------------------------------------------------

# ==============================================================================
# Teste para encontrar os melhores parâmetros para AdaBoost, pode demorar um
# pouco mais devido a quantidade de repetições e comparação entre parâmetros.
# ==============================================================================
# parameters = {
#     'learning_rate': np.arange(start=.1, stop=1.1, step=.1),
#     'n_estimators': np.arange(start=10, stop=100, step=1)
# }
# temp = AdaBoostClassifier()
# gs_cv = GridSearchCV(temp, param_grid=parameters, scoring='f1', cv=sss, refit=True)
# start_time = time.clock()
# gs_cv.fit(features, labels)
# print(f'Tempo execução: {time.clock() - start_time}s')
# print(f'Best precision: {gs_cv.best_score_}')
# print(f'Best params: {gs_cv.best_params_}')
# ==============================================================================
# Melhores parâmetros encontrados para ter um balanço entre precision e recall com tuning manual:
# ==============================================================================
# clf = AdaBoostClassifier(learning_rate=0.6, n_estimators=67)
# ==============================================================================
# Melhores parâmetros encontrados com a utilização de GridSearchCV e
# StratifiedShuffleSplit:
# ==============================================================================
clf = AdaBoostClassifier(learning_rate=0.9, n_estimators=88)
# Obs.: É importante notar que o tunning manual resultou em uma melhor pontuação no teste final, enquanto que o automático teve melhores resultados locais mas a custo de um grande tempo computacional.
# ==============================================================================
# Pontuação:
# ==============================================================================
# Accuracy score: 0.8520
# Precision score: 0.3728
# Recall score: 0.3350
# ------------------------------------------------------------------------------

# ==============================================================================
# Pega a média da Acuracia, Precisão e Validação para verificar se esta com
# uma boa precissão
# ==============================================================================
import time
time_list = []
accuracy_list = []
precision_list = []
recall_list = []
for train_index, test_index in sss.split(features, labels):

    feature_train, feature_test = features[train_index], features[test_index]
    labels_train, labels_test = labels[train_index], labels[test_index]

    # Mensurando tempo de execução do fit.
    start_time = time.clock()
    clf.fit(feature_train, labels_train)
    time_list.append(time.clock() - start_time)

    pred = clf.predict(feature_test)

    accuracy_list.append(accuracy_score(labels_test, pred))
    precision_list.append(precision_score(labels_test, pred))
    recall_list.append(recall_score(labels_test, pred))

print()
print('='*80)
print(str(clf))
print('='*80)
print('Scores and excution time means:')
print(f'Time mean: {np.array(time_list).mean():.4f}s')
print(f'Accuracy score: {np.array(accuracy_list).mean():.4f}')
print(f'Precision score: {np.array(precision_list).mean():.4f}')
print(f'Recall score: {np.array(recall_list).mean():.4f}')
print('-'*80)

# ==============================================================================
# Resultados:
# ==============================================================================
# Com test_size=0.2, melhor resultado:
# Split 5:
# Accuracy score: 0.8276
# Precision score: 0.4286
# Recall score: 0.750000
# ==============================================================================
# Rodando test.py
# ==============================================================================
# AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,learning_rate=0.6,
# n_estimators=67, random_state=None)
#   Accuracy: 0.84380
#   Precision: 0.39234
#   Recall: 0.31250
#   F1: 0.34790
#   F2: 0.32576
# ==============================================================================
#   Total predictions: 15000
#   True positives:  625
#   False positives:  968
#   False negatives: 1375
#   True negatives: 12032
# ------------------------------------------------------------------------------

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)