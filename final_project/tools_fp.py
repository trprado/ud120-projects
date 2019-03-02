""" Ferramentas para o projeto final de Fundamentos de Data Science II

Esse módulo contem funções que facilitam o trabalho de corrigir o dataset para se trabalhar com machine learning no projeto final do curso de FDSII.

TODO::
    * Adicionar mais opções de gráficos.
    * Melhorar label_pos, é necessário? Ou melhor tornar padrão o índice 0?
"""

import sys
import pickle
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.feature_selection import SelectKBest, f_classif

import matplotlib.pyplot as plt
import seaborn as sns

def select_best_features(dataset:Dict[str, Dict[str, float]], feature_list:List[str], label_pos:int=0, score_func:object=f_classif, nan_perc:float=0.40, alpha:float=0.05, plt_matrix:bool=False) -> Dict[str, float]:
    """Seleciona as melhores features com base em SelectKBest e retorna de acordo com porcentagem de nulos e p-valor.

    :param dataset: Dicionário contendo todos os valores do dataset.
    :type dataset: Dict[str, Dict[str, float]]
    :param feature_list: Lista contendo as features para serem selecionadas.
    :type feature_list: List[str]
    :param label_pos: Posição da feature que contém os labels de resposta, defaults to 0
    :type label_pos: int, optional
    :param score_func: Função utilizada pelo SelectKBest , defaults to f_classif
    :type score_func: object, optional
    :param nan_perc: Percentagem máxima devalores NaN permitidos, defaults to 0.40
    :type nan_perc: float, optional
    :param alpha: Valor máximo para seleção do p-valor, defaults to 0.05
    :type alpha: float, optional
    :param plt_matrix: Plota um gráfico único de dispersão entre as features, defaults to False
    :type plt_matrix: bool, optional
    :return: Dicionário contendo a chave o nome da feature selecionada e o valor o p-valor resultante.
    :rtype: Dict[str,float]
    """

    features = pd.DataFrame.from_dict(dataset, orient='index', dtype=None, columns=None)

    # Seleciona o label como lista
    labels = features.loc[:, feature_list[label_pos]].tolist()
    features.drop(columns=feature_list[label_pos], inplace=True)

    for col in features.columns:
        # Remove features que não estão em feature_list
        if col not in feature_list:
            features.drop(columns=col, inplace=True)
            continue

        features[col] = features[col].apply(
            lambda x: np.nan if x == 'NaN' else x)

        # Remove features com porcentagem de nulos maior que definida em
        # n_percent.
        n_percent = features[col].isna().sum() / features.shape[0]
        # print(col, n_percent)
        if n_percent > nan_perc:
            features.drop(columns=col, inplace=True)
            continue

    features.fillna(0, inplace=True)

    if plt_matrix:
        # Plot a scatter matrix of all features
        print(features.info())
        sns.pairplot(features)
        plt.tight_layout()
        fig.savefig('fig/best_features.png')
        plt.show()
    print()

    # Using SelectKBest to get score and p-values
    select = SelectKBest(score_func=score_func, k='all')
    select.fit(features.values, labels)

    print('='*80)
    print('Scores and p-values of all features:')
    print('='*80)
    print(f'Feature scores:\n {np.round(select.scores_)}')
    print(f'Feature p-values:\n {np.round(select.pvalues_, 4)}')
    print('-'*80)
    print()

    selected_features = {}
    for i, feature in enumerate(features.columns):
        if select.pvalues_[i] <= alpha:
            selected_features[feature] = round(select.pvalues_[i], 4)

    # print(list(selected_features.keys()))
    return selected_features


def remove_outliers(dataset:Dict[str, Dict[str, float]], feature_list:List[str], label_pos:int=0, plt_box=False) -> Dict[str, Dict[str, float]]:
    """Remove outliers com base no IRQ score.

    :param dataset: Dicionário contendo todos os valores do dataset.
    :type dataset: Dict[str, Dict[str, float]]
    :param feature_list: Lista com as features utilizadas para a remoção dos outliers.
    :type feature_list: List[str]
    :param label_pos: Posição da feature que contém os labels de resposta, defaults to 0
    :type label_pos: int, optional
    :param plt_box: Flag para definir se ira imprimir o boxplot das features, defaults to False
    :type plt_box: bool, optional
    :return: Dicionário contendo o dataset modificado com a remoção de outliers.
    :rtype: Dict[str, Dict[str, float]]
    """

    features = pd.DataFrame().from_dict(dataset, orient='index', dtype=None, columns=None)

    for col in features.columns:
        if col not in feature_list:
            features.drop(columns=col, inplace=True)
            continue

        features[col] = features[col].apply(
            lambda x: 0 if x == 'NaN' else x)

    # Calcula o IRQ score.
    Q1 = features[feature_list[1:]].quantile(.25)
    Q3 = features[feature_list[1:]].quantile(.75)
    IQR = Q3 - Q1

    # Remove as linhas com base no IQR score.
    features = features[~((features < (Q1-1.5*IQR)) | (features > (Q3+1.5*IQR))).any(axis=1)]

    if plt_box:
        fig, ax = plt.subplots(figsize=(10,10))
        g = sns.boxplot(data=features[feature_list[1:]], ax=ax)
        sns.despine(offset=10, trim=True)
        plt.setp(g.get_xticklabels(), rotation=45, ha="right")
        plt.tight_layout()
        fig.savefig('fig/outliers.png')
        plt.show()

    return features.to_dict('index')


def add_feature(dataset:Dict[str, Dict[str, float]], feature_list:List[str], select_features:List[str], label_pos:int=0) -> Tuple[Dict[str, Dict[str, float]], List[str]]:
    """ Adiciona novas features ao dataset com base em uma lista de features.

    As novas features são [1, x, x^2, x * y, y^2] com base na combinação da lista de features.

    :param dataset: Dicionário contendo todos os valores do dataset.
    :type dataset: Dict[str, Dict[str, float]]
    :param feature_list: Lista com as features usadas.
    :type feature_list: List[str]
    :param select_features: Features que vão gerar as novas features polinomiais.
    :type select_features: List[str]
    :param label_pos: Posição da feature que contém os labels de resposta, defaults to 0
    :type label_pos: int, optional
    :return: Dataset contendo as novas features polinomiais e a lista de features atualizada.
    :rtype: Tuple[Dict[str, Dict[str, float]], List[str]]
    """


    features = pd.DataFrame().from_dict(dataset, orient='index')


    features.fillna(0)
    for col in features.columns:
        if col not in feature_list:
            features.drop(columns=col, inplace=True)
            continue

        if col != feature_list[label_pos]:
            features[col] = features[col].apply(lambda x: 0 if x == 'NaN' else x)

    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(2)
    poly.fit(features[select_features])

    # Gera um novo dataframe com base na transformação polinomial
    new_features = pd.DataFrame(
        poly.transform(features[select_features]),
        columns=poly.get_feature_names(features[select_features].columns))
    # Configura o index para o mesmo do features
    new_features.set_index(features.index, inplace=True)
    # Remove as features que já existem no dataframe features
    new_features.drop(columns=select_features, inplace=True)
    new_features.drop(columns='1', inplace=True)

    feature_list = feature_list + new_features.columns.tolist()
    # Junta ambos os dataframes em um novo dataframe.
    features = features.join(new_features)

    return features.to_dict('index'), feature_list
