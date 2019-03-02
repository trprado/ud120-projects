# Enron Submission Report

## 1.
O objetivo é identificar pessoas de interesse (**POIs**), discriminando quais são **POIs** e quais não são **POIs**. Ao se aplicar métodos de *Machine Learning* é possível, ao analisar os dados, modelar (treinamento) e validar (validação) o modelo formulado para predizer se futuras observações podem ser atribuídas ou não a categoria de POIs. Por exemplo, a variável X é uma variável de importância visto que **POIs** possuem maior valores monetários do que pessoas não **POIs**, assim, uma futura observação que também possui esse valor monetário em destaque tem maior chance de ser um candidato a ser um **POI**.

Foram observados outliers nos dados *TOTAL*, com valores que na realidade são as somas de todos os demais valores do dataset; *THE TRAVEL AGENCY IN THE PARK*, uma agência de viagens que não tem relação direta com a *Enron*; *LOCKHART EUGENE E* uma pessoa onde todos os valores são nulos, e eles foram tratados com a remoção dos seus dados do *dataset*. Valores que o *IQR* score considerava como *outliers* não foram removidos, pois se tratavam de **POIs** com grande valor em seus atributos, outros são investidores que não chegam a ter alguma relação com **POIs** mas devido a não ter uma relação direta com a *Enron* seus dados como (salário, *emails* para **POIs**, etc) tinham valores nulos e foram tratados como zero.

## 2.
As variáveis explicativas (*features*) utilizadas foram: `salary`, `total_payments`, `bonus`, `total_stock_value`, `expenses`, `from_poi_to_this_person`, `exercised_stock_options`, `other`, `shared_receipt_with_poi`, `restricted_stock`, `salary^2`, `salary total_payments`, `total_payments^2` as quais foram identificadas utilizando o algoritmo *SelectKBest* devido sua forma de pegar pontuações por `f1_score` e retornar também o `p-value`, assim fica possível definir estatisticamente a importância de cada *feature*, removendo as *features* que apresentaram a contagem de valores nulos maior que 50% do seu tamanho total e selecionando aquelas com p-valor menor ou igual ao nível de significância de 5% (*alpha*). Para tal, foi criada uma função com o intuito de facilitar o uso do *SelectKBest*, com o objetivo de determinar a porcentagem de valores nulos aceitos e o nível de significância (*alpha*).

### Pontuação das *features* antes da seleção
```
================================================================================
Scores and p-values of all features:
================================================================================
Feature scores:
 [19.  2.  9.  21.  25.  6.  5.  25.  0.  4.  2.  9.  9.]
Feature p-values:
 [0.  0.1878  0.0032  0.  0.  0.0127  0.021  0.  0.6909  0.0407  0.1182  0.0033  0.0025]
--------------------------------------------------------------------------------
================================================================================
Feature List:
['poi', 'salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees', 'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']
================================================================================
```

Novas *features* foram criadas utilizando do algoritmo *PolynomialFeatures*, com ele foi passado duas *features* existentes, e ele gerou quatro novas *features* sendo elas `1`, `x^2`, `x*y` e `y^2`, a *feature* `'1'` foi removida por não gerar um `f1_score` ou `p-value`. Também foi realizado o escalonamento dos valores com *MinMaxScale*, esse escalonamento foi necessário para que alguns algoritmos de *Machine Learning* trabalhassem corretamente com as *features*.

### Pontuação das *features* depois da seleção com novas *features* adicionadas
```
================================================================================
SelectKBest in selected features:
================================================================================
Best scores: [18.2897  8.7728 20.7923 24.1829  6.0942  5.2434 24.8151  4.1875  8.5894  9.2128]
P-values: [0.     0.0036 0.     0.     0.0148 0.0235 0.     0.0426 0.0039 0.0029]
--------------------------------------------------------------------------------
================================================================================
Final feature list:
['poi', 'salary', 'total_payments', 'bonus', 'total_stock_value', 'expenses', 'from_poi_to_this_person', 'exercised_stock_options', 'other', 'shared_receipt_with_poi', 'restricted_stock', 'salary^2', 'salary total_payments', 'total_payments^2']
Count features: 14
================================================================================
```

## 3.
Os algoritmos para a modelagem usados foram: *SVC*, *GaussianNB*, *DecisionTreeClassifier*, *KNeighborsClassifier*, *RandomForestClassifier*, *AdaBoostClassifier*, pois são algoritmos de classificação mais adequados para analisar a variável resposta `poi`. O tempo médio de execução e o `f1_score` foram utilizados para selecionar o melhor algoritmo utilizando validação cruzada, garantindo assim um melhor desempenho. De todos os algoritmos testados apenas três foram selecionados: *SVC*, *RandomForestClassifier* e *AdaBoostClassifier* pois resultaram nas melhores pontuação em `f1` por um baixo tempo de processamento.

O *SVC* foi o melhor em pontuação e tempo, porém ao fazer testes com a *precision* e *recall* ele não obteve um desempenho desejável observado na *precision* e na *recall*. O mesmo ocorreu com *RandomForestClassifier*. Optou-se em utilizar o *AdaBoostClassifier* que obteve os melhores resultados nas pontuações da acurácia, *precision* e 8. Também foram feitas novas validações cruzadas nesses algoritmos, utilizando um maior número de parâmetros, o que aumentou o tempo computacional para validar, também foi utilizado as opções de usar mais núcleos de processamento, mas devido a grande quantidade de zeros presentes nos dados apareciam muitos avisos em tela atrapalhando a visualização dos resultados.

### Resultados da seleção do melhor classificador.
```
================================================================================
SVC
================================================================================
Fit time mean: 0.0027
F1 score: 0.3022
Best parameters: {'svc__C': 20, 'svc__kernel': 'sigmoid'}
--------------------------------------------------------------------------------

================================================================================
GaussianNB
================================================================================
Fit time mean: 0.0027
F1 score: 0.1810
Best parameters: {}
--------------------------------------------------------------------------------

================================================================================
DecisionTreeClassifier
================================================================================
Fit time mean: 0.0028
F1 score: 0.2444
Best parameters: {'dtc__criterion': 'gini', 'dtc__max_depth': 10, 'dtc__min_impurity_decrease': 0.0, 'dtc__min_samples_leaf': 1, 'dtc__min_samples_split': 2}
--------------------------------------------------------------------------------

================================================================================
KNeighborsClassifier
================================================================================
Fit time mean: 0.0023
F1 score: 0.0583
Best parameters: {'knc__leaf_size': 5, 'knc__n_neighbors': 3, 'knc__p': 4}
--------------------------------------------------------------------------------

================================================================================
RandomForestClassifier
================================================================================
Fit time mean: 0.0251
F1 score: 0.2507
Best parameters: {'rfc__max_depth': 5, 'rfc__min_samples_leaf': 2, 'rfc__min_samples_split': 5}
--------------------------------------------------------------------------------

================================================================================
AdaBoostClassifier
================================================================================
Fit time mean: 0.0245
F1 score: 0.2493
Best parameters: {'ada__learning_rate': 0.5, 'ada__n_estimators': 10}
--------------------------------------------------------------------------------
```

## 4.
Usar o *tuning* no algoritmo escolhido possibilitou obter melhor ajuste nas pontuações da acurácia, *precision* e *recall*, consequentemente adquiriu-se resultados mais promissores. Caso não seja realizado um *tuning*, o algoritmo pode não trazer os melhores resultados, incluindo gerar maiores números de erros do tipo 1 e 2. O *tuning* utilizado no algoritmo escolhido (*AdataBoostClassfier*) foi feito utilizando *StratifiedShuffleSplit* e *GridSearchCV*, aumentando também o número de possibilidades de parâmetros e repetições para garantir um melhor resultado. Uma consequência indesejável desse processo foi que o aumento no tempo de processamento (em horas) até selecionar um grupo de parâmetros que melhor equilibre as pontuações.

Em caso de um algoritmo que não possui a necessidade de ajustar seus parâmetros, o *tuning* deveria ser feito nas *features*, selecionando e testando entre aquelas que melhor trazem resultados ao algoritmo. Assim, mesmo sem uma mudança de parâmetros poderíamos chegar a uma conclusão melhor ao usar *features* diferentes.

## 5.
A validação serve para avaliar o quanto o modelo prediz bem a sua variável resposta predita a qual sera correlacionada com a variável resposta verdadeira que foi ocultada no processo de validação. Em resposta obtêm-se a acurácia entre esses valores, quanto maior for a acurácia melhor sera avaliado o modelo. O problema clássico da validação é garantir a representatividade dos dados, tanto na validação quanto no treinamento, isso é, caso exista algum padrão nos dados, a divisão entre treinamento e teste pode render uma baixa acurácia pois por exemplo os dados de uma certa pessoa podem estar no início e de outra no final. Ao tornar os dados de validação aleatórios, garante essa representatividade, assim não usando dados que podem estar seguindo um padrão.

Para validar minha análise utilizei StratifiedShuffleSplit, assim valores seriam pegos de forma aleatória dentro do conjunto de dados, garantindo uma maior acurácia entre o grupo de treinamento e o teste realizado.

## 6.
As métricas *precision* é a capacidade de não classificar um rótulo como positivo em uma amostra que é negativa, a pontuação ser dada por $tp/(tp+fp)$, onde $tp$ é a contagem de verdadeiros positivos e $fp$ a contagem de falsos positivos. Já *recall* é uma pontuação que mostra a capacidade do classificador de encontrar todas as amostras positivas, ele é dado pela fórmula $tp/(tp+fn)$ onde $tp$ é a contagem de verdadeiros positivos e $fn$ a contagem de falsos negativos. Quanto maiores forem os valores de de *recall* e *precision* melhor o modelo é avaliado, porém esses valores estão sujeitos a qualidade dos dados, ou seja, um valor "baixo" de *recall* e *precision* não quer dizer necessariamente que o modelo não prediz bem os dados de teste. Por isso, esses valores com a acurácia devem ser avaliados conjuntamente para avaliar o modelo como um todo.

Os desempenhos médios resultantes do algoritmo *AdaBoostClassifier* com os melhores parâmetros selecionados foram de $0.38846$ em *precision* e $0.30300$ em *recall*, com 1000 divisões na validação cruzada.