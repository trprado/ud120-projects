#!/usr/bin/python

"""
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000

"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "rb"))

# Quizz 1 - How many people has in dataset?
print(f'Number of people: {len(enron_data)}')

# Quizz 2 - For each people, how many attributes are avaliable?
print(f'Number of attributes: {len(enron_data["METTS MARK"])}')

# Quizz 3 - How many POIs are in the dataset?
count = 0
for name in enron_data.keys():
    if enron_data[name]['poi'] == 1:
        count += 1
print(f'POIs: {count}')

# Quizz 4 - How many POIs where there total in final_project/poi_names.txt?
file = open('../final_project/poi_names.txt', 'r')
poi_names = file.readlines()
poi_names = poi_names[2:]
print(f'POI names count: {len(poi_names)}')

# Quizz 5 - What is the total value of the stock belonging to James Prentice?
total_stock = enron_data['PRENTICE JAMES']['total_stock_value']

print(f'Total stock: {total_stock}')

# Quizz 6 - How many messages do we have from Wesley Colwell to POIs?
total_poi = enron_data['COLWELL WESLEY']['from_this_person_to_poi']
print(f'Total POIs: {total_poi}')

# Quizz 7 - What's the value of stock options exercised by Jeff Skilling?
opt_stock_value = enron_data['SKILLING JEFFREY K']['exercised_stock_options']
print(f'Valor das opções de ações: {opt_stock_value}')

# Quizz 8 -
print('1, 2, 5 are corrects')

# Quizz 9 - Who was Enron's CEO during much of the time fraud was ongoing?
print('Jeffrey Skilling')

# Quizz 10 - Who was chairman of the Enron board of directors during much of the time that fraud was ongong?
print('Kenneth Lay')

# Quizz 11 - Who was chief financial officer (CFO) during much of the time that graud was ongoing?
print('Andrew Fastow')

# Quizz 12 - Of Lay, Skilling and Fastow, who took home the most money? How much was it?

names = ['Lay Kenneth L', 'Skilling Jeffrey K', 'Fastow Andrew S']
most_money_took = []

for name in names:
    most_money_took.append((name.upper(), enron_data[name.upper()]['total_payments']))

import pprint
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(most_money_took)

# Quizz 13 - How is an unfilled feature denoted?
# Answer: NaN


# Quizz 14 - How many folks in this dataset have a quantified salary? Known email address?
# count_email = 0
# count_salary = 0
# for name in enron_data.keys():
#     email = enron_data[name]['email_address']
#     if email != 'NaN':
#         count_email += 1
#     salary = enron_data[name]['salary']
#     if salary != 'NaN':
#         count_salary += 1

# print(f'Total know emails: {count_email}')
# print(f'Totak know salaries: {count_salary}')

# import pandas as pd
# import numpy as np

# df = pd.DataFrame.from_dict(enron_data, orient='index')
# print(df.head())

# print(df.shape)
# print(df.columns)
# df.email_address = df.email_address.apply(lambda x: np.nan if x == 'NaN' else x)
# df.dropna(inplace=True)
# print(df.email_address.shape)
# df.salary = df.salary.apply(lambda x: np.nan if x == 'NaN' else x)
# df.dropna(inplace=True)
# print(df.salary.shape)

# Quizz 15 - What percentage of people in the dataset have 'NAN' for their total payments?
total_payments_nan = 0
for name in enron_data:
    if enron_data[name]['total_payments'] == 'NaN':
        total_payments_nan += 1

print(f'Total payments equal NaN: {total_payments_nan}')
print(f'Percent of total payments equal NaN: {total_payments_nan/len(enron_data):.2%}')

# Quizz 16 - What percentage of POIs in the dataset have 'NaN' for their total payments?
poi_nan = 0
for name in enron_data:
    if enron_data[name]['poi'] == 'NaN':
        poi_nan += 1

print(f'Total POIs equal NaN: {poi_nan}')
print(f'Percent of POIs equal NaN: {poi_nan/len(enron_data):.2%}')

# Quizz 17 - If a machine learning algorithm were to use total_payments as a feature, would you experct it to associate a 'NaN'value with POIs or non-POIs?
# Answer: non-POIs

# Quizz 18 - What is the new number of people in the dataset? What is the new number of folks with 'NaN' for total_payments?
total_payments_nan = 0
for name in enron_data:
    if enron_data[name]['total_payments'] == 'NaN':
        total_payments_nan += 1

print(f'Total payments equal NaN: {total_payments_nan+10}')
print(f'Percent of total payments equal NaN: {(total_payments_nan+10)/(len(enron_data)+10):.2%}')

# Quizz 19 - What is the new number of POIs in the dataset? What is the new number of POIS with 'NaN' for total_payments?
# Answer: 18 + 10 = 28
# Anseer: 10

pp.pprint(enron_data['COLWELL WESLEY'].keys())
# pp.pprint(enron_data['Skilling Jeffrey K'.upper()])