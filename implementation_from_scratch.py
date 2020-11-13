# Main file for the implementation of the C4.5 algorithm
import pandas as pd
import numpy as np
from graphviz import Graph
from math import log2

def main():
    data = pd.read_csv('beer.csv')
    train, test = split_data_training_testing(data, (2/3))
    ale_subset, lager_subset, stout_subset = split_data_styles(train)
    
    #train_data, train_target, test_data, test_target = split_into_data_target(train, test)
    #need to create subset of data from train_target for the ale_subset, lager_subset, stout_subset()
    
    subsets = split_into_subsets('calorific_value', train)
    print(subsets)
    gain = information_gain(train, subsets)
    print(gain)
    
    #gain = information_gain(train, [ale_subset, lager_subset, stout_subset])
    #tree = build_tree(train_data, train_target)
    #visualise_tree(tree)
    #test_tree(tree, test_data, test_target)


# # Come back to 
# def build_tree(data):
#     for column in data
#       subsets = split_into_subsets(column)
#       gain = information_gain(train, subsets)
#   
#     return tree

def split_into_subsets(column_header, training_data):
    split_values = []
    maxEnt = -1*float("inf")
    sorted_data = training_data.sort_values(by=[column_header])
    for item in range(0, len(training_data) - 1):
        if sorted_data.iloc[item][column_header] != sorted_data.iloc[item+1][column_header]:
            threshold = (sorted_data.iloc[item][column_header] + sorted_data.iloc[item+1][column_header]) / 2
            less = pd.DataFrame()
            greater = pd.DataFrame()
            for index, row in sorted_data.iterrows():
                if(row[column_header] > threshold):
                    greater = greater.append(row, ignore_index = True)
                else:
                    less = less.append(row, ignore_index = True)

            igain = information_gain(training_data, [less, greater])

            if igain >= maxEnt:
                split_values = [less, greater]
                maxEnt = igain
    return split_values

# Aideen
def split_data_styles(data):
    subsets = {}
    grouped = data.groupby(data['style'])
    for index, beer_style in enumerate(['ale','lager','stout']):
        if beer_style in grouped.groups.keys():
            subsets[index] = grouped.get_group(beer_style)
        else:
            subsets[index] = {}
    return subsets

# Aideen 
def split_data_training_testing(data, ratio):

    division_point = round(len(data)*ratio)
    headers = data.iloc[0]
    
    data = data.drop(columns=['beer_id'])
    
    train = data.sample(frac=ratio,random_state=5)
    test = data.sample(frac=(1-ratio),random_state=5)
    return train, test

def split_into_data_target(train, test):
    train_data = train.drop(columns=['style'])
    train_target = train['style'].values
    test_data = test.drop(columns=['style'])
    test_target = test['style'].values
    return train_data, train_target, test_data, test_target

# Aideen 
def read_csv_data(csv_path):
    data = pd.read_csv(csv_path)
    return data

# def data_class_check(data):
#     return result

# # Aideen
def entropy(training_data):
    entropy = 0
    subsets = split_data_styles(training_data)
    for index in range(len(subsets)):
        probability = len(subsets[index])/ len(training_data)
        if probability != 0:
            entropy = entropy - (probability)*log2(probability)
    return entropy

# # Louise
def information_gain(train_target, subsets):
    # Gain calculation for this function following the lecture notes.
    #Gain = Ent(S) - |S beer =ale |/|S|*( Ent( S beer=ale )) -  |S beer=stout |/|S|*( Ent( S beer=stout )) - |S beer=lager |/|S|*( Ent( S beer=lager )) - ....
    entropyTarget = entropy(train_target)
    total = len(train_target)

    Gain = entropyTarget

    for i in range(0, len(subsets)):
        numBeer = len(subsets[i])
        firstPart = numBeer/total
        secondPart = entropy(subsets[i])
        Gain -= (firstPart*secondPart)

    return Gain

# # Louise
def visualise_tree(tree):
    dot_data = tree.export_graphviz(clf, out_file=None, 
                                feature_names=iris.feature_names,  
                                class_names=iris.target_names,
                                filled=True)
    graph = graphviz.Source(dot_data, format="png") 
   
# # Come back to 
# def test_tree(tree, testing_data):
#     # test tree
    
main()