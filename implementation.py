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
    gain = information_gain(train, [ale_subset, lager_subset, stout_subset])
    print(gain)
    #tree = build_tree(train_data, train_target)
    #visualise_tree(tree)
    #test_tree(tree, test_data, test_target)


# Aideen
def split_data_styles(data):
    grouped = data.groupby(data['style'])
    ale_subset = grouped.get_group('ale')
    lager_subset = grouped.get_group('lager')
    stout_subset = grouped.get_group('stout')
    
    return ale_subset, lager_subset, stout_subset

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

# # Come back to 
# def build_tree(data):
#     return tree

# # Aideen
def entropy(training_data):
    ale_subset, lager_subset, stout_subset = split_data_styles(training_data)
    ale = len(ale_subset)/ len(training_data)
    lager = len(lager_subset)/ len(training_data)
    stout = len(stout_subset)/ len(training_data)
    entropy = -((ale)*log2(ale) + (lager)*log2(lager) + (stout)*log2(stout) ) # shouldn't these all be minus? - I don't think so
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