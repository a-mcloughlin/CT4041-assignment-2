# Main file for the implementation of the C4.5 algorithm
import pandas as pd
import numpy as np
from math import log2

def main():
    data = pd.read_csv('beer.csv')
    train_data, train_target, test_data, test_target = split_data(data, (2/3))
    # entropy(train_target)
    #tree = build_tree(train_data, train_target)
    #visualise_tree(tree)
    #test_tree(tree, test_data, test_target)

# Aideen 
def split_data(data, ratio):

    division_point = round(len(data)*ratio)
    headers = data.iloc[0]
    
    data = data.drop(columns=['beer_id'])
    
    train = data.sample(frac=ratio,random_state=5)
    test = data.sample(frac=(1-ratio),random_state=5)
    
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
def entropy(train_target):

    ale = np.count_nonzero(train_target == 'ale')/ len(train_target)
    lager = np.count_nonzero(train_target == 'lager')/ len(train_target)
    stout = np.count_nonzero(train_target == 'stout')/ len(train_target)
    entropy = -((ale)*log2(ale) + (lager)*log2(lager) + (stout)*log2(stout) )
    return entropy

# # Louise
# def information_gain(leaf):
#     return information_gain

# # Louise
# def visualise_tree(tree):
#     # visualaise
   
# # Come back to 
# def test_tree(tree, testing_data):
#     # test tree
    
main()