# Main file for the implementation of the C4.5 algorithm
import pandas as pd
import numpy as np
from graphviz import Graph
from math import log2
from copy import deepcopy

def main():
    data = pd.read_csv('beer.csv')
    train, test = split_data_training_testing(data, (2/3))
    ale_subset, lager_subset, stout_subset = split_data_styles(train)
    
    #train_data, train_target, test_data, test_target = split_into_data_target(train, test)
    #need to create subset of data from train_target for the ale_subset, lager_subset, stout_subset()
    
    attributes = ['calorific_value','nitrogen','turbidity','alcohol','sugars','bitterness','colour','degree_of_fermentation']
    # for attribute in attributes:
    #     subsets = split_into_subsets(attribute, train)
    #     gain = information_gain(train, subsets)
    #     print("Attribute: " + attribute)
    #     print("Lesser Length: "+ str(len(subsets[0])))
    #     print("Greater Length: " + str(len(subsets[1])))
    #     print("Information Gain:" + str(gain))
    #     print("------------------------------------------")
    
    root_node = build_tree(train, attributes)
    #gain = information_gain(train, [ale_subset, lager_subset, stout_subset])    
    #visualise_tree(tree)
    #test_tree(tree, test_data, test_target)


# 
#
def build_tree(data, attributes):
    #1. check above base cases
        #•  All the examples from the training set belong to the same class ( a tree leaf labeled with that class is returned ).
    data_class_checked = data_class_check(data)
  
    #•  The training set is empty ( returns a tree leaf called failure ).
    if len(data) == 0:
	    return Node(True, "Fail")
    elif data_class_checked is not False:		
	    return Node(True, data_class_checked)

    #  The attribute list is empty ( returns a leaf with the majority class).
    elif len(attributes) == 0:
        #return a node with the majority class
        majClass = getMajorityClass(data, ['ale','lager','stout'])
        return Node(True, majClass)
    else:
        #2. find attribute with highest info gain, retrun best_attribute - done
        best_attribute = find_best_attribute(data, attributes)

        #3. split the set (data) in subsets arrcording to value of best_attribute
        attribute_subsets = split_into_subsets(best_attribute, data)
        remainColumns = deepcopy(data)
        remainColumns.drop(columns=[best_attribute])

        #4. repeat steps for each subset 
        node = Node(False, best_attribute)
        for attr_subset in attribute_subsets:
            node.children.append(build_tree(attr_subset, remainColumns))
        return node


#     for column in data
#       subsets = split_into_subsets(column)
#       gain = information_gain(train, subsets)
#   

# # Louise
def find_best_attribute(train_data, attributes):
    #  Returns the best attribute from all
    best_information_gain = 0
    best_attribute = ""
    for attribute in attributes:
        temp_gain = information_gain(train_data, split_into_subsets(attribute, train_data))
        if temp_gain > best_information_gain:
            best_attribute = attribute
            best_information_gain = temp_gain
    
    return best_attribute
    

def getMajorityClass(data, labels):
    occurrence = [0]*len(labels) # create a zeroed array of length labels ['ale','lager','stout']
    for row in data: 
        i = labels.index(row[3]) # style is the fourth column
        occurrence[i] += 1

    return labels[occurrence.index(max(occurrence))]  	


def split_into_subsets(column_header, training_data):
    split_values = []
    maxEnt = -1*float("inf")
    sorted_data = training_data.sort_values(by=[column_header])
    for item in range(0, len(training_data) - 1):
        if sorted_data.iloc[item][column_header] != sorted_data.iloc[item+1][column_header]:
            threshold = (sorted_data.iloc[item][column_header] + sorted_data.iloc[item+1][column_header]) / 2
            smaller_than_threshold = pd.DataFrame()
            bigger_than_threshold = pd.DataFrame()
            for index, row in sorted_data.iterrows():
                if(row[column_header] > threshold):
                    bigger_than_threshold = bigger_than_threshold.append(row, ignore_index = True)
                else:
                    smaller_than_threshold = smaller_than_threshold.append(row, ignore_index = True)

            igain = information_gain(training_data, [smaller_than_threshold, bigger_than_threshold])

            if igain >= maxEnt:
                split_values = [smaller_than_threshold, bigger_than_threshold]
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

def data_class_check(data):
    for index, row in data.iterrows():
        a = row.iloc[-1]
        b = data.iloc[0].iloc[-1]
        if a != b:
            return False
    return data.iloc[0].iloc[-1]


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

class Node:
    def __init__(self,isLeaf, label):
        self.label = label
        self.isLeaf = isLeaf
        self.children = []
    
main()
