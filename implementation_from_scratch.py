# Main file for the implementation of the C4.5 algorithm
import pandas as pd
import numpy as np
from graphviz import Graph
from math import log2
from copy import deepcopy
from weka_implementation import build_weka_tree
import PySimpleGUI as sg
from PIL import Image, ImageTk, ImageSequence
from multiprocessing import Process, Queue
import multiprocessing as mp
import time

def main():
    file = 'beer.csv'
    data = pd.read_csv(file)
    split = GetSplit()
    quit = mp.Event()
    Q = Queue()
    p1 = Process(target = createTree, args=(data, split, quit, Q,))
    p2 = Process(target = renderLoadingWindow, args=(quit, ))
    starttime = time.time()
    p1.start()
    p2.start()
    quit.wait()
    endtime = time.time()
    queue_data = Q.get()
    root_node = queue_data[0]
    testing_data = queue_data[1]
    print_node_data(root_node, "")
    #visualise_tree(tree)
    
    accuracy = test_tree(root_node, testing_data)
    DisplayTree(accuracy, round(endtime-starttime))
    #build_weka_tree(file, split)

def createTree(data, data_split, quit, queue):
    train_data, test_data = split_data_training_testing(data, (data_split))
    
    attributes = ['calorific_value','nitrogen','turbidity','style','alcohol','sugars','bitterness','colour','degree_of_fermentation']
    root_node = build_tree(train_data, attributes)
    queue.put([root_node, test_data])
    queue.cancel_join_thread()
    quit.set()
    return True

def renderLoadingWindow(quit):
    
    layout_loading = [[sg.Text("Loading")],[sg.Image(r'loading.gif', key='-IMAGE-')]]
    
    # Create the window
    gif_filename = r'loading.gif'
    window = sg.Window("Select Train/Test Split", layout_loading, element_justification='c', margins=(0,0), element_padding=(0,0), finalize=True)
    interframe_duration = Image.open(gif_filename).info['duration']
    while not quit.is_set():
        event, values = window.read(timeout=interframe_duration)
        if event == sg.WIN_CLOSED:
            exit()
            break
        window.FindElement("-IMAGE-").UpdateAnimation("loading.gif",time_between_frames=interframe_duration)
    
    window.close()
    print("Stop Animating")
    return True
    
def GetSplit():
    layout = [[sg.Text("Select Train/Test Split")],
        [sg.Radio('1/3',"1", key="1/3"), 
           sg.Radio('1/2',"1", key="1/2"), 
           sg.Radio('2/3',"1", key="2/3", default=True)],
         [sg.Button('Ok'), sg.Button('Quit')]]
        
    
    # Create the window
    gif_filename = r'loading.gif'
    window = sg.Window("Select Train/Test Split", layout)
    while True:
            event, values = window.read()
            if event == sg.WIN_CLOSED or event == 'Quit': # if user closes window or clicks cancel
                break
            if event == 'Ok':
                if values["1/3"] == True:
                    split = (1/3)
                elif values["1/2"] == True:
                    split = (1/2)
                elif values["2/3"] == True:
                    split = (2/3)
                break
                
    window.close()
    return split

def DisplayTree(accuracy, time_to_build):
    layout = [[sg.Image(r'tree.png', size=(200,200),key='-IMAGE-')],
        [sg.Text("Accuracy: "+str(accuracy*100)+"%")],
        [sg.Text("The Tree took "+str(time_to_build)+" seconds to build")],
        [sg.Button('Quit')]]
    
    window = sg.Window("Generated C4.5 Tree", layout)
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event == 'Quit': # if user closes window or clicks cancel
            break
                
    window.close()

def test_data(data, node, test_results):
    for item in range(0, len(data)):
        test_results.append(test_lr(node, data.iloc[item]))
    return test_results
            
def test_lr(node, row):
    if node.isLeaf:
        return node.label
    else:
        if row[node.label] <= node.divisor:
            return test_lr(node.children[0], row)
        else:
            return test_lr(node.children[1], row)
        
def print_node_data(node, indent):
    print("-------------")
    print(indent+"Label:    "+str(node.label))
    print(indent+"Is Leaf:  "+str(node.isLeaf))
    print(indent+"Threshold Value:  x > "+str(node.divisor))
    if node.children != []:
        print(indent+"Children: ")
        for child in node.children:
            print_node_data(child, indent+"\t")
    else:
        print(indent+"No Children")
# 
#
def build_tree(data, attributes):
    #1. check above base cases
        #•  All the examples from the training set belong to the same class ( a tree leaf labeled with that class is returned ).
    # data_class_checked = data_class_check(data)
    #•  The training set is empty ( returns a tree leaf called failure ).
    if len(data) == 0:
	    return Node(True, "Fail", None)
    # elif data_class_checked is not False:		
	#     return Node(True, data_class_checked, None)

    #2. find attribute with highest info gain, retrun best_attribute - done
    best_attribute, attribute_subsets, threshold_divisor  = find_best_attribute(data, attributes)
    
    if best_attribute == "":
        majClass = getMajorityClass(data)
        return Node(True, majClass, None)
    else:
        #3. split the set (data) in subsets arrcording to value of best_attribute
        #attribute_subsets = split_into_subsets(best_attribute, data)
        remainColumns = deepcopy(data)
        remainColumns = data.drop(columns=[best_attribute])

        #4. repeat steps for each subset 
        node = Node(False, best_attribute, threshold_divisor)
        for attr_subset in attribute_subsets:
            node.children.append(build_tree(attr_subset, remainColumns))
            
        return node
  

# # Louise
def find_best_attribute(train_data, attributes):
    #  Returns the best attribute from all
    best_information_gain = 0
    best_attribute = ""
    threshold_divisor = ""
    subsets = []
    for attribute in attributes:
        if attribute != 'style':
            temp_subsets, temp_divisor = split_into_subsets(attribute, train_data)
            temp_gain = information_gain(train_data, temp_subsets)
            if temp_gain > best_information_gain:
                best_attribute = attribute
                best_information_gain = temp_gain
                subsets = temp_subsets
                threshold_divisor = temp_divisor
    
    return best_attribute, subsets, threshold_divisor
    

def getMajorityClass(data):
    grouped = data.groupby(data['style'])
    return max(grouped.groups)


def split_into_subsets(column_header, training_data):
    split_values = []
    maxEnt = -1*float("inf")
    best_threshold = ""
    sorted_data = training_data.sort_values(by=[column_header])
    for item in range(0, len(training_data) - 1):
        if type(sorted_data.iloc[item][column_header]) != 'style':
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
                    best_threshold = threshold
                    maxEnt = igain
    return split_values, best_threshold

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
        if row.iloc[-1] != data.iloc[0].iloc[-1]:
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
   
# Aideen
def test_tree(root_node, testing_data):
    test_target = testing_data['style'].values
    testing_data = testing_data.drop(columns=['style'])
    test_results = test_data(testing_data, root_node, [])
    correct = 0
    for index in range(0, len(test_results)):
        if test_results[index] == test_target[index]:
            correct = correct +1
        else:
            print(str(test_target[index]) +" incorrectly categorised as "+str(test_results[index]))
    accuracy = correct/len(test_results)
    print("Accuracy: "+str(accuracy))
    return accuracy

class Node:
    def __init__(self,isLeaf, label, divisor):
        self.label = label
        self.isLeaf = isLeaf
        self.divisor = divisor
        self.children = []
    
if __name__ == '__main__':
    main()
