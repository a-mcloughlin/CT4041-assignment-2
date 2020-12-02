import pandas as pd
import numpy as np
from implementation_from_scratch import *
from weka_implementation import *
import weka.core.jvm as jvm
# Define a range of train/test split proportions
proportions = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]

jvm.start()

# For each proportion
for split in proportions:
    # Read in the data 
    data = pd.read_csv('beer.csv')
    
    # declare Queue for Tree output storage
    Q = Queue()
    # Create tree
    createTree(data, split, mp.Event(), Q)
    
    # Get the tree output
    queue_data = Q.get()
    root_node = queue_data[0]
    test_data = queue_data[1]
    
    # Get the accuracy of the tree and test it
    python_accuracy = test_tree(root_node, test_data, split)
    
    # Build and test the weka tree
    weka_accuracy, weka_time_to_build = build_weka(split)
    
jvm.stop()