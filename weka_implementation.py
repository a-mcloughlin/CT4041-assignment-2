from weka.classifiers import Classifier
from weka.core.converters import Loader
import weka.core.jvm as jvm
from weka.core.dataset import create_instances_from_lists
from weka.filters import Filter
from weka.core.classes import Random
from weka.classifiers import Classifier, Evaluation, PredictionOutput
import weka.plot.graph as graph
import graphviz
import time

# Aideen McLoughlin - 17346123
# Taking in the data file location, and the train/test split proportion
# Build a weka C4.5 implementation using the Python Weka Wrapper API
def build_weka_tree(data_file, data_split):
    jvm.start()
    
    # Load the data file
    loader = Loader(classname="weka.core.converters.CSVLoader")
    data = loader.load_file(data_file)
    
    # Set the class to be column 3 - the style column
    data.class_index = 3
    
    # Remove the beer_id column from the data as it is not relevant
    remove = Filter(classname="weka.filters.unsupervised.attribute.Remove", options=["-R", "8"])
    remove.inputformat(data)
    data = remove.filter(data)
    
    # Split the data into training data and testing data
    train, test = data.train_test_split(100*data_split, Random(1))
    
    # Store the time before starting to build the tree
    starttime = time.time()
    
    # Build and Train the weka tree
    cls = Classifier(classname="weka.classifiers.trees.J48")    
    cls.build_classifier(train)
    
    # Store the time once the tree has been built
    endtime = time.time()
    
    # Render a png image of the weka tree to display in the PySimpleGUI popup
    g = graphviz.Source(cls.graph)
    g.format = "png"
    g.render('weka-test.gv', view=False)
    
    # Predict the output for the test data 
    output = PredictionOutput(classname="weka.classifiers.evaluation.output.prediction.CSV", options=["-distribution"])
    evl = Evaluation(train)
    
    # Get the accuracy of the predicted data
    evl.test_model(cls, test, output=output)
    correct = evl.percent_correct
    jvm.stop()
    return round(correct, 2), endtime-starttime
