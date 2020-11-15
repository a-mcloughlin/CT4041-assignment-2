from weka.classifiers import Classifier
from weka.core.converters import Loader
import weka.core.jvm as jvm
from weka.core.dataset import create_instances_from_lists
from weka.filters import Filter
from weka.core.classes import Random
from weka.classifiers import Classifier, Evaluation, PredictionOutput

def build_weka_tree(data_file, data_split):
    jvm.start()
    loader = Loader(classname="weka.core.converters.CSVLoader")
    data = loader.load_file(data_file)
    data.class_index = 3
    train, test = data.train_test_split(100*data_split, Random(1))
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)
    
    output = PredictionOutput(classname="weka.classifiers.evaluation.output.prediction.CSV", options=["-distribution"])
    evl = Evaluation(train)
    evl.test_model(cls, test, output=output)
    print(evl.summary())
    jvm.stop()