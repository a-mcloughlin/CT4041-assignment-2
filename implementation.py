# Main file for the implementation of the C4.5 algorithm

def main():
    data = read_csv_data("data_path.csv")
    training_data, testing_data = split_data(data)
    tree = build_tree(training_data)
    visualise_tree(tree)
    test_tree(tree, testing_data)

def split_data(data):
    return training, testing

def read_csv_data(csv_path):
    return data

def data_class_check(data):
    return result

def build_tree(data):
    return tree

def entropy(leaf):
    return entropy

def information_gain(leaf):
    return information_gain

def visualise_tree(tree):
    # visualaise
   
def test_tree(tree, testing_data):
    # test tree
    
main()