# Main file for the implementation of the C4.5 algorithm

def main():
    data = read_csv_data("data_path.csv")
    training_data, testing_data = split_data(data)
    tree = build_tree(training_data)
    visualise_tree(tree)
    test_tree(tree, testing_data)

# Aideen 
def split_data(data):
    return training, testing

# Aideen 
def read_csv_data(csv_path):
    return data

def data_class_check(data):
    return result

# Come back to 
def build_tree(data):
    return tree

# Aideen
def entropy(leaf):
    return entropy

# Louise
def information_gain(leaf):
    return information_gain

# Louise
def visualise_tree(tree):
    # visualaise
   
# Come back to 
def test_tree(tree, testing_data):
    # test tree
    
main()