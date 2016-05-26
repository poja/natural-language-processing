import random
import sys
from xml.etree import ElementTree
import copy
import numpy as np

# ------------- General Utilities -------------

def save_to_file(text, filename):
    # Save contents to a file using UTF-8 encoding
    file = open(filename, 'w', encoding='utf-8')
    file.write(text)
    file.close()

# ------------ Tree Parse Utilities ------------

def get_sense(instance):
    """
    :param instance: An xml.etree Element, that represents an instance of a word
    :return: The name of the sense as it appears in the tree
    """
    return instance[0].get('senseid')

def replace_instances(tree, new_instances):
    """
    :param tree: An xml.etree ElementTree holding a corpus with instances with different meanings
    :param new_instances: New Element objects that need to replace the old ones
    :return: The same tree (not a copy) but with the new instances
    """
    instance_list = tree.getroot()[0]
    for old_instance in list(instance_list):
        instance_list.remove(old_instance)
    instance_list.extend(new_instances)
    instance_list[-1].tail = '\n\n'
    return tree


# ------------- Dividing the Tree --------------

def divide_instances(instances, test_size):
    """
    Divide a list of instances of a certain word, with different meanings into two trees, one with training samples
     and one with test samples, so that the test samples have exactly test_size random instances of each meaning.

    :param instances: A list of xml.etree Elements representing instances of a word with its meaning (sense)
    :param test_size: The number of test samples of each sense that need to be in the test tree
    :return: Two lists - the second one has exactly test_size samples of each meaning (=senseid) and the first has all
     the rest of the instances
    """
    random.shuffle(instances)
    senses = list(set([get_sense(instance) for instance in instances]))
    training_instances = []
    testing_instances = []
    test_count = np.ndarray((len(senses)))
    for instance in instances:
        for sense_i in range(len(senses)):
            if senses[sense_i] == get_sense(instance):
                if test_count[sense_i] < test_size:
                    testing_instances.append(instance)
                    test_count[sense_i] += 1
                else:
                    training_instances.append(instance)
    if np.any(test_count < test_size):
        raise Exception('test_size is too high ({}). Not all senses have this many instances.'.format(test_size))
    return training_instances, testing_instances


def divide_tree(corpus_tree, test_size=50):
    """
    Divide a tree with different meanings of a certain word, into two trees, one with training samples and one with
     test samples. The test samples have exactly test_size instances of each meaning.
    :param corpus_tree: An xml.etree ElementTree, representing a corpus element, that holds a lexelt element, that holds
     instances of a certain word, with different senses of the word
    :param test_size: The number of test samples of each sense that need to be in the test tree
    :return: Two trees - similar to the original tree, but - the second one has exactly test_size instances of each
     meaning (=senseid) and the first has all the rest of the instances
    """
    instances = corpus_tree.getroot()[0].getchildren()
    training_instances, testing_instances = divide_instances(instances, test_size)

    training_tree = replace_instances(copy.deepcopy(corpus_tree), training_instances)
    testing_tree = replace_instances(copy.deepcopy(corpus_tree), testing_instances)

    return training_tree, testing_tree


if __name__ == "__main__":
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    tree = ElementTree.parse(input_path)
    training, testing = divide_tree(tree)

    save_to_file(ElementTree.tostring(training.getroot(), encoding='unicode'), output_path + './train.xml')
    save_to_file(ElementTree.tostring(testing.getroot(), encoding='unicode'), output_path + './test.xml')
