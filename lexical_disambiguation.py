import string
import sys
from collections import Counter
from math import log2
from xml.etree import ElementTree

import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize


# ----------- Utilities -------------


def is_number(s):
    """
    :param s: A string, that might represent a number, i.e. '5123.41' or not, i.e. 'book'
    :return: Whether the string represents a number
    """
    try:
        float(s)
        return True
    except ValueError:
        return False


def get_wordnet_tag(treebank_tag):
    """
    Translates a TreeBank POS tag to a WordNet POS tag
    :param treebank_tag: TreeBank POS tag
    :return: WordNet POS tag
    """
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        # found none
        return wordnet.NOUN


def argmax(dictionary):
    """
    :param dictionary: A dictionary object, and its values need to be comparable
    :return: The key that has the highest value
    """
    return max(dictionary.items(), key=lambda x: x[1])[0]


def save_to_file(text, filename):
    # Save contents to a file using UTF-8 encoding
    file = open(filename, 'w', encoding='utf-8')
    file.write(text)
    file.close()


# --------- XML File Parsing ------------

def get_instance_list(xml_file):
    """
    :param xml_file: XML file path, that has the corpus. The file's structure: corpus > lexelt > instances
    :return: A list of XML `instance` objects
    """
    return ElementTree.parse(xml_file).getroot()[0]


def get_sense(instance):
    """
    :param instance: An xml.etree Element, that represents an instance of a word
    :return: The name of the sense as it appears in the tree
    """
    return instance[0].get('senseid')


def get_id(instance):
    """
    :param instance: An xml.etree Element, that represents an instance of a word
    :return: The id of the instance as it appears in the tree
    """
    return instance.get('id')


def get_paragraph(instance):
    """
    :param instance: An xml.etree Element, that represents an instance of a word
    :return: The paragraph of the instance
    """
    ans = ''
    for sentence in instance.itertext():
        ans += sentence
    return ans


# ------ Naive Bayes Classifier ------

class TextBayes:
    """
    Naive Bayes Classifier for text.
    """

    tagger = nltk.PerceptronTagger()
    lemmatizer = nltk.WordNetLemmatizer()
    stop_words = stopwords.words('english')

    UNKNOWN_TOKEN = -1

    def __init__(self, smoothing='add-one'):
        self._smoothing = smoothing
        if self._smoothing not in [None, 'add-one']:
            raise Exception('Unknown smoothing option: {}'.format(smoothing))

        self._classes = []
        """ A list of classes that need to be distinguished """

        self._priors = {}
        """ Dictionary from class (string) to prior, sum of all priors 1 """

        self._cond_probabilities = {}
        """ Dictionary from class (string) to dictionary from token (string) to probability,
         for example: _cond_probabilities['formation']['together'] = P[token together | class formation]

         If smoothing is not None, the inner dictionaries have one extra key: UNKNOWN_TOKEN,
          which has its own probability """

    def train(self, paragraphs, classes):
        """
        :param paragraphs: A list of paragraphs (strings), where each paragraph hash a different class
        :param classes: A list, same length as x, where each entry is the class name (string) for each paragraph in x
        """
        if len(paragraphs) != len(classes):
            raise Exception(
                'Parameters `paragraphs` and `classes` should match in size ({}, {}).'.format(len(paragraphs),
                                                                                              len(classes)))
        class_counts = Counter(classes)
        self._classes = list(class_counts.keys())
        for c in self._classes:
            self._priors[c] = class_counts[c] / len(classes)
        for c in self._classes:
            self._cond_probabilities[c] = {}

        # create a bag of words for each class
        word_bags = {}
        for clazz in self._classes:
            word_bags[clazz] = []
        for paragraph_i in range(len(paragraphs)):
            paragraph_strip = TextBayes.break_down(paragraphs[paragraph_i])
            clazz = classes[paragraph_i]
            word_bags[clazz].extend(paragraph_strip)

        # create a multiset for each bag of words
        for clazz in self._classes:
            word_bags[clazz] = Counter(word_bags[clazz])

        # compute conditional probability for each word in each sack
        for clazz in self._classes:
            bag_size = sum(word_bags[clazz].values())
            types_count = len(word_bags[clazz])
            for token, count in word_bags[clazz].items():
                if self._smoothing is None:
                    self._cond_probabilities[clazz][token] = count / bag_size
                else:  # add-one smoothing
                    self._cond_probabilities[clazz][token] = (count + 1) / (bag_size + types_count)
            if self._smoothing == 'add-one':
                self._cond_probabilities[clazz][TextBayes.UNKNOWN_TOKEN] = 1 / (bag_size + types_count)

    def conditional_probability(self, clazz, token):
        if len(self._classes) == 0:
            raise Exception('The classifier has not been trained yet.')
        if clazz not in self._classes:
            raise Exception('Unknown class.')
        if self._smoothing is None:
            return 0 if token not in self._cond_probabilities[clazz] else self._cond_probabilities[clazz][token]
        else:  # add-one
            return self._cond_probabilities[clazz][TextBayes.UNKNOWN_TOKEN] \
                if token not in self._cond_probabilities[clazz] else self._cond_probabilities[clazz][token]

    def predict(self, paragraph):
        """
        Calculates the most probable class that the paragraph belongs to
        :param paragraph: A string made up of one or more sentences
        :return: A prediction of the class that the paragraph belongs to
        """
        probabilities = self.belong_probabilities(paragraph)
        return argmax(probabilities)

    def belong_probabilities(self, paragraph):
        """
        :param paragraph: A string made up of one or more sentences
        :return: A dictionary from class names to probability, stating the probability of the paragraph belonging to
         each class
        """

        # To prevent underflow, we use loglikelihoods instead of likelihoods, and so we add up log-probability instead
        # of multiplying probability
        loglikelihoods = {}
        for clazz in self._classes:
            cur_likelihood = 0
            for token in TextBayes.break_down(paragraph):
                cond_probability = self.conditional_probability(clazz, token)
                cur_likelihood += log2(cond_probability)
            cur_likelihood += log2(self._priors[clazz])
            loglikelihoods[clazz] = cur_likelihood

        # Because we are interested in the ratio between the different likelihoods, we can divide all of the likelihoods
        # by a constant amount, which is the same as subtracting the loglikelihoods by a constant amount (specifically
        # we subtract by the maximum loglikelihood)
        # Then we exponentiate 2 by the new values to get normalized likelihood values
        likelihoods_normalized = {}
        max_likelihood = max(loglikelihoods.values())
        for clazz, loglike in loglikelihoods.items():
            likelihoods_normalized[clazz] = 2 ** (loglike - max_likelihood)

        # Compute the ratios between the likelihoods to get probabilities
        ans = {}
        sum_norm_likelihoods = sum(likelihoods_normalized.values())
        for clazz, norm_likelihood in likelihoods_normalized.items():
            ans[clazz] = norm_likelihood / sum_norm_likelihoods
        return ans

    @staticmethod
    def break_down(paragraph):
        def break_down_weak(paragraph):
            """
            Use natural language processing tools to break down the paragraph into a sequence of tokens

            :param paragraph: A string made up of one or more sentences
            :return: A list of tokens (strings) from the paragraph
            """
            tokens = word_tokenize(paragraph)
            return tokens

        def break_down_strong(paragraph):
            """
            Use natural language processing tools to break down the paragraph into a sequence of lemmatized words.
            Removes English stop words, punctuation, and numbers.

            :param paragraph: A string made up of one or more sentences
            :return: A list of words (strings) from the paragraph
            """
            tokens = word_tokenize(paragraph)
            parts_of_speech = TextBayes.tagger.tag(tokens)
            parts_of_speech = [(t[0], get_wordnet_tag(t[1])) for t in parts_of_speech]
            lemmatized = [TextBayes.lemmatizer.lemmatize(t[0], pos=t[1]) for t in parts_of_speech]
            lowercase = [t.lower() for t in lemmatized]
            return [t for t in lowercase if
                    t not in string.punctuation and t not in TextBayes.stop_words and not is_number(t)]

        return break_down_weak(paragraph)

    @staticmethod
    def from_file(xml_file, smoothing='add-one'):
        """
        Create a TextBayes classifier from a given corpus
        :param xml_file: XML file path, that has the corpus. The file's structure: corpus > lexelt > instances
        :param smoothing: Smoothing technique for the classifier
        :return: Trained TextBayes object
        """
        instance_list = get_instance_list(xml_file)
        paragraphs = [get_paragraph(instance) for instance in instance_list]
        senses = [get_sense(instance) for instance in instance_list]

        ans = TextBayes(smoothing=smoothing)
        ans.train(paragraphs, senses)
        return ans


# -------- Confusing Matrix, Precision and Recall ------

class ConfusionMatrix:
    def __init__(self, classifier, samples, classes):
        """
        Computes a Confusion Matrix for the classifier, and allows precision and recall queriess
        :param classifier: The classifier that needs to be tested. Needs a `predict` method.
        :param samples: The test items that are classified
        :param classes: The classes that the test items belong to
        """
        self._classes = list(set(classes))
        self._matrix = {}
        for real_class in classes:
            self._matrix[real_class] = {}
            for confused_class in classes:
                self._matrix[real_class][confused_class] = 0

        for sample_i in range(len(samples)):
            real_class = classes[sample_i]
            confused_class = classifier.predict(samples[sample_i])
            self._matrix[real_class][confused_class] += 1

        self._samples_count = len(samples)

    def confused(self, real, confused):
        """
        :param real: Class name
        :param confused: Class name
        :return: The number of samples that are from the real class and were classified as the confused class by the
         classifier. The two parameters can be identical.
        """
        if real not in self.classes or confused not in self.classes:
            raise Exception('Unknown class.')
        return self._matrix[real][confused]

    def precision(self, clazz):
        """
        :param clazz: Name of a class
        :return: The precision of the classifier for that class
        """
        classified_as_clazz = 0  # the number of samples classified as the given clazz
        for real_class in self._classes:
            classified_as_clazz += self._matrix[real_class][clazz]
        return self._matrix[clazz][clazz] / classified_as_clazz

    def recall(self, clazz):
        """
        :param clazz: Name of a class
        :return: The recall of the classifier for that class
        """
        samples_of_clazz = 0  # the number of samples that belonged to the given clazz
        for confused_clazz in self._classes:
            samples_of_clazz += self._matrix[clazz][confused_clazz]
        return self._matrix[clazz][clazz] / samples_of_clazz

    def accuracy(self):
        """
        :return: The percentage of samples that were classified correctly
        """
        correct = 0
        for clazz in self._classes:
            correct += self._matrix[clazz][clazz]
        return correct / len(self)

    def classes(self):
        return self._classes

    def __len__(self):
        return self._samples_count


# ------------ Main ---------------

if __name__ == '__main__':

    train_corpus = sys.argv[1]
    test_corpus = sys.argv[2]
    output_file = sys.argv[3]

    classifier = TextBayes.from_file(train_corpus)

    test_instances = get_instance_list(test_corpus)
    paragraphs = [get_paragraph(i) for i in test_instances]
    true_classes = [get_sense(i) for i in test_instances]

    guesses_string = ''
    for instance_i in range(len(test_instances)):
        instance = test_instances[instance_i]
        paragraph = paragraphs[instance_i]
        guesses_string += '{instance_id} {guess}\n'.format(instance_id=get_id(instance),
                                                           guess=classifier.predict(paragraph))
    save_to_file(guesses_string, output_file)

    confusion_matrix = ConfusionMatrix(classifier, paragraphs, true_classes)
    for sense in sorted(confusion_matrix.classes()):
        print('{sense}: precision: {0}, recall {1}'.format(confusion_matrix.precision(sense),
                                                           confusion_matrix.recall(sense), sense=sense))
    print()
    print('overall accuracy: {}'.format(confusion_matrix.accuracy()))
