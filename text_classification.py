import re
import sys
import os
import typing

import scipy.sparse
import numpy as np
from sklearn import cross_validation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# ------------- Constants --------------

# Some words from http://www.words-to-use.com/words/movies-tv/

POSITIVE_WORDS = ['entertaining', 'fascinating', 'beautiful', 'powerful', 'fun', 'funny', 'recommend', 'recommended',
                  'brilliant', 'incredible', 'enjoyed', 'talented', 'colorful', 'best', 'amazing', 'wonderful',
                  'charming', 'touching', 'powerful', 'love', 'loved', 'interesting', 'excellent', 'again', 'great',
                  'favorite', 'perfect', 'perfectly']
NEGATIVE_WORDS = ['poor', 'poorly', 'terrible', 'terribly', 'worst', 'disappointing', 'disappointed', 'wasted',
                  'dislike', 'awful', 'boring', 'predictable', 'annoying', 'hate', 'hated', 'avoid',
                  'failed', 'stupid', 'bad', 'ruined', 'pathetic', 'horrible']
KEYWORDS = POSITIVE_WORDS + NEGATIVE_WORDS

NEG_REVIEW = 0
POS_REVIEW = 1

NEGATIVE_REVIEW_DIRNAME = 'neg'
POSITIVE_REVIEW_DIRNAME = 'pos'


# ---------- Utilities ------------

def arr_length(arr):
    if isinstance(arr, list):
        return len(arr)
    else:
        return arr.shape[0]


def join_arrays(arr1, arr2):
    if isinstance(arr1, list) and isinstance(arr2, list):
        return arr1 + arr2
    elif isinstance(arr1, scipy.sparse.spmatrix) and isinstance(arr2, scipy.sparse.spmatrix):
        return scipy.sparse.vstack((arr1, arr2))
    elif isinstance(arr1, np.ndarray) and isinstance(arr2, np.ndarray):
        return np.vstack((arr1, arr2))
    else:
        raise Exception('Can only join arrays of same base class.')


def expected_results_array(negative_count, positive_count):
    """
    Creates an array of expected results, in the format: [0, 0, 0, ..., 0, 1, 1, ..., 1]
    :param negative_count: The number of negative samples
    :param positive_count: The number of positive samples
    :return: NumPy array of the wanted format
    """
    linear = np.arange(start=1, stop=negative_count + positive_count + 1)
    expected_results = np.where(linear <= negative_count, NEG_REVIEW, POS_REVIEW)
    return expected_results


# -------------- From Files -----------------

def directory_files(directory_path) -> typing.List[typing.AnyStr]:
    filepaths = []
    for dpath, _, filenames in os.walk(directory_path):
        filepaths += [dpath + '/' + file for file in filenames]
    return filepaths


def load_tokenized_file(filename):
    """
    Creates a list of tokens from the tokenized content in the file.

    :param filename: A file with tokenized text (sentences separated by \n, tokens separated with space)
    :return: A list of strings, each a token in the file
    """
    # Read contents from a file
    file = open(filename, 'r', encoding='utf-8')
    text = file.read()
    file.close()
    tokenized = []
    sentences = text.split('\n')
    for sen in sentences:
        tokenized.extend(list(
            filter(lambda tok: tok != '', sen.split(' '))))  # split by ' ' and remove empty tokens

    return tokenized


def build_reviews(directory_path, rating=None):
    """
    Build a list of Review objects from the given directory
    :param directory_path: Directory with "pos" and "neg" subdirectories, each holding review files
    :param rating: Optional, return only positive or only negative reviews. Default - None
    :return: A list of Review objects
    """
    filepaths = []
    if rating != NEG_REVIEW:
        filepaths = directory_files(directory_path + '/' + POSITIVE_REVIEW_DIRNAME)
    if rating != POS_REVIEW:
        filepaths = directory_files(directory_path + '/' + NEGATIVE_REVIEW_DIRNAME)
    if len(filepaths) == 0:
        print('No files found in the given directory')
        exit(1)
    reviews = []
    for f in filepaths:
        reviews.append(Review(f))
    return reviews


def full_feature_vectors(negative_files, positive_files):
    """
    Uses scikit's CountVectorizer to create full feature vectors for the texts - creates a bag of words for each text.
    A bag-of-words is an array indicating which words the text includes, where each position in the array represents a
    different word.
    The words included are ALL the words in ALL the texts (neg and pos) excluding stop words..
    Two different lists are returned - the bags-of-words for negatives, and the bags-of-words for the positive

    :param negative_files: A Python list of files, each being a negative sample
    :param positive_files: A Python list of files, each being a positive sample
    :return: (neg_vectors, pos_vectors, feature_names): Array of bags-of-words of the negative files,
     Array of bags-of-words of the positive files, List of strings of all the features
    """
    all_files = negative_files + positive_files
    counter = CountVectorizer(input='filename', stop_words='english')
    all_vectors = counter.fit_transform(all_files)
    negative_count = len(negative_files)
    return all_vectors[:negative_count], all_vectors[negative_count:], counter.get_feature_names()


def select_k_best(negative_files, positive_files, k=50):
    """
    Calculates the k best features (words) that differentiate between the negative samples and the positive samples.
    Then returns the feature vectors and the features (words) themselves
    :param negative_files: A list of negative review files
    :param positive_files: A list of positive review files
    :return: (neg_feature_vecs, pos_feature_vecs, features), where the first two are two dimensional arrays of size
     num_of_neg_samples * k and num_of_pos_samples * k accordingly, and features is a list of k strings (words
    """
    full_neg_features, full_pos_features, feature_names = full_feature_vectors(negative_files, positive_files)
    reviews_features = join_arrays(full_neg_features, full_pos_features)
    expected_results = expected_results_array(len(negative_files), len(positive_files))
    feature_selector = SelectKBest(k=k)
    new_feature_vectors = feature_selector.fit_transform(reviews_features, expected_results)
    best_indices = feature_selector.get_support(indices=True)
    best_features = [feature_names[i] for i in best_indices]
    neg_count = len(negative_files)
    return new_feature_vectors[:neg_count], new_feature_vectors[neg_count:], best_features


# ----------- Review Class ---------------

class Review:
    def __init__(self, filepath):
        self.prediction = None
        self._rating = int(re.sub(r'.*_([0-9]+).tok.txt', r'\1', filepath))  # TODO remove before handing in
        self._tokens = load_tokenized_file(filepath)
        self._manual_feature_vector = None
        self._path = filepath

    def get_manual_feature_vector(self):
        """
        Creates a feature vector for the review - a bag of words using the keywords chosen manually.
        :return: array, the size of the manual keywords, with boolean values in each position, indicating if the review
         has this word
        """
        if self._manual_feature_vector is None:
            self._manual_feature_vector = feature_vec = np.zeros(len(KEYWORDS))
            for i in range(len(KEYWORDS)):
                keyword = KEYWORDS[i]
                existence = 1 if keyword in self._tokens else 0
                feature_vec[i] = existence
        return self._manual_feature_vector

    def get_filepath(self):
        return self._path

    def get_rating(self):
        return self._rating

    def get_path(self):
        return self._path

    def get_tokens(self):
        return self._tokens


# ---------- Classifiers ------------

def build_classifier(classifier_type, negative_vectors, positive_vectors):
    """
    Build a classifier (using sklearn) that differentiates between negative and positive examples
    :param classifier_type: A class, for example sklearn.svm.SVC. The classifier will be initiated using this variable.
     Should implement the method `fit`, as defined for sklearn classifiers.
    :param negative_vectors: Negative samples that should be used to train the classifier
    :param positive_vectors: Positive samples that should be used to train the classifier
    :return: A trained classifier
    """
    classifier_ans = classifier_type()
    all_samples = join_arrays(negative_vectors, positive_vectors)

    neg_count, pos_count = arr_length(negative_vectors), arr_length(positive_vectors)
    expected_results = expected_results_array(neg_count, pos_count)

    classifier_ans.fit(all_samples, expected_results)
    return classifier_ans


def estimate_accuracy(classifier, negative_samples, positive_samples, cross_val_folds=10):
    """
    Use ten fold cross validation to estimate the accuracy of the given classifier
    :param classifier: A classifier object to test. Should implement a `predict` method that aims to return non-positive
     values for negative samples and positive values for positive samples
    :param negative_samples:
    :param positive_samples:
    :return: Float value that estimates the accuracy of the classifier
    """

    all_samples = join_arrays(negative_samples, positive_samples)

    neg_count, pos_count = arr_length(negative_samples), arr_length(positive_samples)
    expected_results = expected_results_array(neg_count, pos_count)

    tenfold_validation = cross_validation.KFold(arr_length(all_samples), n_folds=cross_val_folds)
    accuracies = cross_validation.cross_val_score(classifier, all_samples, expected_results, scoring='accuracy',
                                                  cv=tenfold_validation)

    accuracy = np.average(accuracies)
    return accuracy


# -------------- Main -----------------

def print_accuracy(name, classifier, negative_samples, positive_samples):
    accuracy = estimate_accuracy(classifier, negative_samples, positive_samples)
    print('- {classifier_name}: {accuracy}'.format(classifier_name=name, accuracy=accuracy))


if __name__ == '__main__':

    if len(sys.argv) != 2:
        print('Please use the right numbers of arguments.')
        print('Usage: python <>.py <FolderWithInputFiles>')
        exit(1)
    input_folder = sys.argv[1]

    neg_reviews = build_reviews(input_folder, rating=NEG_REVIEW)
    pos_reviews = build_reviews(input_folder, rating=POS_REVIEW)

    # -------- Classifying by manual vectors ----------

    neg_manual_vectors = [rev.get_manual_feature_vector() for rev in neg_reviews]
    pos_manual_vectors = [rev.get_manual_feature_vector() for rev in pos_reviews]

    svm = build_classifier(SVC, neg_manual_vectors, pos_manual_vectors)
    naive_bayes = build_classifier(MultinomialNB, neg_manual_vectors, pos_manual_vectors)
    decision_tree = build_classifier(DecisionTreeClassifier, neg_manual_vectors, pos_manual_vectors)
    knn = build_classifier(KNeighborsClassifier, neg_manual_vectors, pos_manual_vectors)

    print_accuracy('SVM', svm, neg_manual_vectors, pos_manual_vectors)
    print_accuracy('Naive Bayes', naive_bayes, neg_manual_vectors, pos_manual_vectors)
    print_accuracy('DecisionTree', decision_tree, neg_manual_vectors, pos_manual_vectors)
    print_accuracy('KNN', knn, neg_manual_vectors, pos_manual_vectors)
    print()

    # ---------- Classifying with Full Vectors ----------------

    negative_files = directory_files(input_folder + '/' + NEGATIVE_REVIEW_DIRNAME)
    positive_files = directory_files(input_folder + '/' + POSITIVE_REVIEW_DIRNAME)
    # neg_full_vectors, pos_full_vectors = full_feature_vectors(negative_files, positive_files)
    #
    # svm = build_classifier(SVC, neg_full_vectors, pos_full_vectors)
    # naive_bayes = build_classifier(MultinomialNB, neg_full_vectors, pos_full_vectors)
    # decision_tree = build_classifier(DecisionTreeClassifier, neg_full_vectors, pos_full_vectors)
    # knn = build_classifier(KNeighborsClassifier, neg_full_vectors, pos_full_vectors)
    #
    # print_accuracy('SVM', svm, neg_full_vectors, pos_full_vectors)
    # print_accuracy('Naive Bayes', naive_bayes, neg_full_vectors, pos_full_vectors)
    # print_accuracy('DecisionTree', decision_tree, neg_full_vectors, pos_full_vectors)
    # print_accuracy('KNN', knn, neg_full_vectors, pos_full_vectors)
    # print()

    # ------------------- Get K Best Features ----------------------

    neg_reviews_best_features, pos_reviews_best_features, best_features = select_k_best(negative_files, positive_files)

    svm = build_classifier(SVC, neg_reviews_best_features, pos_reviews_best_features)
    naive_bayes = build_classifier(MultinomialNB, neg_reviews_best_features, pos_reviews_best_features)
    decision_tree = build_classifier(DecisionTreeClassifier, neg_reviews_best_features, pos_reviews_best_features)
    knn = build_classifier(KNeighborsClassifier, neg_reviews_best_features, pos_reviews_best_features)
    
    print_accuracy('SVM', svm, neg_reviews_best_features, pos_reviews_best_features)
    print_accuracy('Naive Bayes', naive_bayes, neg_reviews_best_features, pos_reviews_best_features)
    print_accuracy('DecisionTree', decision_tree, neg_reviews_best_features, pos_reviews_best_features)
    print_accuracy('KNN', knn, neg_reviews_best_features, pos_reviews_best_features)
    print()
