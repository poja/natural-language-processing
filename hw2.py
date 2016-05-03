import math
from functools import reduce
import sys
import os


# --------------------- Utility --------------------------

def str_ngram(ngram_w_value, amplify=1):
    """
    Returns a string representation of a collocation that has a value.
    The string representation: {First word} {Second word} ... { Last word }     {value}
    :param ngram_w_value: An n-gram with a value, i.e. (('New', 'York'), 34.2354)
    :param amplify: This parameter allows to multiply the value before printing it. Default 1.
    :return: string representation of the collocation with its value.
    """
    ngram = reduce(lambda prev, cur: prev + cur + ' ', ngram_w_value[0], '')
    value = ngram_w_value[1]
    return '{: <35}\t{:.3f}'.format(ngram, value * amplify)


# ----------- Getting n-grams from the text --------------


def all_unigrams(text_tokenized):
    """
    :param text_tokenized: A long text, generally made up of different sentences.
     Tokenized, i.e. given as an array of sentences, each represented by an array of tokens
    :return: A list of unigrams in the text Collocations will appear in the order of appearance in the text -
     collocations that appear more than once in the text will appear more than once in the returned list.
    """
    unigrams = []
    for sentence in text_tokenized:
        sentence = list(filter(lambda x: x != ' ', sentence))
        unigrams.extend(sentence)
    return unigrams


def all_collocations(text_tokenized):
    """
    :param text_tokenized: A long text, generally made up of different sentences.
     Tokenized, i.e. given as an array of sentences, each represented by an array of tokens
    :return: A list of collocations in the text, each in the form (first, second).
     Collocations will appear in the order of appearance in the text - collocations that appear more than once in the
     text will appear more than once in the returned list.
    """
    collocations = []
    for sentence in text_tokenized:
        sentence = list(filter(lambda x: x != ' ', sentence))
        for i in range(len(sentence) - 1):
            collocations.append((sentence[i], sentence[i + 1]))
    return collocations


def all_trigrams(text_tokenized):
    """
    :param text_tokenized: A long text, generally made up of different sentences.
     Tokenized, i.e. given as an array of sentences, each represented by an array of tokens
    :return: A list of trigrams in the text, each in the form (first, second, third).
     Trigrams will appear in the order of appearance in the text - trigrams that appear more than once in the
     text will appear more than once in the returned list.
    """
    trigrams = []
    for sentence in text_tokenized:
        sentence = list(filter(lambda x: x != ' ', sentence))
        for i in range(len(sentence) - 2):
            trigrams.append((sentence[i], sentence[i + 1], sentence[i + 2]))
    return trigrams


# ------------ Calculate n-gram statistics --------------


def unigram_probabilities(tokenized_text):
    """
    Calculate the probability of each unigram, by MLE of the text.
    Calculation: Pr = #{ specific unigram } / #{ unigrams }
    :param tokenized_text: A long text, generally made up of different sentences.
     Tokenized, i.e. given as an array of sentences, each represented by an array of tokens
    :return: A dictionary, from unigrams (string token) to probabilities
    """
    unigrams_list = all_unigrams(tokenized_text)
    unigrams_count = len(unigrams_list)
    unigram_prob = {}
    for unigram in unigrams_list:
        if unigram in unigram_prob:
            unigram_prob[unigram] += 1. / unigrams_count
        else:
            unigram_prob[unigram] = 1. / unigrams_count
    return unigram_prob


def collocation_raw_frequencies(tokenized_text):
    """
    Calculate the raw frequencies of each collocation in the text.
    Calculation: Freq = #{ collocation } / #{ unigrams }
    :param tokenized_text: A long text, generally made up of different sentences.
     Tokenized, i.e. given as an array of sentences, each represented by an array of tokens
    :return: A dictionary, from collocations (2-tuple (first, second)) to raw frequencies
    """
    
    unigrams_count = len(all_unigrams(tokenized_text))
    print('unigrams_count {}'.format(unigrams_count))
    collocations_list = all_collocations(tokenized_text)
    collocation_frequencies = {}
    for collocation in collocations_list:
        if collocation in collocation_frequencies:
            collocation_frequencies[collocation] += 1. / unigrams_count
        else:
            collocation_frequencies[collocation] = 1. / unigrams_count
    return collocation_frequencies


def collocation_probabilities(tokenized_text):
    """
    Calculate the probability of each collocation, by MLE of the text.
    Calculation: Pr = #{ specific collocation } / #{ collocations }
    :param tokenized_text: A long text, generally made up of different sentences.
     Tokenized, i.e. given as an array of sentences, each represented by an array of tokens
    :return: A dictionary, from collocations (2-tuple (first, second)) to probabilities
    """
    
    collocations_list = all_collocations(tokenized_text)
    collocations_count = len(collocations_list)
    print('collocations_count {}'.format(collocations_count))
    collocation_prob = {}
    for collocation in collocations_list:
        if collocation in collocation_prob:
            collocation_prob[collocation] += 1. / collocations_count
        else:
            collocation_prob[collocation] = 1. / collocations_count
    return collocation_prob


def collocation_pmi_values(tokenized_text, wordcount_filter=1):
    """
    Calculate the PMI value of each collocation.
    Calculation: PMI(x,y) = log [ Pr(x,y) / Pr(x)Pr(y) ]
    :param tokenized_text: A long text, generally made up of different sentences.
     Tokenized, i.e. given as an array of sentences, each represented by an array of tokens
    :param: wordcount_filter: Filter out collocations, that their tokens have less than this number of occurrences
     in the text. Default 1 - no filtering.
    :return: A dictionary, from collocations (2-tuple (first, second)) to PMI values
    """
    
    unigrams_pr = unigram_probabilities(tokenized_text)
    uni_count = len(all_unigrams(tokenized_text))
    collocations_pr = collocation_probabilities(tokenized_text)
    collocations_pr_filtered = {col: pr for col, pr in collocations_pr.items()
                                if round(uni_count * unigrams_pr[col[0]]) >= wordcount_filter and
                                round(uni_count * unigrams_pr[col[1]]) >= wordcount_filter}
    collocations_pmi = {}
    for c in collocations_pr_filtered:
        collocations_pmi[c] = math.log2(collocations_pr_filtered[c] / (unigrams_pr[c[0]] * unigrams_pr[c[1]]))

    return collocations_pmi


def trigram_probabilities(tokenized_text):
    """
    Calculate the probability of each trigram, by MLE of the text.
    Calculation: Pr = #{ specific trigram } / #{ trigrams }
    :param tokenized_text: A long text, generally made up of different sentences.
     Tokenized, i.e. given as an array of sentences, each represented by an array of tokens
    :return: A dictionary, from trigrams (3-tuple (first, second, third)) to probabilities
    """
    
    trigrams_list = all_trigrams(tokenized_text)
    trigrams_count = len(trigrams_list)
    print('trigrams_count {}'.format(trigrams_count))
    trigrams_prob = {}
    for trigram in trigrams_list:
        if trigram in trigrams_prob:
            trigrams_prob[trigram] += 1. / trigrams_count
        else:
            trigrams_prob[trigram] = 1. / trigrams_count
    return trigrams_prob


def trigram_pmi_values(tokenized_text, pmi_type, wordcount_filter=1):
    """
    Calculate the PMI value of each trigram.
    Since PMI value of a trigram is not well-defined, parameter pmi_type specifies how to calculate the PMI value.
    :param tokenized_text: A long text, generally made up of different sentences.
     Tokenized, i.e. given as an array of sentences, each represented by an array of tokens
    :param: pmi_type: A character, one of three options 'a', 'b', 'c'. Each specifies a different PMI calculation:
     PMI_a(x, y, z) = Pr(xyz) / Pr(x)Pr(y)Pr(z)
     PMI_b(x, y, z) = Pr(xyz) / Pr(xy)Pr(yz)
     PMI_c(x, y, z) = Pr(xyz) / Pr(x)Pr(y)Pr(z)Pr(xy)Pr(yz)
    :param: wordcount_filter: Filter out trigrams, that their tokens have less than this number of occurrences
     in the text. Default 1 - no filtering.
    :return: A dictionary, from trigrams (3-tuple (first, second, third)) to PMI values
    """
    
    unigrams_pr = unigram_probabilities(tokenized_text)
    uni_count = len(all_unigrams(tokenized_text))
    collocations_pr = collocation_probabilities(tokenized_text) \
        if pmi_type != 'a' else None
    trigrams_pr = trigram_probabilities(tokenized_text)
    trigrams_pr_filtered = {trig: pr for trig, pr in trigrams_pr.items()
                            if round(uni_count * unigrams_pr[trig[0]]) >= wordcount_filter and
                            round(uni_count * unigrams_pr[trig[1]]) >= wordcount_filter and
                            round(uni_count * unigrams_pr[trig[2]]) >= wordcount_filter}
    trigrams_pmi = {}
    for trig in trigrams_pr_filtered:
        if pmi_type == 'a':
            trigrams_pmi[trig] = math.log2(
                trigrams_pr[trig] / (unigrams_pr[trig[0]] * unigrams_pr[trig[1]] * unigrams_pr[trig[2]]))
        elif pmi_type == 'b':
            trigrams_pmi[trig] = math.log2(
                trigrams_pr[trig] / (collocations_pr[trig[0], trig[1]] * collocations_pr[trig[1], trig[2]]))
        elif pmi_type == 'c':
            unigrams_probability_product = unigrams_pr[trig[0]] * unigrams_pr[trig[1]] * unigrams_pr[trig[2]]
            collocations_probability_product = collocations_pr[trig[0], trig[1]] * collocations_pr[trig[1], trig[2]]
            trigrams_pmi[trig] = math.log2(trigrams_pr[trig] / (
                unigrams_probability_product * collocations_probability_product))
    return trigrams_pmi


# -------------- Main ----------------


def save_to_file(text, filename):
    # Save contents to a file using UTF-8 encoding
    file = open(filename, 'w', encoding='utf-8')
    file.write(text)
    file.close()


def file_contents(filename):
    # Read contents from a file
    file = open(filename, 'r', encoding='utf-8')
    ans = file.read()
    file.close()
    return ans


def directory_file_contents(dir_name):
    # Get all file names in directory
    files = []
    for (dirpath, dirnames, filenames) in os.walk(dir_name):
        files = filenames
    # Merge all files into one
    return reduce(lambda old, file: old + file_contents(dir_name + '/' + file) + '\n', files, '')


def best_ngrams(valued_ngrams, elected_count=100):
    """
    Sorts the given valued n-grams and returns the best ones.
    :param valued_ngrams: A dictionary from n-grams (tuples) to values (floats)
    :param: elected_count: The number of best n-grams to leave in the array. Default 100.
    :return: A list of all valued n-grams, in the form (ngram, value), sorted by value (high to low).
     If two are the same value, sorts alphabetically (low to high)
     Then, only the 100 best are left.
    """
    sorted_array = sorted(valued_ngrams.items(), key=lambda x: x[0])  # Sort by increasing ABC
    sorted_array.sort(key=lambda x: x[1], reverse=True)  # Sort by decreasing value
    return sorted_array[0: min(elected_count, len(sorted_array))]  # Return only best ones


if __name__ == '__main__':

    OUTPUT_FILE_FREQ_RAW = 'freq_raw.txt'
    OUTPUT_FILE_PMI_PAIR = 'pmi_pair.txt'
    OUTPUT_FILE_PMI_TRI_A = 'pmi_tri_a.txt'
    OUTPUT_FILE_PMI_TRI_B = 'pmi_tri_b.txt'
    OUTPUT_FILE_PMI_TRI_C = 'pmi_tri_c.txt'

    WORDCOUNT_FILTER = 20

    if len(sys.argv) != 3:
        print('Please use the right numbers of arguments.')
        print('Usage: python <>.py <FolderWithInputFiles> <FolderForOutputFiles>')
        exit(1)
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    corpus_str = directory_file_contents(input_folder)
    # save_to_file(corpus_str, output_folder + '/full_corpus.txt')
    corpus_tok = corpus_str.split('\n')
    for i in range(len(corpus_tok)):
        corpus_tok[i] = list(
            filter(lambda tok: tok != '', corpus_tok[i].split(' ')))  # split by ' ' and remove empty tokens

    # Collocation Raw Frequency
    collocation_rawfreq_dict = collocation_raw_frequencies(corpus_tok)
    collocation_frawfreq = best_ngrams(collocation_rawfreq_dict)
    str_output = reduce(lambda prev, cur: prev + str_ngram(cur, 1000) + '\n', collocation_frawfreq, '')
    save_to_file(str_output, output_folder + '/' + OUTPUT_FILE_FREQ_RAW)

    # Collocation PMI
    collocation_pmi_dict = collocation_pmi_values(corpus_tok, wordcount_filter=WORDCOUNT_FILTER)
    collocation_pmi = best_ngrams(collocation_pmi_dict)
    str_output = reduce(lambda prev, cur: prev + str_ngram(cur) + '\n', collocation_pmi, '')
    save_to_file(str_output, output_folder + '/' + OUTPUT_FILE_PMI_PAIR)

    # Trigram PMI A
    trigram_pmi_dict = trigram_pmi_values(corpus_tok, 'a', wordcount_filter=WORDCOUNT_FILTER)
    trigram_pmi = best_ngrams(trigram_pmi_dict)
    str_output = reduce(lambda prev, cur: prev + str_ngram(cur) + '\n', trigram_pmi, '')
    save_to_file(str_output, output_folder + '/' + OUTPUT_FILE_PMI_TRI_A)

    # Trigram PMI B
    trigram_pmi_dict = trigram_pmi_values(corpus_tok, 'b', wordcount_filter=WORDCOUNT_FILTER)
    trigram_pmi = best_ngrams(trigram_pmi_dict)
    str_output = reduce(lambda prev, cur: prev + str_ngram(cur) + '\n', trigram_pmi, '')
    save_to_file(str_output, output_folder + '/' + OUTPUT_FILE_PMI_TRI_B)

    # Trigram PMI C
    trigram_pmi_dict = trigram_pmi_values(corpus_tok, 'c', wordcount_filter=WORDCOUNT_FILTER)
    trigram_pmi = best_ngrams(trigram_pmi_dict)
    str_output = reduce(lambda prev, cur: prev + str_ngram(cur) + '\n', trigram_pmi, '')
    save_to_file(str_output, output_folder + '/' + OUTPUT_FILE_PMI_TRI_C)
