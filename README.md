# Language Processing

This project is a collection of lanuage processing tools and a proof of concept for each one, all written for Python 3.5 (with NumPy and scikit-learn, best downloaded via [Anaconda](https://www.continuum.io/downloads)). The tools were built throughout a lanuage processing course, and they are independent of each other.
The tools in the project:
* [Tokenization](#tokenization)
* [Collocation and Trigram Identification](#collocation-and-trigram-identification)
* [Text Classification (two classes)](#text-classification)
* [Lexical Disambiguation](#lexical-disambiguation)

## Tokenization

Tokenizing is the process of reading a text (sequence of characters) and dividing it into sentences and tokens. For example, "Dan's home, because he is sick. I hope he gets better." Would be divided to two sentences, and each one would be divided into tokens. After tokenization, each token is separated from its neighbors with a single space, like so:  

Dan's home , because he is sick .  
I hope he gets better .

The tokenizer is used on Hebrew articles from the news website Ynet, like so:  
Usage: `python tokenizing.py <ynet-article-url> <output-directory>`

Three files will be created in the output directory:
* `article.txt` is the article text
* `article_setences.txt` has every sentence on a new line
* `article_tokenized.txt` has every sentence on a new line, and every token is separated from its neighbors with exactly one space.

## Collocation and Trigram Identification

A [collocation](https://en.wikipedia.org/wiki/Collocation) is a pair of words that tend to appear together, and have a meaning or a usage as a couple. For example, "crystal clear" is a collocation. Trigrams are the same but of length 3. There are several ways to identify collocations and trigrams in a corpus, that are based on counting the appearances of the words, used together and apart. One oftenly used measure is called PMI. 

The tool in this project reads a corpus and then outputs lists of collocations and trigrams, each list based on a different measure.  
Usage: `python collocations.py <FolderWithInputFiles> <FolderForOutputFiles>`

The folder with the input files must have text files that together make up a corpus large enough for the collocations to be meaningful. Five files will be added to the output folder: freq_raw.txt, pmi_pair.txt, pmi_tri_a.txt, pmi_tri_b.txt, pmi_tri_c.txt. Each file is a list of collocations (or trigrams), where each collocation appears on its own line, and its score appears, corresponding to the relevant measure.

## Text Classification

This tool implements classification of texts, into two classes, for example: negative movie reviews vs. position movie reviews. This module greatly depends on scikit-learn for classification tools.

Usage: `python text_classification.py <input_dir> <words_file_input_path> <best_words_file_output_path>`

The `input_dir` should have two subdirectories called `pos` and `neg`, which should each have text files representing reviews of the corrsesponding type. The program then tries different classification methods on these reviews (always training on some of the examples and testing on the others, in a technique called tenfold cross validation). The classification methods vary in feature types, and in classification tool.  
The feature types are:
1. Manual words (from the `words_file_input_path`)
2. Bag of words (all words except stop-words)
3. Selected best words (as suggested by scikit's SelectKBest)

The classification tools are: SVM, Naive Bayes, Decision Tree, KNN. 

All of the results are printed to the screen, and the best features (from SelectKBest) are saved in the file `best_words_file_output_path`.

## Lexical Disambiguation

Lexical ambiguity is when one word has multiple meanings, and it is unclear which one is relevant in a given sentence. Disambiguation is a tool to figure out the correct meaning. To do this, a large training corpos is needed. A multinomial Naive Bayes classifier is implemented, and used to train on the training corpus. The training corpus is made up of paragraphs (called instances) and their senses (meanings). For example, the word "line" has five senses: cord, division, formation, phone, product. 

To use, two steps are needed.  
1. Usage: `python div_train_test.py <input_file_path> <output_files_path>`  
2. Usage: `python lexical_disambiguation.py <train_file_path> <test_file_path> <output_file_path>`

The first step takes an input xml file in the right format (see input line.S2.data.clean.xml) and outputs two corpora - `train.xml` and `test.xml`.  
The second step trains the classifier with the training corpus, then classifies each instance in the test corpus, and outputs the results to the output file. It also outputs the precision, recall, and total accuracy values to the screen.
