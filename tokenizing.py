import sys, os

import re

import requests
import lxml.html


# ---- Identifying types of characters ----

def isspace(char):
    return re.search('\s', char)


def ispunc(char):
    return char in ['.', ',', '\'', '"', '-', '(', ')', ':', ';', '/', '\\', '!', '?']


def isdigit(char):
    return ord('0') <= ord(char) <= ord('9')


def issymbol(char):
    return char in ['@', '#', '$', '%', '&', '*', '+']


def isletter(char):
    return not isspace(char) and not ispunc(char) and not isdigit(char) and not issymbol(char)


# ---- Ynet Strip ----

def get_page(url):
    r = requests.get(url)
    return r.text

def ynet_strip(html):
    """
    Receives HTML source of a YNET.co.il article page, and returns only the article itself - title, subtitle,
    and paragraphs.

    :param html: An HTML page, as a string, created using a GET request to a YNET article
    :return: A String representing the articles titles and paragraphs
    """

    root = lxml.html.fromstring(html)

    def get_title():
        ans = ''
        for element in root.find_class('art_header_title'):
            ans += element.text_content() + '\n'
        return ans

    def get_subtitle():
        ans = ''
        for element in root.find_class('art_header_sub_title'):
            ans += element.text_content() + '\n'
        return ans

    def clean_text(text):
        """
        Removes unnecessary spacing caused by naive copy of <p> text contents.
        Two newlines or more are translated into one.
        One newline is assumed to be a mistake - the two lines are merged together.
        Whitespace in the beginning of the string is redundant.

        :param text: A string representing an article body, with redundant whitespaces
        :return: The same string but without the redundant whitespaces.
        """

        def reduce_whitespace(whitespace):
            newlines = re.findall('\n', whitespace.group(0))
            if len(newlines) <= 1:
                return ' '
            else:
                return '\n'

        text = re.sub(pattern="\s+", repl=reduce_whitespace, string=text)
        text = re.sub(pattern="^[ \n]+", repl='', string=text)
        return text

    def get_body():
        """
        scans the body of the article ('.art_body') and finds all <p> elements that have article text in them.
        :return: a string with all of the paragraphs in the article, separated by double \n
        """
        ans = ''
        for article_body in root.find_class('art_body'):
            for bad in article_body.find_class('art_video'):
                bad.drop_tree()
            for bad in article_body.findall('.//script'):
                bad.drop_tree()
            for element in article_body.findall('.//p'):
                if len(element.text_content()) < 2:
                    ans += '\n\n'
                else:
                    ans += element.text_content()
        return clean_text(ans)

    return get_title() + get_subtitle() + get_body()

# ---- Dividing into sentences ----

def sentencize(text):
    """
    Divide a long text into sentences, divided by periods (.) and newlines (\n)
    :param text: a string made of one or more sentences
    :return: an array of strings, each representing a sentence in the text
    """

    def is_sentence_end(text, i):
        """
        checks if the given text has a sentence that ends in index i
        - aware of one-letter abbreviations in Hebrew, like S.I. Agnon and G. Yafit
        - aware of period inside numbers: 23.56 or dates 23.2.2015
        :param text: a text that typically has more than one sentence
        :param i: an index of a character in the text
        :return: boolean, whether a sentence ends in the i-th position in the text
        """
        if i < 0 or i >= len(text):
            return False
        if i == len(text) - 1:
            return True
        if text[i] == '\n':
            return True
        if text[i] in ['?', '!']:
            return text[i + 1] not in ['?', '!', '"']
        if text[i] == '"' and i > 0:
            return text[i - 1] in ['.', '?', '!']
        if text[i] == '.':
            if i <= 1 or re.search(string=text[i - 2], pattern='[\s\.]'):
                return False
            return isspace(text[i + 1])

    ans = []
    current_sentence_begin = 0
    for i in range(len(text)):
        if is_sentence_end(text, i):
            if i - current_sentence_begin > 1:
                clean_sentence = re.sub(pattern='^( )*', repl='', string=text[current_sentence_begin: i + 1])
                clean_sentence = re.sub(pattern='\n', repl='', string=clean_sentence)
                ans.append(clean_sentence)
                current_sentence_begin = i + 1
    return ans


# ---- Dividing a sentence into tokens ----

def tokenize(sentence):
    """
    Divide a sentence into tokens
    :param sentence: a string representing a full sentence
    :return: an array of strings, each representing a token in the sentence (including spaces as tokens).
        Joining the strings should return the original sentence.
    """

    def is_token_end(text, i):
        """
        checks if the given sentence has a token that ends in index i
        - aware of popular special characters and their different uses
        - sometimes fails when the sentence is ambiguous
        :param text: a text made up of one sentence
        :param i: an index of a character in the text
        :return: boolean, whether a token ends in the i-th position in the text
        """
        if i == 0:
            return text[i] in ["'", '"']
        if i == len(text) - 1:
            return True
        if text[i] in ['!', '?']:
            return text[i + 1] not in ['!', '?']
        if i == len(text) - 2 and ispunc(text[-1]) and not ispunc(text[i]):
            return True
        if text[i] == ' ' or (i + 1 < len(text) and text[i + 1] == ' '):
            return True
        if text[i] in ['/', ':'] and i > 0 and i + 1 < len(text):
            return not (isdigit(text[i - 1]) and isdigit(text[i + 1]))
        if text[i + 1] in ['/', ':'] and i + 2 < len(text):
            return not (isdigit(text[i]) and isdigit(text[i + 2]))
        if (isletter(text[i - 1]) or isdigit(text[i - 1])) and (isletter(text[i + 1]) or isdigit(text[i + 1])):
            return False
        if i + 2 < len(text) and \
                (isletter(text[i]) or isdigit(text[i])) and (isletter(text[i + 2]) or isdigit(text[i + 2])):
            return False
        if text[i] == '.':
            return len(set(text[i:])) > 1  # it is not the ending periods.....
        if i + 1 < len(text) and text[i] == '.':
            return len(set(text[i + 1:])) == 1  # it is the ending periods.....

        return issymbol(text[i]) or isspace(text[i]) or ispunc(text[i]) or \
               issymbol(text[i + 1]) or isspace(text[i + 1]) or ispunc(text[i + 1])

    ans = []
    current_token_begin = 0
    for i in range(len(sentence)):
        if is_token_end(sentence, i):
            ans.append(sentence[current_token_begin: i + 1])
            current_token_begin = i + 1

    return ans


# ---- Main ----

def save_to_file(text, filename):
    # Save contents to a file using UTF-8 encoding
    file = open(filename, 'w', encoding='utf-8')
    file.write(text)
    file.close()

def file_contents(filename):
    # Read contents from a file
    file = open(filename, 'r')
    ans = file.read()
    file.close()
    return ans

FILE_NAME_ARTICLE = 'article.txt'
FILE_NAME_SENTENCES = 'article_sentences.txt'
FILE_NAME_TOKENS = 'article_tokenized.txt'

if __name__ == '__main__':

    # Get input parameters: url and output_directory

    if len(sys.argv) != 3:
        print('Usage: python <>.py <ynet-article-url> <output-directory>')
        exit(1)

    url = sys.argv[1]
    output_directory = sys.argv[2]

    if not os.path.isdir(output_directory):
        print('Usage: python <>.py <ynet-article-url> <output-directory>')
        print('The path ' + output_directory + ' does not seem like a directory.')
        exit(1)

    # 1. Get the raw article from the page

    page = get_page(url)
    strip = ynet_strip(page)
    save_to_file(strip, output_directory + '/' + FILE_NAME_ARTICLE)

    # 2. Divide the article into sentences and save a string made up of all sentences, each on its own line

    sentences = sentencize(strip)
    all_sentences = ''
    for s in sentences:
        all_sentences += s + '\n'
    save_to_file(all_sentences, output_directory + '/' + FILE_NAME_SENTENCES)

    # 3. Divide each sentence into tokens, and save a string made up of all the sentences, each on its own line,
    #    where each token is separated using a space from the previous token

    all_sentences_tokenized = ''
    for s in sentences:
        s_tokenized = ''
        for token in tokenize(s):
            if token != ' ':
                s_tokenized += token + ' '
        s_tokenized = s_tokenized[:-1]
        all_sentences_tokenized += s_tokenized + '\n'
    save_to_file(all_sentences_tokenized, output_directory + '/' + FILE_NAME_TOKENS)
