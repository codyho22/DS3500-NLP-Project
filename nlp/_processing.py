
"""

File: _processing.py
Description: Process text files and save in store

"""
from NLP._stores import stores
from collections import Counter, defaultdict
import numpy as np
from string import punctuation
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer



def _read_file(filename):
    file = open(filename,'r', encoding='utf-8')
    text = file.read()
    return text


def remove_stop_words(text):
    stop_words = _read_file("NLP/ignore_words.txt")
    for word in stop_words:
        text = text.replace(word, "")
    return text

def is_stop_word(word):
    stop_words = _read_file("NLP/ignore_words.txt")
    word = word.strip().lower()
    if word.lower().strip() in stop_words.split("\n"):
        return True
    else:
        return False

def sentiment_processor(filename):
    """adds compound polarity score to stores"""
    nltk.download('all')
    analyzer = SentimentIntensityAnalyzer()

    text = _read_file(filename)
    text = text.replace("\"", "")
    for x in punctuation:
        text = text.replace(x, "")
    text_arr = text.split("\n")
    sentiment = [analyzer.polarity_scores(text)["compound"] for text in text_arr]
    metadata = {
        'sentiment': sentiment,
    }
    return metadata

def _default_parser(filename):
    """ this should probably be a default text parser
    for processing simple unformatted text files. """
    text = _read_file(filename)
    text = text.replace("\n", " ")
    text = text.replace("\"", "")
    for x in punctuation:
        text = text.replace(x, "")
    text_arr = text.split(" ")
    metadata = {
        'wordcount': Counter(text_arr),
        'numwords' : len(text_arr),
    }
    return metadata

def load_text(filename, label=None, parser=None):
    if parser is None:
        metadata = _default_parser(filename)
    else:
        metadata = parser(filename)
    if label is None:
        label = filename
    keys = list(metadata.keys())
    for key in keys:
        if isinstance(metadata[key], dict):
            stores.store(filename, key, [key], list(metadata[key].keys()), list(metadata[key].values()))
        elif isinstance(metadata[key], list):
            stores.store(filename, key, [key], list(range(len(metadata[key]))), metadata[key])
        else:
            stores.store(filename, key, [key], [metadata[key]], [metadata[key]])
