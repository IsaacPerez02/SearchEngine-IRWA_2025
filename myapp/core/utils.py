import datetime
from random import random

from faker import Faker

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords


fake = Faker()

def get_random_date():
    """Generate a random datetime between `start` and `end`"""
    return fake.date_time_between(start_date='-30d', end_date='now')


def get_random_date_in(start, end):
    """Generate a random datetime between `start` and `end`"""
    return start + datetime.timedelta(
        # Get a random amount of seconds between `start` and `end`
        seconds=random.randint(0, int((end - start).total_seconds())), )

def remove_punctuation(text):
    cleaned = ""
    for char in text:
        if char.isalnum() or char.isspace() or char == "-":
            cleaned += char
        else:
            cleaned += " "  # Replace punctuation with space
    return cleaned

def build_terms(line):
    """
    Preprocess a line:
    ●  Removing stop words 
    ●  Tokenization 
    ●  Removing punctuation marks 
    ●  Stemming 
    ●  Transforming to lowercase

    Argument:
    line -- string (text) to be preprocessed

    Returns:
    line - a list of tokens corresponding to the input text after the preprocessing
    """

    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))
    line = line.lower()
    line = remove_punctuation(line)
    line = line.split()
    line = [x for x in line if x not in stop_words]
    line = [stemmer.stem(word) for word in line]
    return line
