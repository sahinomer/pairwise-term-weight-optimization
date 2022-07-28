import numpy as np
from itertools import combinations

from nltk import ngrams
from nltk.tokenize import RegexpTokenizer
from nltk import PorterStemmer
from nltk.corpus import stopwords

tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()


def print_progress_bar(i, n, text=''):
    percent = int(((i + 1) / n) * 100)
    print('\r[%s%s] %3d%%  %8d/%-8d  %s' % ('#' * percent, '-' * (100 - percent), percent, i+1, n, text), end='')
    if percent == 100:
        print()  # new line


def tokenize(text):
    return tokenizer.tokenize(text.lower())


def stem_term(term):
    stemmed_term = [stemmer.stem(t) for t in term.split(' ')]
    return ' '.join(stemmed_term)


def extract_phrase(text, max_phrase_length=None, stem=False, as_set=False):
    tokens = tokenize(text)
    if stem:
        tokens = [stem_term(term) for term in tokens]
    phrase_list = tokens.copy()

    if max_phrase_length is None:
        max_phrase_length = len(tokens)

    for n in range(2, max_phrase_length+1):
        for phrase in ngrams(tokens, n):
            if phrase[0] not in stop_words and phrase[-1] not in stop_words:
                phrase_list.append(' '.join(phrase))

    if as_set:
        return set(phrase_list)

    return phrase_list


def combine_terms(tokenized_terms, ngram_list, combination=False):
    # extract base terms that are non-stopwords
    base_terms = [term for term in tokenized_terms if term not in stop_words]

    for n in ngram_list:
        if combination:
            for comb in combinations(np.arange(len(base_terms)), n):
                tokenized_terms.append(' '.join([base_terms[c] for c in comb]))

        else:  # ngram (default)
            for gram in ngrams(base_terms, n):
                tokenized_terms.append(' '.join(gram))

    return tokenized_terms
