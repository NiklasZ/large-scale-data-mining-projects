# This separate file exists because the pickling code in the threading logic of scikit's Pipeline will not pickle functions
# that are not part of some module.
# See https://stackoverflow.com/questions/45335524/custom-sklearn-pipeline-transformer-giving-pickle-picklingerror
import re
from typing import Dict, List

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from nltk.corpus import wordnet
import nltk
from nltk import WordNetLemmatizer, PorterStemmer, TreebankWordDetokenizer

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
detokeniser = TreebankWordDetokenizer()
number_regex = re.compile(r'^-?\d+(\.\d+)?$')
lemmatized_cache = {}
stemmed_cache = {}


# Taken from https://stackoverflow.com/questions/15586721/wordnet-lemmatization-and-pos-tagging-in-python
def map_pos_to_wordnet(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        # This is the default used by WordNet if no POS tag is given, so we may as well return it here.
        return wordnet.NOUN


def clean(text: str):
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    texter = re.sub(r"<br />", " ", text)
    texter = re.sub(r"&quot;", "\"", texter)
    texter = re.sub('&#39;', "\"", texter)
    texter = re.sub('\n', " ", texter)
    texter = re.sub(' u ', " you ", texter)
    texter = re.sub('`', "", texter)
    texter = re.sub(' +', ' ', texter)
    texter = re.sub(r"(!)\1+", r"!", texter)
    texter = re.sub(r"(\?)\1+", r"?", texter)
    texter = re.sub('&amp;', 'and', texter)
    texter = re.sub('\r', ' ', texter)
    clean = re.compile('<.*?>')
    texter = texter.encode('ascii', 'ignore').decode('ascii')
    texter = re.sub(clean, '', texter)
    if texter == "":
        texter = ""
    return texter


def lemmatize(text: str):
    cleaned = clean(text)
    tokens = nltk.word_tokenize(cleaned)
    tagged_tokens = nltk.pos_tag(tokens)
    lemmatized_tokens = [lemmatizer.lemmatize(t.lower(), pos=map_pos_to_wordnet(tag)) for t, tag in tagged_tokens]
    filtered_punctuation = [t for t in lemmatized_tokens if t not in """!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""]
    filtered_numbers = [t for t in filtered_punctuation if not bool(number_regex.search(t))]
    return detokeniser.detokenize(filtered_numbers)


def stem(text: str):
    cleaned = clean(text)
    tokens = nltk.word_tokenize(cleaned)
    filtered_punctuation = [re.sub(r'[^\w]', ' ', t) for t in tokens]  # strip everything non-alphanumeric
    filtered_numbers = [t for t in filtered_punctuation if not bool(number_regex.search(t))]
    stemmed_tokens = [stemmer.stem(t) for t in filtered_numbers]
    return detokeniser.detokenize(stemmed_tokens)


# Some lazy local caching so we don't pointlessly re-lemmatise or restub the same text
def lemmatize_cache(corpus: pd.Series):
    key = (corpus.size, corpus.iloc[0], corpus.iloc[-1])
    if key not in lemmatized_cache:
        lemmatized_cache[key] = np.vectorize(lemmatize)(corpus)
    return lemmatized_cache[key]


def stem_cache(corpus: pd.Series):
    key = (corpus.size, corpus.iloc[0], corpus.iloc[-1])
    if key not in stemmed_cache:
        stemmed_cache[key] = np.vectorize(stem)(corpus)
    return stemmed_cache[key]


def lemmatize_corpus(corpus: pd.Series):
    return lemmatize_cache(corpus)


def stem_corpus(corpus: pd.Series):
    return stem_cache(corpus)


def resample_and_merge(X: np.ndarray, y: np.ndarray, foccer_count=-1):
    # Unfortunately, the sampler doesn't tell us the named label of a class, (only 0,1)
    # so we have to employ a somewhat clumsy method to figure it out.
    unique, counts = np.unique(y, return_counts=True)
    ros = RandomOverSampler(random_state=42)
    for u, c in zip(unique, counts):
        if c == foccer_count:
            majority_class_size = np.max(counts)
            sample_proportions = {i: majority_class_size if i != u else majority_class_size * 2 for i in unique}
            ros = RandomOverSampler(random_state=42, sampling_strategy=sample_proportions)
    X_res, y_res = ros.fit_resample(X, y)
    return X_res, y_res


def glove_avg(text: str, glove_dict: Dict[str, np.ndarray], stop_words: List[str] = []) -> np.ndarray:
    tokens = nltk.word_tokenize(text)
    embedded = [glove_dict[t] for t in tokens if t not in stop_words and t in glove_dict]
    aggregated = np.vstack(embedded).mean(axis=0)
    return aggregated


def glove_corpus_avg(X: np.ndarray, glove_dict: Dict[str, np.ndarray], stop_words: List[str] = []):
    X_transformed = np.vstack([glove_avg(x, glove_dict, stop_words) for x in X])
    return X_transformed
