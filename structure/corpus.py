# coding: utf-8
import csv
import string
import os
import gensim
from gensim.parsing.preprocessing import STOPWORDS
import spacy as sp
import pickle

from scipy.sparse import dok_matrix
import numpy as np

__authors__ = "Adrien Guille, Hussein AL-NATSHEH"
__emails__ = "adrien.guille@univ-lyon2.fr, hussein.al-natsheh@ish-lyon.cnrs.fr"


class WikiSentences(object):
    def __init__(self, dirname):
        self.dirname = dirname
        self.parser = sp.load('en')
 
    def __iter__(self):
        tokenize = gensim.utils.tokenize
        for sub in os.listdir(self.dirname):
            subdir = self.dirname + sub
            for fname in os.listdir(subdir):
                for line in open(os.path.join(subdir, fname)):
                    if len(line.split()) < 2 or line[:8] == '<doc id=':
                        continue
                    else:                        
                        try:
                            line = line.encode('utf-8','ignore').decode('utf-8')
                            for sent in self.parser(line).sents:
                                tokens = [token for token in tokenize(sent.orth_, lower=True) if token not in STOPWORDS and len(token) > 2]
                                yield tokens
                        except:
                            continue

class Corpus:
    def __init__(self, sentences, max_nb_features=100000, window_size=100, decreasing_weighting=False):

        # first pass to identify features, i.e. the vocabulary
        print('   Identifying features (i.e. the vocabulary)...')
        self.size = 0
        word_frequency = {}
        for sent in sentences:
            self.size += 1
            # update word frequency
            for word in sent:
                if len(word) > 0:
                    frequency = 0
                    if word_frequency.get(word) is not None:
                        frequency = word_frequency[word]
                    frequency += 1
                    word_frequency[word] = frequency
        # sort words w.r.t frequency
        vocabulary = list(word_frequency.items())
        vocabulary.sort(key=lambda x: x[1], reverse=True)
        self.vocabulary = []
        self.vocabulary_map = {}
        # construct the structures
        for i in range(min(max_nb_features, len(vocabulary))):
            feature = vocabulary[i][0]
            self.vocabulary.append(feature)
            self.vocabulary_map[feature] = i
        print('      Corpus size: %d' % self.size)
        print('      Vocabulary size: %d' % len(self.vocabulary))

        # second pass to compute the co-occurrence matrix

        print('   Computing X (i.e. the co-occurrence frequency matrix)...')
        if decreasing_weighting:
            self.X = dok_matrix((len(self.vocabulary), len(self.vocabulary)), dtype=np.float32)
        else:
            self.X = dok_matrix((len(self.vocabulary), len(self.vocabulary)), dtype=np.short)
        # go back to the beginning of the csv file
        for words in sentences:
            self.size += 1
            nb_words = len(words)
            for i in range(nb_words):
                # check whether the current word is part of the vocabulary or not
                if self.vocabulary_map.get(words[i]) is not None:
                    row_index = self.vocabulary_map[words[i]]
                    # extract surrounding words w.r.t window size
                    start = i - window_size
                    if start < 0:
                        start = 0
                    end = i + window_size
                    if end >= nb_words:
                        end = nb_words - 1
                    # scan left context
                    context_left = words[start:i]
                    for j in range(0, len(context_left)):
                        if self.vocabulary_map.get(context_left[j]) is not None:
                            column_index = self.vocabulary_map[context_left[j]]
                            # update co-occurrence count
                            count = .0
                            weight = 1.
                            if decreasing_weighting:
                                weight = len(context_left) - j
                            if (row_index, column_index) in self.X:
                                count = self.X[row_index, column_index]
                            self.X[row_index, column_index] = count + 1 / weight
                    # scan right context
                    context_right = words[i + 1:end + 1]
                    for j in range(0, len(context_right)):
                        if self.vocabulary_map.get(context_right[j]) is not None:
                            column_index = self.vocabulary_map[context_right[j]]
                            # update co-occurrence count
                            count = .0
                            weight = 1.
                            if decreasing_weighting:
                                weight = j + 1
                            if (row_index, column_index) in self.X:
                                count = self.X[row_index, column_index]
                            self.X[row_index, column_index] = count + 1. / weight
        print('      Number of non-zero entries: %d (%f)' % (self.X.getnnz(), self.X.getnnz() / len(self.vocabulary)**2))
