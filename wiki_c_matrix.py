# coding: utf-8
from structure.corpus import Corpus, WikiSentences
import pickle
import argparse

__authors__ = "Hussein AL-NATSHEH"
__emails__ = "hussein.al-natsheh@ish-lyon.cnrs.fr"

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--corpus", default='../wikiextractor/output/', type=str) #path to wikipedia parsed corpus
	parser.add_argument("--window_size", default=120, type=int)
	parser.add_argument("--vocab_size", default=120000, type=int)
	parser.add_argument("--output_file", default='wiki_co_occurence_matrix.pickle', type=str)

	args = parser.parse_args()
	corpus = args.corpus
	window_size = args.window_size
	vocab_size = args.vocab_size
	output_file = args.output_file

	sentences = WikiSentences(corpus)
	wiki_co_occurence_matrix = Corpus(sentences, max_nb_features=vocab_size, window_size=window_size, )
	pickle.dump( wiki_co_occurence_matrix, open(output_file,'wb'))
