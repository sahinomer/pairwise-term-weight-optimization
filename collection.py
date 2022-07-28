from argparse import ArgumentParser
from collections import Counter

from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
import tables as tb

from msmarco import get_queries, get_qrels, get_results, get_collections
from query import Query
from utils import extract_phrase, print_progress_bar


class Collection:

    def __init__(self, *query_paths, collection_path=None,
                 max_phrase_length=1, save_path='collection_data'):

        self._vocabulary = dict()
        self._term_count = None
        self._document_count = dict()
        self._document_length = dict()
        self._average_document_length = 0

        self.max_phrase_length = max_phrase_length

        self.exception_list = list()

        self.save_path = save_path
        if collection_path:
            Path(self.save_path).mkdir(parents=True, exist_ok=True)
            self._initialize_vocabulary(query_paths=query_paths)
            self._count_terms_documents(collection_path=collection_path)
            self.save(path=save_path)
        else:
            self.load(path=save_path)

    @property
    def vocabulary(self):
        return self._vocabulary

    @property
    def number_of_document(self):
        return len(self._document_length)

    @property
    def average_document_length(self):
        if self._average_document_length == 0:
            for doc_len in self._document_length.values():
                self._average_document_length += doc_len
            self._average_document_length = self._average_document_length / self.number_of_document

        return self._average_document_length

    def iterate_query(self, queries, triplet_list, bm25_k1=0.82, bm25_b=0.68):

        for qid, relevant_pid_list, irrelevant_pid_list in triplet_list:

            query = Query(qid=qid, query=queries[qid], phrase_length=self.max_phrase_length)

            # Unique passages in the query triplets
            unique_pid_list = np.array(relevant_pid_list + irrelevant_pid_list, dtype=np.int)

            # Stemmed query terms
            stemmed_query_terms = extract_phrase(queries[qid], max_phrase_length=self.max_phrase_length, stem=True)

            try:
                bm25_score_map = self._bm25_score(term_list=stemmed_query_terms, pid_index=unique_pid_list,
                                                  k1=bm25_k1, b=bm25_b)

                for relevant_pid in relevant_pid_list:
                    for irrelevant_pid in irrelevant_pid_list:
                        query.add_sample(relevant=bm25_score_map[relevant_pid],
                                         irrelevant=bm25_score_map[irrelevant_pid])

                yield query
            except IndexError as error:
                self.exception_list.append(f'qid:{qid} - {error}')

    def _idf(self, term):
        document_count = self._document_count[term]
        return np.log(((self.number_of_document - document_count + 0.5) / (document_count + 0.5)) + 1)

    def _bm25_score(self, term_list, pid_index, k1=0.82, b=0.68):

        term_index = [self._vocabulary[term] for term in term_list]
        document_length_vector = np.array([self._document_length[str(pid)] for pid in pid_index])

        idf = np.zeros(len(term_list))
        for i, term in enumerate(term_list):
            for token in term.split():
                idf[i] += self._idf(term=token)

        term_frequency_vector = self._term_count[pid_index, :][:, term_index]

        document_norm = k1 * (1 - b + b * document_length_vector / self.average_document_length)

        tf = term_frequency_vector / (term_frequency_vector.T + document_norm).T

        bm25 = np.multiply(tf, idf)

        score_map = dict()
        for pid, vector in zip(pid_index, bm25):
            score_map[str(pid)] = vector.A1

        return score_map

    def _initialize_vocabulary(self, query_paths):
        vocabulary_set = set()
        for path in query_paths:
            queries = get_queries(path=path)
            for i, (qid, query) in enumerate(queries.items()):
                phrases = extract_phrase(query, max_phrase_length=self.max_phrase_length, stem=True, as_set=True)

                vocabulary_set.update(phrases)

                print_progress_bar(i, n=len(queries), text=f'Parse queries: {path}')

        for i, term in enumerate(sorted(vocabulary_set)):
            self._vocabulary[term] = i
            self._document_count[term] = 0
            print_progress_bar(i, n=len(vocabulary_set), text=f'Initialize vocabulary')

    def _count_terms_documents(self, collection_path):

        file = tb.open_file(f'{self.save_path}/term_count.h5', 'w')
        pid_index = file.create_earray(file.root, 'pid_index', tb.Int32Atom(), shape=(0,))
        term_index = file.create_earray(file.root, 'term_index', tb.Int32Atom(), shape=(0,))
        term_count = file.create_earray(file.root, 'term_count', tb.Int32Atom(), shape=(0,))

        collection = get_collections(path=collection_path)

        for i, (pid, passage) in enumerate(collection.items()):

            passage_phrases = extract_phrase(text=passage, max_phrase_length=self.max_phrase_length, stem=True)

            self._document_length[pid] = len(passage_phrases)

            pid_list = list()
            term_list = list()
            count_list = list()
            phrase_counts = Counter(passage_phrases)
            for term, count in phrase_counts.items():
                try:
                    self._document_count[term] += 1
                    pid_list.append(int(pid))
                    term_list.append(self._vocabulary[term])
                    count_list.append(count)
                except KeyError:
                    pass

            pid_index.append(pid_list)
            term_index.append(term_list)
            term_count.append(count_list)

            print_progress_bar(i, n=len(collection), text=f'Count terms and documents')

        file.close()

    def save(self, path):
        np.savez(f'{path}/msmarco_passage',
                 vocabulary=self._vocabulary,
                 document_count=self._document_count,
                 document_length=self._document_length)

    def load(self, path):
        data = np.load(f'{path}/msmarco_passage.npz', allow_pickle=True)
        self._vocabulary = data['vocabulary'].item()
        self._document_count = data['document_count'].item()
        self._document_length = data['document_length'].item()

        with tb.open_file(f'{path}/term_count.h5', 'r') as h5:
            self._term_count = csr_matrix((h5.root.term_count[:], (h5.root.pid_index[:], h5.root.term_index[:])))

    @staticmethod
    def create_triplets(queries, qrel_path, result_path, exclude_set=None):

        qrels = get_qrels(path=qrel_path)
        results = get_results(path=result_path)

        triplet_list = list()
        for i, qid in enumerate(queries.keys()):

            if exclude_set and qid in exclude_set:
                continue

            if qid not in results:
                continue

            relevant_pids = qrels[qid]
            irrelevant_pids = [pid for pid in results[qid] if pid not in relevant_pids]

            triplet_list.append((qid, relevant_pids, irrelevant_pids))

            print_progress_bar(i=i, n=len(queries), text='Create triplets...')

        return triplet_list


if __name__ == '__main__':

    parser = ArgumentParser(description='Index MS-MARCO collection')

    parser.add_argument('queries', metavar='N', type=str, nargs='+',
                        help='Path of queries')
    parser.add_argument('--collection_path', type=str, help='Path of collection')
    parser.add_argument('--max_phrase_length', default=1, type=int, help='Max length of a phrase')
    parser.add_argument('--save_path', default='collection_data', type=str, help='Save path of collection')

    args = parser.parse_args()

    Collection(*args.queries, collection_path=args.collection_path,
               max_phrase_length=args.max_phrase_length, save_path=args.save_path)
