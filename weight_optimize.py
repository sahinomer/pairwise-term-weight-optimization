import json
from argparse import ArgumentParser
from multiprocessing import cpu_count, Pool

from msmarco import get_queries
from collection import Collection
from weight_model import WeightModel
from utils import print_progress_bar


def model_worker(query):
    model = WeightModel(sequence_length=query.sequence_length, constraint='non-neg', use_tf=False)
    model.train(relevant=query.relevant_features, irrelevant=query.irrelevant_features)
    query.weights = model.get_weights(norm=None)
    return query


def train_model(collection, query_path, qrel_path, result_path, output_path, out_mode='w'):
    # Exclude trained queries
    exclude_set = set()
    if out_mode.startswith('a'):
        try:
            with open(output_path, mode='r', encoding='utf8') as query_file:
                for query_line in query_file:
                    exclude_qid = json.loads(query_line)['qid']
                    exclude_set.add(exclude_qid)
        except FileNotFoundError:
            out_mode = 'w'

    queries = get_queries(path=query_path)

    triplet_list = collection.create_triplets(queries=queries, qrel_path=qrel_path, result_path=result_path,
                                              exclude_set=exclude_set)

    number_of_query = len(triplet_list)
    with open(output_path, mode=out_mode, encoding='utf8') as query_file:
        with Pool(cpu_count()) as pool:
            for i, query in enumerate(pool.imap_unordered(model_worker,
                                                          collection.iterate_query(queries, triplet_list))):

                try:
                    query.to_file(output=query_file)
                except IndexError:
                    continue

                print_progress_bar(i, number_of_query, text='Pair-wise model training...')


def check_queries(collection, query_path, qrel_path, result_path):
    queries = get_queries(path=query_path)

    triplet_list = collection.create_triplets(queries=queries, qrel_path=qrel_path, result_path=result_path)
    number_of_query = len(triplet_list)
    for i, _ in enumerate(collection.iterate_query(queries, triplet_list)):
        print_progress_bar(i, number_of_query, text='Check queries...')


if __name__ == '__main__':

    parser = ArgumentParser(description='Optimize term weights')

    parser.add_argument('--collection', type=str, help='Directory of indexed collection')
    parser.add_argument('--query_path', type=str, help='Path of queries')
    parser.add_argument('--qrel_path', type=str, help='Path of query-relations')
    parser.add_argument('--result_path', type=str, help='Path of ranking results')
    parser.add_argument('--output_path', type=str, help='Path of output (.json or .tsv)')
    parser.add_argument('--max_phrase_length', default=1, type=int, help='Max length of a phrase')

    args = parser.parse_args()

    collection_statistic = Collection(save_path=args.collection, max_phrase_length=args.max_phrase_length)

    train_model(collection=collection_statistic,
                query_path=args.query_path,
                qrel_path=args.qrel_path,
                result_path=args.result_path,
                output_path=args.output_path)

    print(f'\nNumber of exceptions: {len(collection_statistic.exception_list)}')
    for exception in collection_statistic.exception_list:
        print('  ', exception)
