import json
from argparse import ArgumentParser

from utils import extract_phrase


def iter_json_file(path):
    with open(path, mode='r', encoding='utf8') as weight_file:
        for query_line in weight_file:
            yield json.loads(query_line)


def weights_to_query(weighted_terms, weighted_query, boost_weight, max_phrase_length=None,
                     precision='.10f', fuzzy=False):
    with open(weighted_query, mode='w', encoding='utf8') as query_file:
        for query in iter_json_file(path=weighted_terms):

            term_list = []
            for term in extract_phrase(text=query['query'], max_phrase_length=max_phrase_length):

                weight = max(query[boost_weight][term], 0)

                window_size = len(term.split(' '))
                if window_size == 1:
                    term_list.append(f'{term}^{weight:{precision}}')

                elif weight > 0:
                    if fuzzy:
                        term_list.append(f'"{term}"~{window_size}^{weight:{precision}}')
                    else:
                        term_list.append(f'"{term}"^{weight:{precision}}')

            query_file.write(query['qid'] + '\t' + ' '.join(term_list) + '\n')


if __name__ == '__main__':

    parser = ArgumentParser(description='Convert weighted terms (.json) to query (.tsv)')

    parser.add_argument('-i', '--input', type=str, help='Path of weighted terms (.json)')
    parser.add_argument('-o', '--output', type=str, help='Path of weighted query (.tsv)')

    parser.add_argument('--boost_weight', default='term_weight', type=str, help='Key of boosting weight')
    parser.add_argument('--max_phrase_length', default=1, type=int, help='Max length of a phrase')
    parser.add_argument('--fuzzy', action='store_true', help='Fuzzy terms')

    args = parser.parse_args()

    weights_to_query(weighted_terms=args.input, weighted_query=args.output,
                     boost_weight=args.boost_weight, max_phrase_length=args.max_phrase_length, fuzzy=args.fuzzy)
