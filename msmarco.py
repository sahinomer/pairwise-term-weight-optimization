
import csv
import glob
import json


def iter_collections(directory):
    file_paths = glob.glob(f'{directory}/*.json')
    for path in file_paths:
        with open(path, mode='r', encoding='utf8') as file:
            for line in file:
                passage = json.loads(line)
                yield passage['id'], passage['contents']


def get_collections(path, limit=0):
    collections = dict()
    with open(path, mode='r', encoding='utf8') as file:
        tsvreader = csv.reader(file, delimiter="\t")
        for [pid, passage] in tsvreader:
            collections[pid] = passage

            limit -= 1
            if limit == 0:
                break

    return collections


def get_queries(path):
    queries = dict()
    with open(path, mode='r', encoding='utf8') as file:
        tsvreader = csv.reader(file, delimiter="\t")
        for [qid, query] in tsvreader:
            queries[qid] = query

    return queries


def get_qrels(path):
    qrels = dict()
    with open(path, mode='r', encoding='utf8') as file:
        tsvreader = csv.reader(file, delimiter="\t")
        for [qid, _, pid, rel] in tsvreader:
            assert rel == "1"
            if qid not in qrels:
                qrels[qid] = list()
            qrels[qid].append(pid)

    return qrels


def get_results(path):
    results = dict()
    with open(path, mode='r', encoding='utf8') as file:
        tsvreader = csv.reader(file, delimiter="\t")
        for [qid, pid, rank] in tsvreader:
            if qid not in results:
                results[qid] = list()
            results[qid].append(pid)

    return results
