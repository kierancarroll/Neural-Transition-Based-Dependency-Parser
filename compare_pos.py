#!/usr/bin/env python3
#Compare parser predictions with vs. without POS features on specific sentences.

import sys
import torch
import utils.parser_utils as pu
from parser_model import ParserModel
from utils.parser_utils import load_and_preprocess_data

SENT_1 = {
    'word': ['My', 'back', 'is', 'still', 'in', 'knots', 'and', 'my',
             'hands', 'are', 'still', 'shaking', '.'],
    'pos':  ['PRP$', 'NN', 'VBZ', 'RB', 'IN', 'NNS', 'CC', 'PRP$',
             'NNS', 'VBP', 'RB', 'VBG', '.'],
    'head': [2, 6, 6, 6, 6, 0, 6, 9, 12, 12, 12, 6, 6],
    'label': ['nmod:poss', 'nsubj', 'cop', 'advmod', 'case', 'root',
              'cc', 'nmod:poss', 'nsubj', 'aux', 'advmod', 'conj', 'punct'],
}
SENT_2 = {
    'word': ['The', 'ACLU', 'and', 'worker', 'organizations', 'back',
             'tighter', 'laws', ',', 'but', 'employers', 'and', 'device',
             'manufacturers', 'object', '.'],
    'pos':  ['DT', 'NNP', 'CC', 'NN', 'NNS', 'VBP', 'JJR', 'NNS', ',',
             'CC', 'NNS', 'CC', 'NN', 'NNS', 'VBP', '.'],
    'head': [2, 6, 2, 5, 2, 0, 8, 6, 6, 6, 15, 11, 14, 11, 6, 6],
    'label': ['det', 'nsubj', 'cc', 'compound', 'conj', 'root', 'amod',
              'dobj', 'punct', 'cc', 'nsubj', 'cc', 'compound', 'conj',
              'conj', 'punct'],
}


def run_model(use_pos, model_path, raw_sentences):
    pu.Config.use_pos = use_pos
    parser, embeddings, _, _, test_data = load_and_preprocess_data(reduced=False)
    parser.model = ParserModel(embeddings, n_features=parser.n_features)
    parser.model.load_state_dict(torch.load(model_path))
    parser.model.eval()
    print(f"Evaluating on test set (use_pos={use_pos})")
    test_UAS, _ = parser.parse(test_data)
    print(f"Test UAS: {test_UAS*100}")

    vectorized = parser.vectorize(raw_sentences)
    _, deps = parser.parse(vectorized)
    return deps


def heads_from_deps(deps, n_words):
    head = [-1] * (n_words + 1)
    for h, t in deps:
        head[t] = h
    return head


def print_comparison(sent, with_pos_deps, no_pos_deps):
    words = ['ROOT'] + sent['word']
    n = len(sent['word'])
    wp_head = heads_from_deps(with_pos_deps, n)
    np_head = heads_from_deps(no_pos_deps, n)
    gold_head = [-1] + sent['head']

    print("Sentence:", ' '.join(sent['word']))
    print(f"{'idx'}{'word'}{'gold'}{'with_POS'}{'no_POS'}{'flags'}")
    for t in range(1, n + 1):
        g = words[gold_head[t]] if gold_head[t] >= 0 else '?'
        wp = words[wp_head[t]] if 0 <= wp_head[t] < len(words) else '?'
        npos = words[np_head[t]] if 0 <= np_head[t] < len(words) else '?'
        flag = ''
        if wp != g: flag += ' WP-WRONG'
        if npos != g: flag += ' NP-WRONG'
        print(f"{t}{words[t]}{g}{wp}{npos}{flag}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(1)
    with_pos_path, no_pos_path = sys.argv[1], sys.argv[2]

    print("Running WITH POS features...")
    with_pos_deps = run_model(True, with_pos_path, [SENT_1, SENT_2])

    print("Running WITHOUT POS features...")
    no_pos_deps = run_model(False, no_pos_path, [SENT_1, SENT_2])

    print("COMPARISON")
    for i, sent in enumerate([SENT_1, SENT_2]):
        print(f"Sentence {i+1}")
        print_comparison(sent, with_pos_deps[i], no_pos_deps[i])