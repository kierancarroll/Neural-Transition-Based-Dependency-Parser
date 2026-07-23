#!/usr/bin/env python3
import sys
import torch
from parser_model import ParserModel
from utils.parser_utils import load_and_preprocess_data


TARGETS = [
    ("freeze on physician fees next year",  "fees"),
]

GOLD_CONLL = "./data/test.conll" 


def load_gold_sentences(path):
    sentences, current = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                if current:
                    sentences.append(current); current = []
                continue
            cols = line.split('\t')
            if len(cols) >= 8 and '-' not in cols[0]:
                current.append((cols[1], cols[4], int(cols[6]), cols[7]))
    if current:
        sentences.append(current)
    return sentences


def main(model_path):
    parser, embeddings, _, _, test_data = load_and_preprocess_data(reduced=False)
    parser.model = ParserModel(embeddings)
    parser.model.load_state_dict(torch.load(model_path))
    parser.model.eval()

    gold_sentences = load_gold_sentences(GOLD_CONLL)
    assert len(gold_sentences) == len(test_data), \
        f"mismatch: {len(gold_sentences)} gold vs {len(test_data)} loaded"

    print("Parsing test set")
    _, dependencies = parser.parse(test_data)

    for substring, target_word in TARGETS:
        for sent_idx, sent in enumerate(gold_sentences):
            sentence_str = " ".join(w for w, _, _, _ in sent)
            if substring not in sentence_str:
                continue
            target_idx = None
            for j, (w, _, _, _) in enumerate(sent):
                if w.lower() == target_word.lower():
                    target_idx = j + 1   
                    break
            if target_idx is None:
                continue

            gold_h = sent[target_idx - 1][2]  
            pred_head = [-1] * (len(sent) + 1)
            for h, t in dependencies[sent_idx]:
                pred_head[t] = h
            pred_h = pred_head[target_idx]

            def name(idx):
                if idx == 0: return "ROOT"
                return sent[idx - 1][0] if 1 <= idx <= len(sent) else f"?({idx})"

            print(f"Sentence: {sentence_str}")
            print(f"Dependent: '{target_word}' (index {target_idx})")
            print(f"Gold head: '{name(gold_h)}' (index {gold_h})")
            print(f"Pred head: '{name(pred_h)}' (index {pred_h})")
            print(f"{'AGREE' if gold_h == pred_h else 'DISAGREE'}")
            break
        else:
            print(f"(could not find sentence containing '{substring}')")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(1)
    main(sys.argv[1])