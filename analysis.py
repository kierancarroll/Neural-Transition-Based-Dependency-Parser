#!/usr/bin/env python3
import sys
from collections import defaultdict
import torch
from parser_model import ParserModel
from utils.parser_utils import load_and_preprocess_data, L_PREFIX, P_PREFIX


def main(model_path):
    parser, embeddings, _, _, test_data = load_and_preprocess_data(reduced=False)
    parser.model = ParserModel(embeddings)
    parser.model.load_state_dict(torch.load(model_path))
    parser.model.eval()

    print("Parsing test set")
    UAS, dependencies = parser.parse(test_data)
    print(f"Overall test UAS: {UAS * 100}%")

    label_total, label_correct = defaultdict(int), defaultdict(int)
    length_total, length_correct = defaultdict(int), defaultdict(int)
    dist_total, dist_correct = defaultdict(int), defaultdict(int)
    pp_errors, all_errors = [], []

    def length_bucket(n):
        for lo, hi, name in [(1,10,"1-10"), (11,20,"11-20"),
                             (21,30,"21-30"), (31,40,"31-40")]:
            if lo <= n <= hi: return name
        return "41+"

    def dist_bucket(d):
        if d == 1: return "1"
        if d <= 3: return "2-3"
        if d <= 6: return "4-6"
        return "7+"

    for i, ex in enumerate(test_data):
        n_words = len(ex['word']) - 1
        head_pred = [-1] * len(ex['word'])
        for h, t in dependencies[i]:
            head_pred[t] = h
        words = [parser.id2tok.get(w, '?') for w in ex['word']]
        words[0] = "ROOT"
        lb = length_bucket(n_words)

        for t in range(1, len(ex['word'])):
            pred_h, gold_h = head_pred[t], ex['head'][t]
            gold_l_id, pos_id = ex['label'][t], ex['pos'][t]

            label = (parser.id2tok[gold_l_id][len(L_PREFIX):]
                     if gold_l_id in parser.id2tok else "<unk>")
            pos_str = parser.id2tok[pos_id][len(P_PREFIX):]

            correct = (pred_h == gold_h)
            db = dist_bucket(abs(gold_h - t))

            label_total[label] += 1
            length_total[lb] += 1
            dist_total[db] += 1
            if correct:
                label_correct[label] += 1
                length_correct[lb] += 1
                dist_correct[db] += 1
            else:
                err = {
                    'sent': ' '.join(words[1:]),
                    'word': words[t], 't': t,
                    'gold_h_word': words[gold_h] if 0 <= gold_h < len(words) else 'N/A',
                    'gold_h': gold_h,
                    'pred_h_word': words[pred_h] if 0 <= pred_h < len(words) else 'N/A',
                    'pred_h': pred_h,
                    'label': label, 'pos': pos_str,
                    'severity': abs(pred_h - gold_h) if pred_h >= 0 else 999,
                }
                if label in ('nmod', 'case', 'acl', 'advmod') or pos_str == 'IN':
                    pp_errors.append(err)
                all_errors.append(err)

    # per pos tag label
    print("UAS by gold dependency label (labels with >= 20 occurrences)")
    print(f"{'Label'}{'Count'}{'Correct'}{'UAS%'}")
    rows = sorted(
        [(l, label_total[l], label_correct[l],
          100.0 * label_correct[l] / label_total[l])
         for l in label_total if label_total[l] >= 20],
        key=lambda r: -r[1])
    for label, n, c, uas in rows:
        print(f"{label}{n}{c}{uas}")

    # per sentence length
    print("UAS by sentence length")
    for b in ["1-10","11-20","21-30","31-40","41+"]:
        if length_total[b]:
            print(f"Length {b}  n={length_total[b]}"
                  f"UAS={100.0*length_correct[b]/length_total[b]}")

    # per dependency distance
    print("UAS by dependency distance |gold_head_idx - dependent_idx|")
    for b in ["1","2-3","4-6","7+"]:
        if dist_total[b]:
            print(f"Distance {b}  n={dist_total[b]}"
                  f"UAS={100.0*dist_correct[b]/dist_total[b]}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(1)
    main(sys.argv[1])