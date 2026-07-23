#!/usr/bin/env python3
#Train an ablation parser without POS features (n_features=18 instead of 36).
import os, math, time
from datetime import datetime

import torch
from torch import nn, optim
from tqdm import tqdm

import utils.parser_utils as pu
pu.Config.use_pos = False

from parser_model import ParserModel
from utils.parser_utils import minibatches, load_and_preprocess_data, AverageMeter


def train_for_epoch(parser, train_data, dev_data, optimizer, loss_func, bs):
    parser.model.train()
    n_mb = math.ceil(len(train_data) / bs)
    loss_meter = AverageMeter()
    with tqdm(total=n_mb) as prog:
        for train_x, train_y in minibatches(train_data, bs):
            optimizer.zero_grad()
            train_x = torch.from_numpy(train_x).long()
            train_y = torch.from_numpy(train_y.nonzero()[1]).long()
            logits = parser.model(train_x)
            loss = loss_func(logits, train_y)
            loss.backward()
            optimizer.step()
            prog.update(1)
            loss_meter.update(loss.item())
    print(f"Average Train Loss: {loss_meter.avg:.4f}")
    parser.model.eval()
    dev_UAS, _ = parser.parse(dev_data)
    print(f"- dev UAS (NO POS): {dev_UAS*100:.2f}")
    return dev_UAS


if __name__ == "__main__":
    parser, embeddings, train_data, dev_data, test_data, _ = load_and_preprocess_data(reduced=False)
    print(f"n_features = {parser.n_features}  (should be 18)")

    parser.model = ParserModel(embeddings, n_features=parser.n_features)
    optimizer = optim.Adam(parser.model.parameters(), lr=0.0005)
    loss_func = nn.CrossEntropyLoss()

    output_dir = f"results_no_pos/{datetime.now():%Y%m%d_%H%M%S}/"
    os.makedirs(output_dir, exist_ok=True)
    output_path = output_dir + "model.weights"

    best = 0
    for epoch in range(10):
        print(f"Epoch {epoch+1}/10")
        dev_UAS = train_for_epoch(parser, train_data, dev_data,
                                  optimizer, loss_func, 1024)
        if dev_UAS > best:
            best = dev_UAS
            torch.save(parser.model.state_dict(), output_path)
            print("  -> new best, saved")

    print(f"\nBest dev UAS without POS: {best*100:.2f}")
    print(f"Saved to: {output_path}")