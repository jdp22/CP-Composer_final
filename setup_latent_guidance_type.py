#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import yaml
import argparse
from tqdm import tqdm
import math

from data.format import VOCAB

import torch

from generate import get_best_ckpt, to_device
from data import create_dataloader, create_dataset

amino_acid_map = {
    'A': 1,  # Alanine
    'R': 2,  # Arginine
    'N': 3,  # Asparagine
    'D': 4,  # Aspartic acid
    'C': 5,  # Cysteine
    'E': 6,  # Glutamic acid
    'Q': 7,  # Glutamine
    'G': 8,  # Glycine
    'H': 9,  # Histidine
    'I': 10, # Isoleucine
    'L': 11, # Leucine
    'K': 12, # Lysine
    'M': 13, # Methionine
    'F': 14, # Phenylalanine
    'P': 15, # Proline
    'S': 16, # Serine
    'T': 17, # Threonine
    'W': 18, # Tryptophan
    'Y': 19, # Tyrosine
    'V': 20  # Valine
}


def main(args):
    config = yaml.safe_load(open(args.config, 'r'))
    # load model
    b_ckpt = args.ckpt if args.ckpt.endswith('.ckpt') else get_best_ckpt(args.ckpt)
    print(f'Using checkpoint {b_ckpt}')
    model = torch.load(b_ckpt, map_location='cpu')
    device = torch.device('cpu' if args.gpu == -1 else f'cuda:{args.gpu}')
    model.to(device)
    model.eval()
    
    # load data
    _, _, test_set = create_dataset(config['dataset'])
    test_loader = create_dataloader(test_set, config['dataloader'])

    all_aa_dicts = {}

    with torch.no_grad():
        for batch in tqdm(test_loader):
            batch = to_device(batch, device)
            H, Z, _, _ = model.autoencoder.encode(
                batch['X'], batch['S'], batch['mask'], batch['position_ids'],
                batch['lengths'], batch['atom_mask'], no_randomness=True
            )
            aa_type = batch['S'][batch['mask']]
            for idx in aa_type:
                abrv = VOCAB.idx_to_symbol(idx) 
                if abrv not in all_aa_dicts.keys():
                    all_aa_dicts[abrv] = [H[idx]]
                else:
                    all_aa_dicts[abrv] += [H[idx]]     
    for key in all_aa_dicts.keys():
        all_aa_dicts[key] = torch.mean(torch.stack(all_aa_dicts[key]),dim=0).to(H.device)
    model.atom_type_embedding = all_aa_dicts
    torch.save(model, b_ckpt)

def parse():
    parser = argparse.ArgumentParser(description='Calculate distance between consecutive latent points')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    return parser.parse_args()


if __name__ == '__main__':
    main(parse())