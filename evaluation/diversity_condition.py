import torch
from Bio import PDB
from diversity import seq_diversity,struct_diversity
from tqdm import tqdm
import json
import os 
import pandas as pd
import math
import numpy as np
from Bio.SeqUtils import seq1

def read_peptide_pdb(pdb_file,peptide_id):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)

    ca_coords = []
    
    chain = structure[0][peptide_id]
    seq = str()
    for residue in chain:  # 遍历所有残基
        if PDB.is_aa(residue):  # 仅提取氨基酸（忽略水分子和其他杂质）
            three_letter_code = residue.get_resname()  # 获取三字母代码
            seq+=seq1(three_letter_code)  # 转换为一字母代码
    for residue in chain:  # 遍历残基
        if 'CA' in residue:  # 仅提取 alpha Carbon
            ca_atom = residue['CA']
            ca_coords.append(ca_atom.coord)

    ca_tensor = torch.tensor(ca_coords, dtype=torch.float32)  # 转换为 PyTorch Tensor
    return ca_tensor,seq

if __name__ == '__main__':
    file_path = '/data/private/jdp/PepGLAD/datasets/LNR/test.txt'
    df = pd.read_csv(file_path, sep='\t', header=None, names=['index', 'protein_id', 'peptide_id', 'label'])
    df.set_index('index', inplace=True)

    # 筛选 label 为 0 的行
    df = df[df['label'] == 0]
    strcut_div_list = []
    seq_div_list = []
    for idx in tqdm(df.index):
        peptide_struct_list = []
        peptide_seq_list = []
        peptide_path_root = os.path.join('/data/private/jdp/PepGLAD/results/condition2_w4_40samples/candidates', idx)
        if not os.path.exists(peptide_path_root):
            continue
        peptide_id = df.loc[idx, 'peptide_id']
        for i in range(40):
            peptide_path = os.path.join(peptide_path_root, idx + '_gen_' + str(i)+'.pdb')
            peptide_tensor,peptide_seq = read_peptide_pdb(peptide_path,peptide_id)
            peptide_struct_list.append(peptide_tensor)
            peptide_seq_list.append(peptide_seq)
        peptide_tensor = torch.stack(peptide_struct_list,dim=0)
        strcut_div_list.append(struct_diversity(peptide_tensor.numpy())[0])
        seq_div_list.append(seq_diversity(peptide_seq_list,0.45)[0])
    
    print(f'The structure diversity is {np.mean(strcut_div_list)}')
    print(f'The sequence diversity is {np.mean(seq_div_list)}')
               