from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import Bio.PDB
import numpy as np
import json
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Load model and tokenizer
cache_dir = "/data/private/jdp/esm2"
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D", cache_dir=cache_dir)
model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D", cache_dir=cache_dir)

# Function to extract amino acid embeddings and optionally compute adjacency matrix
def embed_peptide(pdb_file, peptide_idx, peptide_sequence=None,include_feature2=False):
    if peptide_sequence is None:
        parser = Bio.PDB.PDBParser(QUIET=True)
        structure = parser.get_structure("peptide", pdb_file)
        chain = structure[0][peptide_idx]

        # Extract sequence from PDB
        residues = [res for res in chain.get_residues() if Bio.PDB.is_aa(res, standard=True)]
        sequence = "".join([res.get_resname() for res in residues])
        peptide_sequence = sequence
    # Tokenize sequence
    inputs = tokenizer(peptide_sequence, return_tensors="pt", max_length=22,padding='max_length', truncation=True)

    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states[-1]  # Take the last hidden state

    # Remove special tokens ([CLS] and [SEP]) embeddings and apply mean pooling
    embedding = hidden_states[0,1:-1,:].view(-1)

    if embedding.shape[0] != 1280*20:
        breakpoint()

    if not include_feature2:
        return embedding

    # Compute adjacency matrix
    coordinates = [res["CA"].get_vector().get_array() for res in residues if "CA" in res]
    coordinates = np.array(coordinates)
    distances = np.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=-1)

    # Flatten adjacency matrix
    flattened_distances = distances.flatten()

    # Concatenate features
    features = torch.cat([embedding, torch.tensor(flattened_distances).unsqueeze(1)], dim=0)

    return features

def generation_embed(directory):
    emb_list = []
    with open(directory, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            # 跳过空行
            if line.strip():
                # 将每行解析为 JSON 对象
                json_object = json.loads(line)
                peptide_path = json_object['gen_pdb']
                peptide_sequence = json_object['gen_seq']
                feature = embed_peptide(peptide_path,peptide_sequence=peptide_sequence,peptide_idx=json_object['lig_chain'], include_feature2=False)
                emb_list.append(feature)
    tensor = torch.stack(emb_list)

    return tensor.numpy()

def gt_embed():
    emb_list = []
    file_path = '/data/private/jdp/PepGLAD/datasets/LNR/test.txt'
    df = pd.read_csv(file_path, sep='\t', header=None, names=['index', 'protein_id', 'peptide_id', 'label'])
    df.set_index('index', inplace=True)

    # 筛选 label 为 0 的行
    df = df[df['label'] == 0]
    for idx in tqdm(df.index):
        peptide_path = os.path.join('/data/private/jdp/PepGLAD/datasets/LNR/pdbs', idx + '.pdb')
        peptide_id = df.loc[idx, 'peptide_id']
        feature = embed_peptide(peptide_path,peptide_idx=peptide_id,peptide_sequence=None, include_feature2=False)
        emb_list.append(feature)
    tensor = torch.stack(emb_list)

    return tensor.numpy()

if __name__ == '__main__':
    plt.figure(figsize=(8, 6))
    
    generation1_array = generation_embed('/home/jiangdapeng/PepGLAD/cluster_cache/condition1.jsonl')
    generation2_array = generation_embed('/home/jiangdapeng/PepGLAD/cluster_cache/condition2.jsonl')
    generation3_array = generation_embed('/home/jiangdapeng/PepGLAD/cluster_cache/condition3.jsonl')
    generation4_array = generation_embed('/home/jiangdapeng/PepGLAD/cluster_cache/condition4.jsonl')
    gt_array = gt_embed()
    

    
    # data = np.vstack((generation1_array, gt_array))

    # pca = PCA(n_components=2)
    # data_2d = pca.fit_transform(data)

    # # 3. 分开 A 和 B 的降维结果
    # g1_2d = data_2d[:generation1_array.shape[0]]  # A 的降维结果
    # gt_2d = data_2d[generation1_array.shape[0]:]  # B 的降维结果

    # plt.scatter(g1_2d[:, 0], g1_2d[:, 1], c='blue', label='generation Samples', alpha=0.7, edgecolors='k')
    # plt.scatter(gt_2d[:, 0], gt_2d[:, 1], c='red', label='gt Samples', alpha=0.7, edgecolors='k')
    # plt.title("2D PCA Visualization")
    # plt.xlabel("Principal Component 1")
    # plt.ylabel("Principal Component 2")
    # plt.grid(True)
    # plt.show()
    # plt.savefig("./pca_visualization.pdf", dpi=300, bbox_inches='tight') 

    # 合并数据
    data = np.vstack((generation1_array,generation2_array,generation3_array,generation4_array,gt_array))
    labels = np.array([0] * generation1_array.shape[0]+[1] * generation2_array.shape[0]+ [2] * generation3_array.shape[0]+[3]*generation4_array.shape[0]+[4] * gt_array.shape[0])  # 标签

    # T-SNE 降维
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    data_2d = tsne.fit_transform(data)

    # 可视化
    plt.figure(figsize=(8, 6))
    plt.scatter(data_2d[labels == 0, 0], data_2d[labels == 0, 1], c='blue', label='Stapled peptide', alpha=0.7)
    plt.scatter(data_2d[labels == 1, 0], data_2d[labels == 1, 1], c='green', label='Head-to-tail peptide', alpha=0.7)
    plt.scatter(data_2d[labels == 2, 0], data_2d[labels == 2, 1], c='purple', label='Disulfide peptide', alpha=0.7)
    plt.scatter(data_2d[labels == 3, 0], data_2d[labels == 3, 1], c='pink', label='Bicycle peptide', alpha=0.7)
    plt.scatter(data_2d[labels == 4, 0], data_2d[labels == 4, 1], c='red', label='GT Samples', alpha=0.7)
    # plt.title("T-SNE Visualization")
    # plt.xlabel("TSNE Component 1")
    # plt.ylabel("TSNE Component 2")
    plt.legend()
    # plt.grid(True)
    plt.show()
    plt.xticks([])  # 隐藏x轴刻度
    plt.yticks([])  # 隐藏y轴刻度
    plt.savefig("./TSNE_visualization.pdf", dpi=500, bbox_inches='tight') 
    plt.savefig("./TSNE_visualization.png", dpi=500, bbox_inches='tight') 


