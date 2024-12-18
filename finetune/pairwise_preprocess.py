import mdtraj as md
import numpy as np
import os
from tqdm import tqdm
import argparse
from Bio.PDB import PDBParser
from concurrent.futures import ProcessPoolExecutor
import pandas as pd

def alpha_preprocess(folder_path,prompt_save_path,prompt_all = True):
    '''
    Get the prompt of every pdb file in the folder.
    folder_path: the root path to the orgin folder
    prompt_save_path: the root path to save the generated prompt file
    prompt_all: whether to preprocess all of the pdb files in the folder
    '''
    if prompt_all:
        prompt_path = os.path.join(prompt_save_path,'prompts_all.txt')
    print(f'save the processed prompt file in the path {prompt_path}')
    if os.path.exists(prompt_path):
        os.remove(prompt_path)
    time_bar = tqdm(total=len(os.listdir(folder_path)))
    for file in os.listdir(folder_path):
        time_bar.update(1)
        file_name = os.path.join(folder_path, file)
        traj = md.load(file_name)
        topology = traj.topology
        chain_B_atoms = topology.select("chainid == 0")  # 通常 PDB 文件中的链从 0 开始编号，这里假设链B的编号为1
        # Slice the trajectory to only include chain B
        traj_chain_B = traj.atom_slice(chain_B_atoms)

        # Compute secondary structure for chain B
        pdb_ss = md.compute_dssp(traj_chain_B, simplified=False)[0]  # (L, )

        # Count the secondary structure elements
        ss_count = {
            'alpha': np.sum(pdb_ss == 'H')
        }
        if not prompt_all:
            if ss_count['alpha']==0:
                continue
        pdb_name = os.path.basename(file_name)
        pdb_name = pdb_name.split('.')[0]
        with open(prompt_path, "a") as file:
            file.write(f"{pdb_name}\tThe peptide has {ss_count['alpha']} alpha helices.\n")
    time_bar.close()

def process_pdb_file(pdb_file):
    """
    处理单个PDB文件,计算距离并返回结果。
    """
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('peptide', pdb_file)
        
        # 提取指定链
        chain = structure[0]['L']  # 假设只取第一个模型和指定的链
        
        # 获取链中第一个和最后一个氨基酸
        residues = list(chain.get_residues())
        first_residue = residues[0]
        last_residue = residues[-1]
        
        # 提取N端和C端原子坐标
        first_atom = first_residue['N']  # N端的氮原子
        last_atom = last_residue['C']   # C端的羧基碳原子
        
        # 计算欧几里得距离
        distance = np.linalg.norm(first_atom.coord - last_atom.coord)
        pdb_name = os.path.basename(pdb_file)
        pdb_name = pdb_name.split('.')[0]
        
        return f"{pdb_name}\tThe distance between the N-terminal and C-terminal atoms of the peptide is {distance:.2f} Å.\n"
    except Exception as e:
        return f"Error processing {pdb_file}: {e}\n"

def distance_preprocess_parallel(folder_path, prompt_save_path, prompt_all=True):
    """
    并行处理PDB文件,生成距离提示。
    """
    if prompt_all:
        prompt_path = os.path.join(prompt_save_path, 'prompts_distance.txt')
    print(f'Save the processed prompt file in the path {prompt_path}')
    if os.path.exists(prompt_path):
        os.remove(prompt_path)

    pdb_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path)]
    with open(prompt_path, "a") as prompt_file:
        with ProcessPoolExecutor() as executor:
            # 使用tqdm显示进度条
            with tqdm(total=len(pdb_files)) as time_bar:
                for result in executor.map(process_pdb_file, pdb_files):
                    time_bar.update(1)
                    prompt_file.write(result)

def distance_preprocess(folder_path,prompt_save_path,prompt_all = True):
    '''
    Get the prompt of every pdb file in the folder.
    folder_path: the root path to the orgin pdb folder
    prompt_save_path: the root path to save the generated prompt file
    prompt_all: whether to preprocess all of the pdb files in the folder
    '''
    if prompt_all:
        prompt_path = os.path.join(prompt_save_path,'prompts_distance.txt')
    print(f'save the processed prompt file in the path {prompt_path}')
    index_path = os.path.join(prompt_save_path,'index.txt')
    df = pd.read_csv(index_path, sep = '\t', header=None)
    if os.path.exists(prompt_path):
        os.remove(prompt_path)
    time_bar = tqdm(total=len(os.listdir(folder_path)))
    for row in df.itertuples(index=False):
        file = row[0]
        peptide_index = row[8]
        time_bar.update(1)
        pdb_file = os.path.join(folder_path, file+'.pdb')
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('peptide', pdb_file)
        # 提取指定链
        
        chain = structure[0][peptide_index]  # 假设只取第一个模型和指定的链
        
        # 获取链中第一个和最后一个氨基酸
        residues = list(chain.get_residues())
        first_residue = residues[0]
        last_residue = residues[-1]
        
        # 提取N端和C端原子坐标
        first_atom = first_residue['N']  # N端的氮原子
        last_atom = last_residue['C']   # C端的羧基碳原子
        # 计算欧几里得距离
        distance = np.linalg.norm(first_atom.coord - last_atom.coord)
        pdb_name = os.path.basename(pdb_file)
        pdb_name = pdb_name.split('.')[0]
        with open(prompt_path, "a") as file:
            file.write(f"{pdb_name}\tThe distance between the N-terminal and C-terminal atoms of the peptide is {distance:.2f} Å.\n")
    time_bar.close()

def parse():
    parser = argparse.ArgumentParser(description='Generate the prompt of each pdb')
    return parser.parse_known_args()

if __name__ == "__main__":
    # alpha_preprocess(folder_path = "/data/private/jdp/PepGLAD/datasets/ProtFrag/pdbs",prompt_save_path='/data/private/jdp/PepGLAD/datasets/ProtFrag/processed')
    distance_preprocess(folder_path = "/data/private/jdp/PepGLAD/datasets/train_valid/pdbs",prompt_save_path='/data/private/jdp/PepGLAD/datasets/train_valid/processed')