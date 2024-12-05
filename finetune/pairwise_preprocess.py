import mdtraj as md
import numpy as np
import os
from tqdm import tqdm
import argparse

def preprocess(folder_path,prompt_save_path,prompt_all = True):
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
        chain_B_atoms = topology.select("chainid == 1")  # 通常 PDB 文件中的链从 0 开始编号，这里假设链B的编号为1
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
    
def parse():
    parser = argparse.ArgumentParser(description='Generate the prompt of each pdb')
    return parser.parse_known_args()

if __name__ == "__main__":
    preprocess(folder_path = "/data/private/jdp/PepGLAD/datasets/train_valid/pdbs",prompt_save_path='/data/private/jdp/PepGLAD/datasets/train_valid/processed')