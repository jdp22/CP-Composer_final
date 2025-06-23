
import os
from typing import Optional, Any

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from utils import register as R
from utils.const import sidechain_atoms

from data.converter.list_blocks_to_pdb import list_blocks_to_pdb

from .format import VOCAB, Block, Atom
from .mmap_dataset import MMAPDataset
from .resample import ClusterResampler
from transformers import AutoTokenizer, AutoModel

# 指定模型名称，例如 "allenai/scibert_scivocab_uncased"
model_name = "allenai/scibert_scivocab_uncased"

# 指定保存路径
save_path = "/data4/private/jdp/scibert"

# 下载并保存模型
model = AutoModel.from_pretrained(save_path)
tokz = AutoTokenizer.from_pretrained(save_path)

model.eval()

# from transformers import AutoTokenizer, AutoModelForCausalLM
# cache_dir = "/data/private/jdp/Qwen2.5-1.5B-Instruct"
# model_name = "Qwen/Qwen2.5-1.5B-Instruct"
# Qw_model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype="auto",
#     device_map="auto",
#     cache_dir = cache_dir
# )
# Qw_tokenizer = AutoTokenizer.from_pretrained(model_name)

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

# amino_acid_map = {
#     'A': 'Alanine',
#     'R': 'Arginine',
#     'N': 'Asparagine',
#     'D': 'Aspartic acid',
#     'C': 'Cysteine',
#     'E': 'Glutamic acid',
#     'Q': 'Glutamine',
#     'G': 'Glycine',
#     'H': 'Histidine',
#     'I': 'Isoleucine',
#     'L': 'Leucine',
#     'K': 'Lysine',
#     'M': 'Methionine',
#     'F': 'Phenylalanine',
#     'P': 'Proline',
#     'S': 'Serine',
#     'T': 'Threonine',
#     'W': 'Tryptophan',
#     'Y': 'Tyrosine',
#     'V': 'Valine'
# }




def calculate_covariance_matrix(point_cloud):
    # Calculate the covariance matrix of the point cloud
    covariance_matrix = np.cov(point_cloud, rowvar=False)
    return covariance_matrix


@R.register('CoDesignDataset')
class CoDesignDataset(MMAPDataset):

    MAX_N_ATOM = 14

    def __init__(
            self,
            mmap_dir: str,
            backbone_only: bool,  # only backbone (N, CA, C, O) or full-atom
            specify_data: Optional[str] = None,
            specify_index: Optional[str] = None,
            padding_collate: bool = False,
            cluster: Optional[str] = None,
            use_covariance_matrix: bool = False
        ) -> None:
        super().__init__(mmap_dir, specify_data, specify_index)
        self.mmap_dir = mmap_dir
        self.backbone_only = backbone_only
        self._lengths = [len(prop[-1].split(',')) + int(prop[1]) for prop in self._properties]
        self.padding_collate = padding_collate
        self.resampler = ClusterResampler(cluster) if cluster else None  # should only be used in training!
        self.use_covariance_matrix = use_covariance_matrix

        self.dynamic_idxs = [i for i in range(len(self))]
        self.update_epoch() # should be called every epoch

    def update_epoch(self):
        if self.resampler is not None:
            self.dynamic_idxs = self.resampler(len(self))

    def get_len(self, idx):
        return self._lengths[self.dynamic_idxs[idx]]

    def get_summary(self, idx: int):
        props = self._properties[idx]
        _id = self._indexes[idx][0].split('.')[0]
        ref_pdb = os.path.join(self.mmap_dir, '..', 'pdbs', _id + '.pdb')
        rec_chain, lig_chain = props[4], props[5]
        return _id, ref_pdb, rec_chain, lig_chain

    def __getitem__(self, idx: int):
        idx = self.dynamic_idxs[idx]
        rec_blocks, lig_blocks = super().__getitem__(idx)
        # receptor, (lig_chain_id, lig_blocks) = super().__getitem__(idx)
        # pocket = {}
        # for i in self._properties[idx][-1].split(','):
        #     chain, i = i.split(':')
        #     if chain not in pocket:
        #         pocket[chain] = []
        #     pocket[chain].append(int(i))
        # rec_blocks = []
        # for chain_id, blocks in receptor:
        #     for i in pocket[chain_id]:
        #         rec_blocks.append(blocks[i])
        pocket_idx = [int(i) for i in self._properties[idx][-1].split(',')]
        rec_position_ids = [i + 1 for i, _ in enumerate(rec_blocks)]
        rec_blocks = [rec_blocks[i] for i in pocket_idx]
        rec_position_ids = [rec_position_ids[i] for i in pocket_idx]
        rec_blocks = [Block.from_tuple(tup) for tup in rec_blocks]
        lig_blocks = [Block.from_tuple(tup) for tup in lig_blocks]

        # for block in lig_blocks:
        #     block.units = [Atom('CA', [0, 0, 0], 'C')]
        # if idx == 0:
        #     print(self._properties[idx])
        #     print(''.join(VOCAB.abrv_to_symbol(block.abrv) for block in lig_blocks))
        #     list_blocks_to_pdb([
        #         rec_blocks, lig_blocks
        #     ], ['B', 'A'], 'pocket.pdb')

        mask = [0 for _ in rec_blocks] + [1 for _ in lig_blocks]
        position_ids = rec_position_ids + [i + 1 for i, _ in enumerate(lig_blocks)]
        X, S, atom_mask = [], [], []
        for block in rec_blocks + lig_blocks:
            symbol = VOCAB.abrv_to_symbol(block.abrv)
            atom2coord = { unit.name: unit.get_coord() for unit in block.units }
            bb_pos = np.mean(list(atom2coord.values()), axis=0).tolist()
            coords, coord_mask = [], []
            for atom_name in VOCAB.backbone_atoms + sidechain_atoms.get(symbol, []):
                if atom_name in atom2coord:
                    coords.append(atom2coord[atom_name])
                    coord_mask.append(1)
                else:
                    coords.append(bb_pos)
                    coord_mask.append(0)
            n_pad = self.MAX_N_ATOM - len(coords)
            for _ in range(n_pad):
                coords.append(bb_pos)
                coord_mask.append(0)

            X.append(coords)
            S.append(VOCAB.symbol_to_idx(symbol))
            atom_mask.append(coord_mask)
        
        X, atom_mask = torch.tensor(X, dtype=torch.float), torch.tensor(atom_mask, dtype=torch.bool)
        mask = torch.tensor(mask, dtype=torch.bool)
        if self.backbone_only:
            X, atom_mask = X[:, :4], atom_mask[:, :4]

        if self.use_covariance_matrix:
            cov = calculate_covariance_matrix(X[~mask][:, 1][atom_mask[~mask][:, 1]].numpy()) # only use the receptor to derive the affine transformation
            eps = 1e-4
            cov = cov + eps * np.identity(cov.shape[0])
            L = torch.from_numpy(np.linalg.cholesky(cov)).float().unsqueeze(0)
        else:
            L = None

        item =  {
            'X': X,                                                         # [N, 14] or [N, 4] if backbone_only == True
            'S': torch.tensor(S, dtype=torch.long),                         # [N]
            'position_ids': torch.tensor(position_ids, dtype=torch.long),   # [N]
            'mask': mask,                                                   # [N], 1 for generation
            'atom_mask': atom_mask,                                         # [N, 14] or [N, 4], 1 for having records in the PDB
            'lengths': len(S),
        }
        if L is not None:
            item['L'] = L
        return item

    def collate_fn(self, batch):
        if self.padding_collate:
            results = {}
            pad_idx = VOCAB.symbol_to_idx(VOCAB.PAD)
            for key in batch[0]:
                values = [item[key] for item in batch]
                if values[0] is None:
                    results[key] = None
                    continue
                if key == 'lengths':
                    results[key] = torch.tensor(values, dtype=torch.long)
                elif key == 'S':
                    results[key] = pad_sequence(values, batch_first=True, padding_value=pad_idx)
                else:
                    results[key] = pad_sequence(values, batch_first=True, padding_value=0)
            return results
        else:
            results = {}
            for key in batch[0]:
                values = [item[key] for item in batch]
                if values[0] is None:
                    results[key] = None
                    continue
                if key == 'lengths':
                    results[key] = torch.tensor(values, dtype=torch.long)
                else:
                    results[key] = torch.cat(values, dim=0)
            return results
        
@R.register('PromptDataset')
class PromptDataset(MMAPDataset):

    MAX_N_ATOM = 14

    def __init__(
            self,
            mmap_dir: str,
            backbone_only: bool,  # only backbone (N, CA, C, O) or full-atom
            specify_data: Optional[str] = None,
            specify_index: Optional[str] = None,
            padding_collate: bool = False,
            cluster: Optional[str] = None,
            use_covariance_matrix: bool = False,
            text_guidance:str =None
        ) -> None:
        super().__init__(mmap_dir, specify_data, specify_index)
        self.mmap_dir = mmap_dir
        self.backbone_only = backbone_only
        self._lengths = [len(prop[-1].split(',')) + int(prop[1]) for prop in self._properties]
        self.padding_collate = padding_collate
        self.resampler = ClusterResampler(cluster) if cluster else None  # should only be used in training!
        self.use_covariance_matrix = use_covariance_matrix

        self.dynamic_idxs = [i for i in range(len(self))]
        self.update_epoch() # should be called every epoch
        self.text_guidance= text_guidance

    def update_epoch(self):
        if self.resampler is not None:
            self.dynamic_idxs = self.resampler(len(self))

    def get_len(self, idx):
        return self._lengths[self.dynamic_idxs[idx]]

    def get_summary(self, idx: int):
        props = self._properties[idx]
        _id = self._indexes[idx][0].split('.')[0]
        ref_pdb = os.path.join(self.mmap_dir, '..', 'pdbs', _id + '.pdb')
        rec_chain, lig_chain = props[4], props[5]
        return _id, ref_pdb, rec_chain, lig_chain

    def __getitem__(self, idx: int):
        idx = self.dynamic_idxs[idx]
        rec_blocks, lig_blocks = super().__getitem__(idx)
        # receptor, (lig_chain_id, lig_blocks) = super().__getitem__(idx)
        # pocket = {}
        # for i in self._properties[idx][-1].split(','):
        #     chain, i = i.split(':')
        #     if chain not in pocket:
        #         pocket[chain] = []
        #     pocket[chain].append(int(i))
        # rec_blocks = []
        # for chain_id, blocks in receptor:
        #     for i in pocket[chain_id]:
        #         rec_blocks.append(blocks[i])
        if self.text_guidance is None:
            pp_idx = -3
        else:
            pp_idx = -1
        try:
            pocket_idx = [int(i) for i in self._properties[idx][pp_idx].split(',')]
        except ValueError as e:
            print(f"Error occurred at idx={idx}")
            print(f"Error occurred at qdb sequence={self._properties[idx]}")
        rec_position_ids = [i + 1 for i, _ in enumerate(rec_blocks)]
        rec_blocks = [rec_blocks[i] for i in pocket_idx]
        rec_position_ids = [rec_position_ids[i] for i in pocket_idx]
        rec_blocks = [Block.from_tuple(tup) for tup in rec_blocks]
        lig_blocks = [Block.from_tuple(tup) for tup in lig_blocks]

        # for block in lig_blocks:
        #     block.units = [Atom('CA', [0, 0, 0], 'C')]
        # if idx == 0:
        #     print(self._properties[idx])
        #     print(''.join(VOCAB.abrv_to_symbol(block.abrv) for block in lig_blocks))
        #     list_blocks_to_pdb([
        #         rec_blocks, lig_blocks
        #     ], ['B', 'A'], 'pocket.pdb')

        mask = [0 for _ in rec_blocks] + [1 for _ in lig_blocks]
        position_ids = rec_position_ids + [i + 1 for i, _ in enumerate(lig_blocks)]
        X, S, atom_mask = [], [], []
        for block in rec_blocks + lig_blocks:
            symbol = VOCAB.abrv_to_symbol(block.abrv)
            atom2coord = { unit.name: unit.get_coord() for unit in block.units }
            bb_pos = np.mean(list(atom2coord.values()), axis=0).tolist()
            coords, coord_mask = [], []
            for atom_name in VOCAB.backbone_atoms + sidechain_atoms.get(symbol, []):
                if atom_name in atom2coord:
                    coords.append(atom2coord[atom_name])
                    coord_mask.append(1)
                else:
                    coords.append(bb_pos)
                    coord_mask.append(0)
            n_pad = self.MAX_N_ATOM - len(coords)
            for _ in range(n_pad):
                coords.append(bb_pos)
                coord_mask.append(0)

            X.append(coords)
            S.append(VOCAB.symbol_to_idx(symbol))
            atom_mask.append(coord_mask)
        
        X, atom_mask = torch.tensor(X, dtype=torch.float), torch.tensor(atom_mask, dtype=torch.bool)
        mask = torch.tensor(mask, dtype=torch.bool)
        if self.backbone_only:
            X, atom_mask = X[:, :4], atom_mask[:, :4]

        if self.use_covariance_matrix:
            cov = calculate_covariance_matrix(X[~mask][:, 1][atom_mask[~mask][:, 1]].numpy()) # only use the receptor to derive the affine transformation
            eps = 1e-4
            cov = cov + eps * np.identity(cov.shape[0])
            L = torch.from_numpy(np.linalg.cholesky(cov)).float().unsqueeze(0)
        else:
            L = None
        
        # Use LLM to encode the text guidance
        if self.text_guidance is None:
            prompt1 = self._properties[idx][-2]
            prompt2 = self._properties[idx][-1]
            atom_sequence = self._properties[idx][-5]
        else:
            # prompt = self.text_guidance
            prompt1 = 'The length between the N-terminal and C-terminal atoms in the peptide is 3.8 Å.'	
            prompt2 = 'The amino acid at the 7th position is Serine.'
            atom_sequence = self._properties[idx][-3]

        atom_embedding_list = []
        for i in range(len(atom_sequence)):
            atom = atom_sequence[i]
            one_hot_vector = torch.zeros(20)
            one_hot_vector[amino_acid_map[atom]-1] = 1
            atom_embedding_list.append(one_hot_vector)
        atom_embedding = torch.stack(atom_embedding_list,dim=0)
        if False:
            one_hot_vector = torch.zeros(20)
            one_hot_vector[amino_acid_map[prompt[0]]-1] = 1
            one_hot_vector = one_hot_vector.unsqueeze(0)
        else:
            with torch.no_grad():
                inputs = tokz(prompt1, return_tensors="pt")
                prompt1 = model(**inputs)
                prompt1 = prompt1['last_hidden_state'].detach().squeeze(0)
                inputs = tokz(prompt2, return_tensors="pt")
                prompt2 = model(**inputs)
                prompt2 = prompt2['last_hidden_state'].detach().squeeze(0)
                prompt = {'prompt1':prompt1,'prompt2':prompt2}
                
        item =  {
            'X': X,                                                         # [N, 14] or [N, 4] if backbone_only == True
            'S': torch.tensor(S, dtype=torch.long),                         # [N]
            'prompt': prompt,                 # text embedding ['last_hidden_state', 'pooler_output']
            'position_ids': torch.tensor(position_ids, dtype=torch.long),   # [N]
            'mask': mask,                                                   # [N], 1 for generation
            'atom_mask': atom_mask,                                         # [N, 14] or [N, 4], 1 for having records in the PDB
            'lengths': len(S),
            'atom_gt':atom_embedding
        }

        if L is not None:
            item['L'] = L
        return item

    def collate_fn(self, batch):
        results = {}
        if self.padding_collate:
            pad_idx = VOCAB.symbol_to_idx(VOCAB.PAD)
            for key in batch[0]:
                values = [item[key] for item in batch]
                if values[0] is None:
                    results[key] = None
                    continue
                if key == 'lengths':
                    results[key] = torch.tensor(values, dtype=torch.long)
                elif key == 'S':
                    results[key] = pad_sequence(values, batch_first=True, padding_value=pad_idx)
                elif key == 'prompt_lengths':
                    lengths = [x['prompt'].size(0) for x in batch]  # The length of every sequence
                    results['prompt_lengths'] = torch.tensor(lengths,dtype=torch.long)
                elif key == 'prompt':
                    prompts = [x['prompt'] for x in batch]
                    results['prompt'] = pad_sequence(prompts, batch_first=True, padding_value=0.0)
                else:
                    # results[key] = pad_sequence(values, batch_first=True, padding_value=0)
                    results[key] = torch.cat(values, dim=0)
            return results
        else:
            for key in batch[0]:
                values = [item[key] for item in batch]
                if values[0] is None:
                    results[key] = None
                    continue
                if key == 'lengths':
                    results[key] = torch.tensor(values, dtype=torch.long)
                elif key == 'prompt_lengths':
                    lengths = [x['prompt'].size(0) for x in batch]  # The length of every sequence
                    results['prompt_lengths'] = torch.tensor(lengths,dtype=torch.long)
                elif key == 'prompt':
                    key_masks = {}
                    prompts_dict = {}
                    prompts = [x['prompt']['prompt1'] for x in batch]
                    prompts_lengths = torch.tensor([prompt.shape[0] for prompt in prompts])
                    max_prompt_length = prompts_lengths.max()
                    key_mask = torch.arange(max_prompt_length).expand(len(batch), max_prompt_length) < prompts_lengths.unsqueeze(1)
                    key_masks['prompt1_mask'] = key_mask
                    prompts_dict['prompt1'] = pad_sequence(prompts, batch_first=True, padding_value=0.0)

                    prompts = [x['prompt']['prompt2'] for x in batch]
                    prompts_lengths = torch.tensor([prompt.shape[0] for prompt in prompts])
                    max_prompt_length = prompts_lengths.max()
                    key_mask = torch.arange(max_prompt_length).expand(len(batch), max_prompt_length) < prompts_lengths.unsqueeze(1)
                    key_masks['prompt2_mask'] = key_mask
                    prompts_dict['prompt2'] = pad_sequence(prompts, batch_first=True, padding_value=0.0)

                    results['prompt'] = prompts_dict
                    results['key_mask'] = key_masks
                else:
                    results[key] = torch.cat(values, dim=0)
            return results


@R.register('ShapeDataset')
class ShapeDataset(CoDesignDataset):
    def __init__(
            self,
            mmap_dir: str,
            specify_data: Optional[str] = None,
            specify_index: Optional[str] = None,
            padding_collate: bool = False,
            cluster: Optional[str] = None
        ) -> None:
        super().__init__(mmap_dir, False, specify_data, specify_index, padding_collate, cluster)
        self.ca_idx = VOCAB.backbone_atoms.index('CA')
    
    def __getitem__(self, idx: int):
        item = super().__getitem__(idx)

        # refine coordinates to CA and the atom furthest from CA
        X = item['X'] # [N, 14, 3]
        atom_mask = item['atom_mask']
        ca_x = X[:, self.ca_idx].unsqueeze(1) # [N, 1, 3]
        sc_x = X[:, 4:]  # [N, 10, 3], sidechain atom indexes
        dist = torch.norm(sc_x - ca_x, dim=-1) # [N, 10]
        dist = dist.masked_fill(~atom_mask[:, 4:], 1e10)
        furthest_atom_x = sc_x[torch.arange(sc_x.shape[0]), torch.argmax(dist, dim=-1)] # [N, 3]
        X = torch.cat([ca_x, furthest_atom_x.unsqueeze(1)], dim=1)
        
        item['X'] = X
        return item


if __name__ == '__main__':
    import sys
    dataset = CoDesignDataset(sys.argv[1], backbone_only=True)
    print(dataset[0])
