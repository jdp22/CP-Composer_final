import enum

import torch
import torch.nn as nn

import utils.register as R
from utils.oom_decorator import oom_decorator
from data.format import VOCAB

from .diffusion.dpm_full import FullDPM,PromptDPM
from .energies.dist import dist_energy
from ..autoencoder.model import AutoEncoder
from .ldm import LDMPepDesign

@R.register('Prompt_LDMPepDesign')
class Prompt_LDMPepDesign(LDMPepDesign):
    def __init__(self,
            autoencoder_ckpt,
            autoencoder_no_randomness,
            hidden_size,
            num_steps,
            n_layers,
            dist_rbf=0,
            dist_rbf_cutoff=7.0,
            n_rbf=0,
            cutoff=1.0,
            max_gen_position=30,
            mode='codesign',
            h_loss_weight=None,
            diffusion_opt={}):
        super().__init__(autoencoder_ckpt,
            autoencoder_no_randomness,
            hidden_size,
            num_steps,
            n_layers,
            dist_rbf=dist_rbf,
            dist_rbf_cutoff=dist_rbf_cutoff,
            n_rbf=n_rbf,
            cutoff=cutoff,
            max_gen_position=max_gen_position,
            mode=mode,
            h_loss_weight=h_loss_weight,
            diffusion_opt=diffusion_opt)
        
        latent_size = self.autoencoder.latent_size if self.train_sequence else hidden_size
        self.diffusion = PromptDPM(
            latent_size=latent_size,
            hidden_size=hidden_size,
            n_channel=self.autoencoder.n_channel,
            num_steps=num_steps,
            n_layers=n_layers,
            n_rbf=n_rbf,
            cutoff=cutoff,
            dist_rbf=dist_rbf,
            dist_rbf_cutoff=dist_rbf_cutoff,
            **diffusion_opt
        )
    
    @oom_decorator
    def forward(self, X, S,prompt, mask, position_ids, lengths, atom_mask,key_mask, L=None):
        '''
            L: [bs, 3, 3], cholesky decomposition of the covariance matrix \Sigma = LL^T
        '''

        # encode latent_H_0 (N*d) and latent_X_0 (N*3)
        with torch.no_grad():
            self.autoencoder.eval()
            H, Z, _, _ = self.autoencoder.encode(X, S, mask, position_ids, lengths, atom_mask, no_randomness=self.autoencoder_no_randomness)
        
        # diffusion model
        if self.train_sequence:
            S = S.clone()
            S[mask] = self.latent_idx

        with torch.no_grad():
            H_0, (atom_embeddings, _) = self.autoencoder.aa_feature(S, position_ids)
        position_embedding = self.abs_position_encoding(torch.where(mask, position_ids + 1, torch.zeros_like(position_ids)))

        if self.train_sequence:
            H_0 = self.hidden2latent(H_0)
            H_0 = H_0.clone()
            H_0[mask] = H
        
        if self.train_structure:
            X = X.clone()
            X[mask] = self.autoencoder._fill_latent_channels(Z)
            atom_mask = atom_mask.clone()
            atom_mask_gen = atom_mask[mask]
            atom_mask_gen[:, :self.autoencoder.latent_n_channel] = 1
            atom_mask_gen[:, self.autoencoder.latent_n_channel:] = 0
            atom_mask[mask] = atom_mask_gen
            del atom_mask_gen
        else:  # fixbb, only retain backbone atoms in masked region
            atom_mask = self.autoencoder._remove_sidechain_atom_mask(atom_mask, mask)

        loss_dict = self.diffusion.forward(
            H_0=H_0,
            X_0=X,
            prompt=prompt,
            position_embedding=position_embedding,
            mask_generate=mask,
            lengths=lengths,
            atom_embeddings=atom_embeddings,
            atom_mask=atom_mask,
            key_mask=key_mask,
            L=L,
            sample_structure=self.train_structure,
            sample_sequence=self.train_sequence
        )
        

        # loss
        loss = 0
        if self.train_sequence:
            loss = loss + loss_dict['H'] * self.h_loss_weight
        if self.train_structure:
            loss = loss + loss_dict['X']

        return loss, loss_dict
    
    @torch.no_grad()
    def sample(
        self,
        X, S,prompt, mask, position_ids, lengths, atom_mask,key_mask, L=None,
        sample_opt={
            'pbar': False,
            'energy_func': None,
            'energy_lambda': 0.0,
            'autoencoder_n_iter': 1
        },
        return_tensor=False,
        optimize_sidechain=True,
    ):
        self.autoencoder.eval()
        # diffusion sample
        if self.train_sequence:
            S = S.clone()
            S[mask] = self.latent_idx

        H_0, (atom_embeddings, _) = self.autoencoder.aa_feature(S, position_ids)
        position_embedding = self.abs_position_encoding(torch.where(mask, position_ids + 1, torch.zeros_like(position_ids)))

        if self.train_sequence:
            H_0 = self.hidden2latent(H_0)
            H_0 = H_0.clone()
            H_0[mask] = 0 # no possibility for leakage

        if self.train_structure:
            X = X.clone()
            X[mask] = 0
            atom_mask = atom_mask.clone()
            atom_mask_gen = atom_mask[mask]
            atom_mask_gen[:, :self.autoencoder.latent_n_channel] = 1
            atom_mask_gen[:, self.autoencoder.latent_n_channel:] = 0
            atom_mask[mask] = atom_mask_gen
            del atom_mask_gen
        else:  # fixbb, only retain backbone atoms in masked region
            atom_mask = self.autoencoder._remove_sidechain_atom_mask(atom_mask, mask)

        sample_opt['sample_sequence'] = self.train_sequence
        sample_opt['sample_structure'] = self.train_structure
        if 'energy_func' in sample_opt:
            if sample_opt['energy_func'] is None:
                pass
            elif sample_opt['energy_func'] == 'default':
                sample_opt['energy_func'] = self.latent_geometry_guidance
            # otherwise this should be a function
        autoencoder_n_iter = sample_opt.pop('autoencoder_n_iter', 1)
        
        traj = self.diffusion.sample(
            H_0, X,prompt, position_embedding, mask, lengths, atom_embeddings, atom_mask, key_mask,L, **sample_opt)
        X_0, H_0 = traj[0]
        X_0, H_0 = X_0[mask][:, :self.autoencoder.latent_n_channel], H_0[mask]

        # autodecoder decode
        batch_X, batch_S, batch_ppls = self.autoencoder.test(
            X, S, mask, position_ids, lengths, atom_mask,
            given_laten_H=H_0, given_latent_X=X_0, return_tensor=return_tensor,
            allow_unk=False, optimize_sidechain=optimize_sidechain,
            n_iter=autoencoder_n_iter
        )

        return batch_X, batch_S, batch_ppls