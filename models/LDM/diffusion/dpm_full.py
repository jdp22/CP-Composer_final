import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import copy
from tqdm.auto import tqdm

from torch.autograd import grad
from torch_scatter import scatter_mean

from utils.nn_utils import variadic_meshgrid

from .transition import construct_transition

from ...dyMEAN.modules.am_egnn import AMEGNN,Prompt_AMEGNN
from ...dyMEAN.modules.radial_basis import RadialBasis
from torch.nn import MultiheadAttention
import random


def low_trianguler_inv(L):
    # L: [bs, 3, 3]
    L_inv = torch.linalg.solve_triangular(L, torch.eye(3).unsqueeze(0).expand_as(L).to(L.device), upper=False)
    return L_inv


class EpsilonNet(nn.Module):

    def __init__(
            self,
            input_size,
            hidden_size,
            n_channel,
            prompt_size,
            n_layers=3,
            edge_size=0,
            n_rbf=0,
            cutoff=1.0,
            dropout=0.1,
            additional_pos_embed=True,
            attention = False
        ):
        super().__init__()
        
        atom_embed_size = hidden_size // 4
        edge_embed_size = hidden_size // 4
        pos_embed_size, seg_embed_size = input_size, input_size
        # enc_input_size = input_size + seg_embed_size + 3 + (pos_embed_size if additional_pos_embed else 0)
        enc_input_size = input_size + 3 + (pos_embed_size if additional_pos_embed else 0)
        if attention:
            self.encoder = Prompt_AMEGNN(enc_input_size, hidden_size, hidden_size, n_channel,
            channel_nf=atom_embed_size, radial_nf=hidden_size,
            in_edge_nf=edge_embed_size + edge_size*2, n_layers=n_layers, residual=True,
            dropout=dropout, dense=False, n_rbf=n_rbf, cutoff=cutoff)
        else:
            self.encoder = AMEGNN(
                enc_input_size, hidden_size, hidden_size, n_channel,
                channel_nf=atom_embed_size, radial_nf=hidden_size,
                in_edge_nf=edge_embed_size + edge_size*2, n_layers=n_layers, residual=True,
                dropout=dropout, dense=False, n_rbf=n_rbf, cutoff=cutoff)
        self.hidden2input = nn.Linear(hidden_size, input_size)
        # self.pos_embed2latent = nn.Linear(hidden_size, pos_embed_size)
        # self.segment_embedding = nn.Embedding(2, seg_embed_size)
        self.edge_embedding = nn.Embedding(2, edge_embed_size)

    def forward(
            self, H_noisy, X_noisy,prompt, position_embedding, ctx_edges, inter_edges,
            atom_embeddings, atom_weights, mask_generate, beta,guidance_edges = None,
            ctx_edge_attr=None, inter_edge_attr=None,guidance_edge_attr=None,batch_ids = None,k_mask=None,text_guidance=False,inference=False):
        """
        Args:
            H_noisy: (N, hidden_size)
            X_noisy: (N, 14, 3)
            mask_generate: (N)
            batch_ids: (N)
            beta: (N)
        Returns:
            eps_H: (N, hidden_size)
            eps_X: (N, 14, 3)
        """
        t_embed = torch.stack([beta, torch.sin(beta), torch.cos(beta)], dim=-1)
        # seg_embed = self.segment_embedding(mask_generate.long())
        if position_embedding is None:
            in_feat = torch.cat([H_noisy, t_embed], dim=-1) # [N, hidden_size * 2 + 3]
        else:
            in_feat = torch.cat([H_noisy, t_embed, position_embedding], dim=-1)
        edges = torch.cat([ctx_edges, inter_edges], dim=-1)
        if guidance_edges is not None:
            edges = torch.cat([ctx_edges, inter_edges,guidance_edges], dim=-1)
        edge_embed = torch.cat([
            torch.zeros_like(ctx_edges[0]), torch.ones_like(inter_edges[0])
        ], dim=-1)
        edge_embed = self.edge_embedding(edge_embed)
        if ctx_edge_attr is None:
            edge_attr = edge_embed
        elif guidance_edge_attr is None:
            edge_attr = torch.cat([
                edge_embed,
                torch.cat([ctx_edge_attr, inter_edge_attr], dim=0)],
                dim=-1
            ) # [E, embed size + edge_attr_size]
        else:
            edge_attr = torch.cat([
                edge_embed,
                torch.cat([ctx_edge_attr, inter_edge_attr], dim=0),
                guidance_edge_attr],
                dim=-1
            ) # [E, embed size + edge_attr_size]
        # next_H, next_X = self.encoder(in_feat, X_noisy,prompt, edges,key_mask_list = k_mask, ctx_edge_attr=edge_attr, channel_attr=atom_embeddings, channel_weights=atom_weights,batch_ids=batch_ids,text_guidance = text_guidance,inference = inference)

        next_H, next_X = self.encoder(in_feat, X_noisy, edges, ctx_edge_attr=edge_attr, channel_attr=atom_embeddings, channel_weights=atom_weights)

        # equivariant vector features changes
        eps_X = next_X - X_noisy
        eps_X = torch.where(mask_generate[:, None, None].expand_as(eps_X), eps_X, torch.zeros_like(eps_X)) 

        # invariant scalar features changes
        next_H = self.hidden2input(next_H)
        eps_H = next_H - H_noisy
        eps_H = torch.where(mask_generate[:, None].expand_as(eps_H), eps_H, torch.zeros_like(eps_H))

        return eps_H, eps_X


class FullDPM(nn.Module):

    def __init__(
        self, 
        latent_size,
        hidden_size,
        n_channel,
        num_steps, 
        n_layers=3,
        dropout=0.1,
        trans_pos_type='Diffusion',
        trans_seq_type='Diffusion',
        trans_pos_opt={}, 
        trans_seq_opt={},
        n_rbf=0,
        cutoff=1.0,
        std=10.0,
        additional_pos_embed=True,
        dist_rbf=0,
        dist_rbf_cutoff=7.0
    ):
        super().__init__()
        # self.eps_net = EpsilonNet(
        #     latent_size, hidden_size, n_channel, n_layers=n_layers, edge_size=dist_rbf,
        #     n_rbf=n_rbf, cutoff=cutoff, dropout=dropout, additional_pos_embed=additional_pos_embed)
        if dist_rbf > 0:
            self.dist_rbf = RadialBasis(dist_rbf, dist_rbf_cutoff)
        self.num_steps = num_steps
        self.trans_x = construct_transition(trans_pos_type, num_steps, trans_pos_opt)
        self.trans_h = construct_transition(trans_seq_type, num_steps, trans_seq_opt)

        self.register_buffer('std', torch.tensor(std, dtype=torch.float))

    def _normalize_position(self, X, batch_ids, mask_generate, atom_mask, L=None):
        ctx_mask = (~mask_generate[:, None].expand_as(atom_mask)) & atom_mask
        ctx_mask[:, 0] = 0
        ctx_mask[:, 2:] = 0 # only retain CA
        centers = scatter_mean(X[ctx_mask], batch_ids[:, None].expand_as(ctx_mask)[ctx_mask], dim=0) # [bs, 3]
        centers = centers[batch_ids].unsqueeze(1) # [N, 1, 3]
        if L is None:
            X = (X - centers) / self.std
        else:
            with torch.no_grad():
                L_inv = low_trianguler_inv(L)
                # print(L_inv[0])
            X = X - centers
            X = torch.matmul(L_inv[batch_ids][..., None, :, :], X.unsqueeze(-1)).squeeze(-1)
        return X, centers

    def _unnormalize_position(self, X_norm, centers, batch_ids, L=None):
        if L is None:
            X = X_norm * self.std + centers
        else:
            X = torch.matmul(L[batch_ids][..., None, :, :], X_norm.unsqueeze(-1)).squeeze(-1) + centers
        return X
    
    @torch.no_grad()
    def _get_batch_ids(self, mask_generate, lengths):

        # batch ids
        batch_ids = torch.zeros_like(mask_generate).long()
        batch_ids[torch.cumsum(lengths, dim=0)[:-1]] = 1
        batch_ids.cumsum_(dim=0)

        return batch_ids

    @torch.no_grad()
    def _get_edges(self, mask_generate, batch_ids, lengths,sample = False):
        row, col = variadic_meshgrid(
            input1=torch.arange(batch_ids.shape[0], device=batch_ids.device),
            size1=lengths,
            input2=torch.arange(batch_ids.shape[0], device=batch_ids.device),
            size2=lengths,
        ) # (row, col)
        
        is_ctx = mask_generate[row] == mask_generate[col] # the edge is in the same protein, 1 is peptide
        is_inter = ~is_ctx 
        ctx_edges = torch.stack([row[is_ctx], col[is_ctx]], dim=0) # [2, Ec]
        inter_edges = torch.stack([row[is_inter], col[is_inter]], dim=0) # [2, Ei]

        if sample:
            is_peptide = mask_generate[row]==1 & mask_generate[col]==1
            peptide_edges = torch.stack([row[is_peptide], col[is_peptide]], dim=0)
            return ctx_edges, inter_edges,peptide_edges
        
        return ctx_edges, inter_edges
    
    @torch.no_grad()
    def _get_edge_dist(self, X, edges, atom_mask):
        '''
        Calculate the distance.
        Args:
            X: [N, 14, 3]
            edges: [2, E]
            atom_mask: [N, 14]
        '''
        ca_x = X[:, 1] # [N, 3]
        no_ca_mask = torch.logical_not(atom_mask[:, 1]) # [N]
        ca_x[no_ca_mask] = X[:, 0][no_ca_mask] # latent coordinates
        dist = torch.norm(ca_x[edges[0]] - ca_x[edges[1]], dim=-1)  # [N]
        return dist

    def forward(self, H_0, X_0, position_embedding, mask_generate, lengths, atom_embeddings, atom_mask, L=None, t=None, sample_structure=True, sample_sequence=True):
        # if L is not None:
        #     L = L / self.std
        batch_ids = self._get_batch_ids(mask_generate, lengths)
        batch_size = batch_ids.max() + 1
        if t == None:
            t = torch.randint(0, self.num_steps + 1, (batch_size,), dtype=torch.long, device=H_0.device)
        X_0, centers = self._normalize_position(X_0, batch_ids, mask_generate, atom_mask, L)
        
        #When we use this module?
        if sample_structure:
            X_noisy, eps_X = self.trans_x.add_noise(X_0, mask_generate, batch_ids, t)
        else:
            X_noisy, eps_X = X_0, torch.zeros_like(X_0)
        if sample_sequence:
            H_noisy, eps_H = self.trans_h.add_noise(H_0, mask_generate, batch_ids, t)
        else:
            H_noisy, eps_H = H_0, torch.zeros_like(H_0)

        ctx_edges, inter_edges = self._get_edges(mask_generate, batch_ids, lengths)
        if hasattr(self, 'dist_rbf'):
            ctx_edge_attr = self._get_edge_dist(self._unnormalize_position(X_noisy, centers, batch_ids, L), ctx_edges, atom_mask)
            inter_edge_attr = self._get_edge_dist(self._unnormalize_position(X_noisy, centers, batch_ids, L), inter_edges, atom_mask)
            ctx_edge_attr = self.dist_rbf(ctx_edge_attr).view(ctx_edges.shape[1], -1)
            inter_edge_attr = self.dist_rbf(inter_edge_attr).view(inter_edges.shape[1], -1)
        else:
            ctx_edge_attr, inter_edge_attr = None, None

        beta = self.trans_x.get_timestamp(t)[batch_ids]  # [N]
        eps_H_pred, eps_X_pred = self.eps_net(
            H_noisy, X_noisy, position_embedding, ctx_edges, inter_edges, atom_embeddings, atom_mask.float(), mask_generate, beta,
            ctx_edge_attr=ctx_edge_attr, inter_edge_attr=inter_edge_attr)

        loss_dict = {}

        # equivariant vector feature loss, TODO: latent channel
        if sample_structure:
            mask_loss = mask_generate[:, None] & atom_mask
            loss_X = F.mse_loss(eps_X_pred[mask_loss], eps_X[mask_loss], reduction='none').sum(dim=-1)  # (Ntgt * n_latent_channel)
            loss_X = loss_X.sum() / (mask_loss.sum().float() + 1e-8)
            loss_dict['X'] = loss_X
        else:
            loss_dict['X'] = 0

        # invariant scalar feature loss
        if sample_sequence:
            loss_H = F.mse_loss(eps_H_pred[mask_generate], eps_H[mask_generate], reduction='none').sum(dim=-1)  # [N]
            loss_H = loss_H.sum() / (mask_generate.sum().float() + 1e-8)
            loss_dict['H'] = loss_H
        else:
            loss_dict['H'] = 0

        return loss_dict

    @torch.no_grad()
    def sample(self, H, X, position_embedding, mask_generate, lengths, atom_embeddings, atom_mask,
        L=None, sample_structure=True, sample_sequence=True, pbar=False, energy_func=None, energy_lambda=0.01
    ):
        """
        Args:
            H: contextual hidden states, (N, latent_size)
            X: contextual atomic coordinates, (N, 14, 3)
            L: cholesky decomposition of the covariance matrix \Sigma=LL^T, (bs, 3, 3)
            energy_func: guide diffusion towards lower energy landscape
        """
        # if L is not None:
        #     L = L / self.std
        batch_ids = self._get_batch_ids(mask_generate, lengths)
        X, centers = self._normalize_position(X, batch_ids, mask_generate, atom_mask, L)
        # print(X[0, 0])
        # Set the orientation and position of residues to be predicted to random values
        if sample_structure:
            X_rand = torch.randn_like(X) # [N, 14, 3]
            X_init = torch.where(mask_generate[:, None, None].expand_as(X), X_rand, X)
        else:
            X_init = X

        if sample_sequence:
            H_rand = torch.randn_like(H)
            H_init = torch.where(mask_generate[:, None].expand_as(H), H_rand, H)
        else:
            H_init = H

        # traj = {self.num_steps: (self._unnormalize_position(X_init, centers, batch_ids, L), H_init)}
        traj = {self.num_steps: (X_init, H_init)}
        if pbar:
            pbar = functools.partial(tqdm, total=self.num_steps, desc='Sampling')
        else:
            pbar = lambda x: x
        for t in pbar(range(self.num_steps, 0, -1)):
            X_t, H_t = traj[t]
            # X_t, _ = self._normalize_position(X_t, batch_ids, mask_generate, atom_mask, L)
            X_t, H_t = torch.round(X_t, decimals=4), torch.round(H_t, decimals=4) # reduce numerical error
            # print(t, 'input', X_t[0, 0] * 1000)
            
            # beta = self.trans_x.var_sched.betas[t].view(1).repeat(X_t.shape[0])
            beta = self.trans_x.get_timestamp(t).view(1).repeat(X_t.shape[0])
            t_tensor = torch.full([X_t.shape[0], ], fill_value=t, dtype=torch.long, device=X_t.device)

            ctx_edges, inter_edges = self._get_edges(mask_generate, batch_ids, lengths)
            if hasattr(self, 'dist_rbf'):
                ctx_edge_attr = self._get_edge_dist(self._unnormalize_position(X_t, centers, batch_ids, L), ctx_edges, atom_mask)
                inter_edge_attr = self._get_edge_dist(self._unnormalize_position(X_t, centers, batch_ids, L), inter_edges, atom_mask)
                ctx_edge_attr = self.dist_rbf(ctx_edge_attr).view(ctx_edges.shape[1], -1)
                inter_edge_attr = self.dist_rbf(inter_edge_attr).view(inter_edges.shape[1], -1)
            else:
                ctx_edge_attr, inter_edge_attr = None, None
            eps_H, eps_X = self.eps_net(
                H_t, X_t, position_embedding, ctx_edges, inter_edges, atom_embeddings, atom_mask.float(), mask_generate, beta,
                ctx_edge_attr=ctx_edge_attr, inter_edge_attr=inter_edge_attr)
            if energy_func is not None:
                with torch.enable_grad():
                    cur_X_state = X_t.clone().double()
                    cur_X_state.requires_grad = True
                    energy = energy_func(
                        X=self._unnormalize_position(cur_X_state, centers.double(), batch_ids, L.double()),
                        mask_generate=mask_generate, batch_ids=batch_ids)
                    energy_eps_X = grad([energy], [cur_X_state], create_graph=False, retain_graph=False)[0].float()
                # print(energy_lambda, energy / mask_generate.sum())
                energy_eps_X[~mask_generate] = 0
                energy_eps_X = -energy_eps_X
                # print(t, 'energy', energy_eps_X[mask_generate][0, 0] * 1000)
            else:
                energy_eps_X = None
            
            # print(t, 'eps X', eps_X[mask_generate][0, 0] * 1000)
            H_next = self.trans_h.denoise(H_t, eps_H, mask_generate, batch_ids, t_tensor)
            X_next = self.trans_x.denoise(X_t, eps_X, mask_generate, batch_ids, t_tensor, guidance=energy_eps_X, guidance_weight=energy_lambda)
            # print(t, 'output', X_next[mask_generate][0, 0] * 1000)
            # if t == 90:
            #     aa

            if not sample_structure:
                X_next = X_t
            if not sample_sequence:
                H_next = H_t

            # traj[t-1] = (self._unnormalize_position(X_next, centers, batch_ids, L), H_next)
            traj[t-1] = (X_next, H_next)
            traj[t] = (self._unnormalize_position(traj[t][0], centers, batch_ids, L).cpu(), traj[t][1].cpu())
            # traj[t] = tuple(x.cpu() for x in traj[t])    # Move previous states to cpu memory.
        traj[0] = (self._unnormalize_position(traj[0][0], centers, batch_ids, L), traj[0][1])
        return traj
    

class PromptDPM(FullDPM):
    def __init__(self,latent_size,
        hidden_size,
        n_channel,
        num_steps, 
        n_layers=3,
        dropout=0.1,
        trans_pos_type='Diffusion',
        trans_seq_type='Diffusion',
        trans_pos_opt={}, 
        trans_seq_opt={},
        n_rbf=0,
        cutoff=1.0,
        std=10.0,
        additional_pos_embed=True,
        dist_rbf=0,
        dist_rbf_cutoff=7.0,
        text_encoder = 'Attention',):
        super().__init__( 
            latent_size,
            hidden_size,
            n_channel,
            num_steps, 
            n_layers=n_layers,
            dropout=dropout,
            trans_pos_type=trans_pos_type,
            trans_seq_type=trans_seq_type,
            trans_pos_opt=trans_pos_opt, 
            trans_seq_opt=trans_seq_opt,
            n_rbf=n_rbf,
            cutoff=cutoff,
            std=std,
            additional_pos_embed=additional_pos_embed,
            dist_rbf=dist_rbf,
            dist_rbf_cutoff=dist_rbf_cutoff,
            )
        #Train a eplison net from sctrach
        # self.prompted_eps_net = EpsilonNet(
        #     latent_size, hidden_size, n_channel, n_layers=n_layers, edge_size=dist_rbf,
        #     n_rbf=n_rbf, cutoff=cutoff, dropout=dropout, additional_pos_embed=additional_pos_embed)

        self.text_encoder = text_encoder
        self.prompt_size = 768
        self.eps_net = EpsilonNet(
            latent_size, hidden_size,n_channel,prompt_size=8, n_layers=n_layers, edge_size=dist_rbf,
            n_rbf=n_rbf, cutoff=cutoff, dropout=dropout, additional_pos_embed=additional_pos_embed,attention = False)
        if text_encoder == 'Linear':
            self.prompt_encoder_H = nn.Linear(self.prompt_size,8)
            for param in self.prompt_encoder_H.parameters():
                param.requires_grad = True
        elif text_encoder == 'Attention':
            self.attention_H = MultiheadAttention(8,num_heads=1,kdim=768,vdim=768)
            for param in self.attention_H.parameters():
                param.requires_grad = True

        self.max_length = 170
        self.p_con = 0.5
        self.balance = torch.nn.Parameter(torch.tensor([5.0],requires_grad=True))

        for param in self.eps_net.parameters():
            param.requires_grad = True

    @torch.no_grad()
    def _get_edges(self, mask_generate, batch_ids, lengths,sample = True):
        row, col = variadic_meshgrid(
            input1=torch.arange(batch_ids.shape[0], device=batch_ids.device),
            size1=lengths,
            input2=torch.arange(batch_ids.shape[0], device=batch_ids.device),
            size2=lengths,
        ) # (row, col)
        
        is_ctx = mask_generate[row] == mask_generate[col] # the edge is in the same protein, 1 is peptide
        is_inter = ~is_ctx 
        ctx_edges = torch.stack([row[is_ctx], col[is_ctx]], dim=0) # [2, Ec]
        inter_edges = torch.stack([row[is_inter], col[is_inter]], dim=0) # [2, Ei]

        if sample:
            is_peptide = mask_generate[row] & mask_generate[col]
            peptide_indices = torch.where(is_peptide)[0]
            peptide_edges = torch.stack([row[is_peptide], col[is_peptide]], dim=0)
            is_valid = peptide_edges[0] > peptide_edges[1] + 1

            filtered_edges = peptide_edges[:, is_valid]
            filter_indices = peptide_indices[is_valid]
            num_samples = int(torch.sum(is_valid).item()*0.7)

            # 获取每个连续True的第一个True的位置
            positions = []
            for i in range(1, len(mask_generate)):
                if mask_generate[i] != mask_generate[i-1]:
                    positions.append(i)
            
            positions.append(len(mask_generate)-1)

            sample_indices = torch.randperm(filtered_edges.shape[1])[:num_samples]
            sampled_edges = filtered_edges[:, sample_indices]
            filter_indices = filter_indices[sample_indices]

            reversed_edges = sampled_edges.flip(0)
            reversed_indices = torch.where(
            (row == reversed_edges[0][:, None]) & (col == reversed_edges[1][:, None])
            )[1]
            
            augmented_edges = torch.cat([sampled_edges, reversed_edges], dim=1)
            augmented_indices = torch.cat([filter_indices, reversed_indices], dim=0)
            return ctx_edges, inter_edges,augmented_edges,augmented_indices
        return ctx_edges, inter_edges
        
    @torch.no_grad()
    def _get_edge_dist(self, X, edges, atom_mask,debug=False):
        '''
        Args:
            X: [N, 14, 3]
            edges: [2, E]
            atom_mask: [N, 14]
        '''
        ca_x = X[:, 1] # [N, 3]
        no_ca_mask = torch.logical_not(atom_mask[:, 1]) # [N]
        ca_x[no_ca_mask] = X[:, 0][no_ca_mask] # latent coordinates
        dist = torch.norm(ca_x[edges[0]] - ca_x[edges[1]], dim=-1)  # [N]
        return dist

    def forward(self, H_0, X_0, prompt, position_embedding, mask_generate,lengths,atom_embeddings, atom_mask,key_mask, L=None,X_true=None, t=None, sample_structure=True, sample_sequence=True):
        # if L is not None:
        #     L = L / self.std
        batch_ids = self._get_batch_ids(mask_generate, lengths)
        batch_size = batch_ids.max() + 1
        if t == None:
            t = torch.randint(0, self.num_steps + 1, (batch_size,), dtype=torch.long, device=H_0.device)
        X_0, centers = self._normalize_position(X_0, batch_ids, mask_generate, atom_mask, L)
        
        #When we use this module?
        if sample_structure:
            X_noisy, eps_X = self.trans_x.add_noise(X_0, mask_generate, batch_ids, t)
        else:
            X_noisy, eps_X = X_0, torch.zeros_like(X_0)
        if sample_sequence:
            H_noisy, eps_H = self.trans_h.add_noise(H_0, mask_generate, batch_ids, t)
        else:
            H_noisy, eps_H = H_0, torch.zeros_like(H_0)

        ctx_edges, inter_edges,sampled_edges,sample_indices = self._get_edges(mask_generate, batch_ids, lengths,sample=True)
        if hasattr(self, 'dist_rbf'):
            ctx_edge_attr = self._get_edge_dist(self._unnormalize_position(X_noisy, centers, batch_ids, L), ctx_edges, atom_mask)
            inter_edge_attr = self._get_edge_dist(self._unnormalize_position(X_noisy, centers, batch_ids, L), inter_edges, atom_mask)
            guidance_edge_attr = self._get_edge_dist(X_true, sampled_edges, atom_mask)
            ctx_edge_attr = self.dist_rbf(ctx_edge_attr).view(ctx_edges.shape[1], -1)
            inter_edge_attr = self.dist_rbf(inter_edge_attr).view(inter_edges.shape[1], -1)
            guidance_edge_attr = self.dist_rbf(guidance_edge_attr).view(sampled_edges.shape[1], -1)

            result = torch.zeros(ctx_edge_attr.shape[0]+inter_edge_attr.shape[0],guidance_edge_attr.shape[1]).to(ctx_edge_attr.device)
            result[sample_indices] = guidance_edge_attr
            guidance_edge_attr = result

            
        else:
            ctx_edge_attr, inter_edge_attr = None, None

        beta = self.trans_x.get_timestamp(t)[batch_ids]  # [N]
        # max_prompt_length = prompt_lengths.max()
        # key_mask = torch.arange(max_prompt_length).expand(batch_size, max_prompt_length).to(prompt_lengths.device) < prompt_lengths.unsqueeze(1)
        
        # Train a eps net with text guidance
        # prompt = prompt[batch_ids]
        if self.text_encoder == 'Attention':
            pass
            # padding_mask = self.generate_padding_mask(batch_ids)
            # sequence_H,attn_mask = self.organize_to_batches(H_noisy,batch_ids)
            # # sequence_X = self.organize_to_batches(X_noisy.view(X_noisy.shape[0],-1),batch_ids)
            # prompt_H,_= self.attention_H(sequence_H,prompt,prompt,mask=attn_mask)
            # prompt_H = prompt_H[padding_mask]
            ## TODO: the update of X
        elif self.text_encoder == 'MLP':
            prompt_H = self.prompt_encoder_H(prompt[batch_ids])
            
        if random.random()<0.5:
            eps_H_pred, eps_X_pred= self.eps_net(H_noisy, X_noisy,prompt,position_embedding, ctx_edges, inter_edges, atom_embeddings, atom_mask.float(), mask_generate, beta,ctx_edge_attr=ctx_edge_attr, inter_edge_attr=inter_edge_attr,guidance_edge_attr = guidance_edge_attr,k_mask=key_mask,batch_ids=batch_ids,text_guidance=True)
        else:
            guidance_edge_attr = torch.zeros_like(guidance_edge_attr)
            eps_H_pred, eps_X_pred= self.eps_net(H_noisy, X_noisy,prompt, position_embedding, ctx_edges, inter_edges, atom_embeddings, atom_mask.float(), mask_generate, beta,ctx_edge_attr=ctx_edge_attr, inter_edge_attr=inter_edge_attr,guidance_edge_attr=guidance_edge_attr,k_mask=key_mask,batch_ids=batch_ids,text_guidance=False)
        loss_dict = {}
        
        # equivariant vector feature loss, TODO: latent channel
        if sample_structure:
            mask_loss = mask_generate[:, None] & atom_mask
            loss_X = F.mse_loss(eps_X_pred[mask_loss], eps_X[mask_loss], reduction='none').sum(dim=-1)  # (Ntgt * n_latent_channel)
            loss_X = loss_X.sum() / (mask_loss.sum().float() + 1e-8)
            loss_dict['X'] = loss_X
        else:
            loss_dict['X'] = 0

        # invariant scalar feature loss
        if sample_sequence:
            loss_H = F.mse_loss(eps_H_pred[mask_generate], eps_H[mask_generate], reduction='none').sum(dim=-1)  # [N]
            loss_H = loss_H.sum() / (mask_generate.sum().float() + 1e-8)
            loss_dict['H'] = loss_H
        else:
            loss_dict['H'] = 0
        return loss_dict
    
    @torch.no_grad()
    def generate_padding_mask(self,B):
        """
        根据样本归属张量 B 生成 padding mask。
        :param B: 样本归属张量，形状为 (N,) 表示每个特征的样本归属
        :param max_length: 最大序列长度，用于构造统一长度的 mask
        :return: padding mask, 形状为 (num_samples, max_length)
        """
        unique_samples = torch.unique(B)
        num_samples = len(unique_samples)

        padding_mask = torch.zeros((num_samples, self.max_length), dtype=torch.bool, device=B.device)

        for i, sample_id in enumerate(unique_samples):
            length = (B == sample_id).sum().item()
            padding_mask[i, :length] = 1 

        return padding_mask
    
    def organize_to_batches(self,Nodes, batch_ids):
        """
        将 A[N, H] 张量根据 B[N, 1] 分组，变为 [num_samples, L, H] 张量。
        :param A: 输入特征张量，形状为 (N, H)
        :param B: 样本归属标识张量，形状为 (N, 1)
        :return: 重组后的张量，形状为 (num_samples, max_length, H)
        """

        unique_samples = torch.unique(batch_ids)
        num_samples = len(unique_samples)

        N, H = Nodes.shape
        grouped_tensor = torch.full((num_samples, self.max_length, H),0, dtype=Nodes.dtype, device=Nodes.device)
        attn_mask = torch.zeros((num_samples, self.max_length, 23),dtype=bool, device=Nodes.device)
        for i, sample_id in enumerate(unique_samples):
            indices = (batch_ids == sample_id).nonzero(as_tuple=True)[0]
            grouped_tensor[i, :len(indices), :] = Nodes[indices]
            attn_mask[sample_id,:len(indices)] = True
        
        return grouped_tensor,attn_mask
    
    @torch.no_grad()
    def sample(self, H, X, prompt,position_embedding, mask_generate, lengths, atom_embeddings, atom_mask,key_mask,L=None,X_true=None, sample_structure=True, sample_sequence=True, pbar=False, energy_func=None, energy_lambda=0.01
    ):
        """
        Args:
            H: contextual hidden states, (N, latent_size)
            X: contextual atomic coordinates, (N, 14, 3)
            L: cholesky decomposition of the covariance matrix \Sigma=LL^T, (bs, 3, 3)
            energy_func: guide diffusion towards lower energy landscape
        """
        # if L is not None: 
        #     L = L / self.std
        self.w = 1
        batch_ids = self._get_batch_ids(mask_generate, lengths)
        batch_size = batch_ids.max() + 1
        X, centers = self._normalize_position(X, batch_ids, mask_generate, atom_mask, L)
        # print(X[0, 0])
        # Set the orientation and position of residues to be predicted to random values
        if sample_structure:
            X_rand = torch.randn_like(X) # [N, 14, 3]
            X_init = torch.where(mask_generate[:, None, None].expand_as(X), X_rand, X)
        else:
            X_init = X

        if sample_sequence:
            H_rand = torch.randn_like(H)
            H_init = torch.where(mask_generate[:, None].expand_as(H), H_rand, H)
        else:
            H_init = H

        # traj = {self.num_steps: (self._unnormalize_position(X_init, centers, batch_ids, L), H_init)}
        traj = {self.num_steps: (X_init, H_init)}
        if pbar:
            pbar = functools.partial(tqdm, total=self.num_steps, desc='Sampling')
        else:
            pbar = lambda x: x
        for t in pbar(range(self.num_steps, 0, -1)):
            X_t, H_t = traj[t]
            # X_t, _ = self._normalize_position(X_t, batch_ids, mask_generate, atom_mask, L)
            X_t, H_t = torch.round(X_t, decimals=4), torch.round(H_t, decimals=4) # reduce numerical error
            # print(t, 'input', X_t[0, 0] * 1000)
            
            beta = self.trans_x.get_timestamp(t).view(1).repeat(X_t.shape[0])
            t_tensor = torch.full([X_t.shape[0], ], fill_value=t, dtype=torch.long, device=X_t.device)

            ctx_edges, inter_edges,sampled_edges,sample_indices = self._get_edges(mask_generate, batch_ids, lengths,sample=True)
            if hasattr(self, 'dist_rbf'):
                ctx_edge_attr = self._get_edge_dist(self._unnormalize_position(X_t, centers, batch_ids, L), ctx_edges, atom_mask)
                inter_edge_attr = self._get_edge_dist(self._unnormalize_position(X_t, centers, batch_ids, L), inter_edges, atom_mask)
                guidance_edge_attr = self._get_edge_dist(X_true, sampled_edges, atom_mask)
                
                guidance_edge_attr.fill_(0)
                
                ctx_edge_attr = self.dist_rbf(ctx_edge_attr).view(ctx_edges.shape[1], -1)
                inter_edge_attr = self.dist_rbf(inter_edge_attr).view(inter_edges.shape[1], -1)
                guidance_edge_attr = self.dist_rbf(guidance_edge_attr).view(sampled_edges.shape[1], -1)


                result = torch.zeros(ctx_edge_attr.shape[0]+inter_edge_attr.shape[0],guidance_edge_attr.shape[1]).to(ctx_edge_attr.device)
                result[sample_indices] = guidance_edge_attr
                guidance_edge_attr = result
            
            else:
                ctx_edge_attr, inter_edge_attr = None, None
            
            beta = self.trans_x.get_timestamp(t).view(1).repeat(X_t.shape[0])
            t_tensor = torch.full([X_t.shape[0], ], fill_value=t, dtype=torch.long, device=X_t.device)

            # max_prompt_length = prompt_lengths.max()
            # key_mask = torch.arange(max_prompt_length).expand(batch_size, max_prompt_length).to(prompt_lengths.device) < prompt_lengths.unsqueeze(1)
            
            prompted_eps_H_pred, prompted_eps_X_pred= self.eps_net(H_t, X_t,prompt,position_embedding, ctx_edges, inter_edges, atom_embeddings, atom_mask.float(), mask_generate, beta,ctx_edge_attr=ctx_edge_attr, inter_edge_attr=inter_edge_attr,guidance_edge_attr = guidance_edge_attr,k_mask=key_mask,batch_ids=batch_ids,text_guidance=True)

            guidance_edge_attr = torch.zeros_like(guidance_edge_attr)
            eps_H, eps_X= self.eps_net(H_t, X_t,prompt,position_embedding, ctx_edges, inter_edges, atom_embeddings, atom_mask.float(), mask_generate, beta,ctx_edge_attr=ctx_edge_attr, inter_edge_attr=inter_edge_attr,guidance_edge_attr = guidance_edge_attr,k_mask=key_mask,batch_ids=batch_ids,text_guidance=False)
            eps_H = (1+self.w)*prompted_eps_H_pred-self.w*eps_H
            eps_X = (1+self.w)*prompted_eps_X_pred-self.w*eps_X
            
            if energy_func is not None:
                with torch.enable_grad():
                    cur_X_state = X_t.clone().double()
                    cur_X_state.requires_grad = True
                    energy = energy_func(
                        X=self._unnormalize_position(cur_X_state, centers.double(), batch_ids, L.double()),
                        mask_generate=mask_generate, batch_ids=batch_ids)
                    energy_eps_X = grad([energy], [cur_X_state], create_graph=False, retain_graph=False)[0].float()
                # print(energy_lambda, energy / mask_generate.sum())
                energy_eps_X[~mask_generate] = 0
                energy_eps_X = -energy_eps_X
                # print(t, 'energy', energy_eps_X[mask_generate][0, 0] * 1000)
            else:
                energy_eps_X = None
            
            H_next = self.trans_h.denoise(H_t, eps_H, mask_generate, batch_ids, t_tensor)
            X_next = self.trans_x.denoise(X_t, eps_X, mask_generate, batch_ids, t_tensor, guidance=energy_eps_X, guidance_weight=energy_lambda)
            # print(t, 'output', X_next[mask_generate][0, 0] * 1000)
            # if t == 90:
            #     aa

            if not sample_structure:
                X_next = X_t
            if not sample_sequence:
                H_next = H_t

            # traj[t-1] = (self._unnormalize_position(X_next, centers, batch_ids, L), H_next)
            traj[t-1] = (X_next, H_next)
            traj[t] = (self._unnormalize_position(traj[t][0], centers, batch_ids, L).cpu(), traj[t][1].cpu())
            # traj[t] = tuple(x.cpu() for x in traj[t])    # Move previous states to cpu memory.
        traj[0] = (self._unnormalize_position(traj[0][0], centers, batch_ids, L), traj[0][1])
        return traj
    
class CrossAttention(nn.Module):
    def __init__(self, hidden_dim, k_dim,v_dim,num_heads, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert self.head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"
        
        # Linear layers for Query, Key, and Value
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(k_dim, hidden_dim)
        self.value_proj = nn.Linear(v_dim, hidden_dim)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        """
        :param query: Tensor of shape (batch_size, query_len, hidden_dim)
        :param key: Tensor of shape (batch_size, key_len, hidden_dim)
        :param value: Tensor of shape (batch_size, value_len, hidden_dim)
        :param mask: Optional Tensor of shape (batch_size, query_len, key_len)
                     mask[i, j, k] = 0 means position (j, k) is valid, -inf means it should be ignored
        :return: Tensor of shape (batch_size, query_len, hidden_dim)
        """
        batch_size, query_len, hidden_dim = query.size()
        key_len = key.size(1)
        
        # Project inputs
        Q = self.query_proj(query)  # (batch_size, query_len, hidden_dim)
        K = self.key_proj(key)      # (batch_size, key_len, hidden_dim)
        V = self.value_proj(value)  # (batch_size, value_len, hidden_dim)
        
        # Split into multiple heads
        Q = Q.view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, query_len, head_dim)
        K = K.view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)    # (batch_size, num_heads, key_len, head_dim)
        V = V.view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)    # (batch_size, num_heads, key_len, head_dim)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))  # (batch_size, num_heads, query_len, key_len)
        
        # Apply mask (if provided)
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1,self.num_heads,1,1)
            scores = scores.masked_fill(mask == 0, 0)
        
        # Attention weights
        attn_weights = F.softmax(scores, dim=-1)  # (batch_size, num_heads, query_len, key_len)
        attn_weights = self.dropout(attn_weights)
        
        # Weighted sum of values
        attn_output = torch.matmul(attn_weights, V)  # (batch_size, num_heads, query_len, head_dim)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, query_len, hidden_dim)  # (batch_size, query_len, hidden_dim)
        # Apply output projection
        output = self.out_proj(attn_output)  # (batch_size, query_len, hidden_dim)
        
        return output