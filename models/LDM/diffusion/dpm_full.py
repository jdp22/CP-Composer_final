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
        enc_input_size = input_size + 3 + (pos_embed_size if additional_pos_embed else 0) +20
        if attention:
            self.encoder = Prompt_AMEGNN(enc_input_size, hidden_size, hidden_size, n_channel,
            channel_nf=atom_embed_size, radial_nf=hidden_size,
            in_edge_nf=edge_embed_size + edge_size*2, n_layers=n_layers, residual=True,
            dropout=dropout, dense=False, n_rbf=n_rbf, cutoff=cutoff)
        else:
            self.encoder = AMEGNN(
                enc_input_size, hidden_size, hidden_size, n_channel,
                channel_nf=atom_embed_size, radial_nf=hidden_size,
                in_edge_nf=edge_embed_size + edge_size, n_layers=n_layers, residual=True,
                dropout=dropout, dense=False, n_rbf=n_rbf, cutoff=cutoff)
        self.hidden2input = nn.Linear(hidden_size, input_size)
        # self.pos_embed2latent = nn.Linear(hidden_size, pos_embed_size)
        # self.segment_embedding = nn.Embedding(2, seg_embed_size)
        self.edge_embedding = nn.Embedding(2, edge_embed_size)

    def forward(
            self, H_noisy, X_noisy,prompt, position_embedding, ctx_edges, inter_edges,
            atom_embeddings, atom_weights, mask_generate, beta,atom_gt,guidance_edges = None,
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
            in_feat = torch.cat([H_noisy, t_embed,atom_gt], dim=-1) # [N, hidden_size * 2 + 3]
        else:
            in_feat = torch.cat([H_noisy, t_embed,atom_gt, position_embedding], dim=-1)
        edges = torch.cat([ctx_edges, inter_edges], dim=-1)
        edge_embed = torch.cat([
            torch.zeros_like(ctx_edges[0]), torch.ones_like(inter_edges[0])
        ], dim=-1)
        edge_embed = self.edge_embedding(edge_embed)
        if ctx_edge_attr is None:
            edge_attr = edge_embed
        else:
            try:
                edge_attr = torch.cat([
                    edge_embed,
                    torch.cat([ctx_edge_attr, inter_edge_attr], dim=0)],
                    dim=-1
                ) # [E, embed size + edge_attr_size]
            except:
                breakpoint()
        # next_H, next_X = self.encoder(in_feat, X_noisy,prompt, edges,key_mask_list = k_mask, ctx_edge_attr=edge_attr, channel_attr=atom_embeddings, channel_weights=atom_weights,batch_ids=batch_ids,text_guidance = text_guidance,inference = inference)
        next_H, next_X = self.encoder(in_feat, X_noisy, edges, ctx_edge_attr=edge_attr, channel_attr=atom_embeddings, channel_weights=atom_weights,guidance_edges = guidance_edges,guidance_edge_attr = guidance_edge_attr)

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

        self.guidance_dist_rbf = RadialBasis(dist_rbf,cutoff=20)
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
            sampled_edges = []
            for k in [3,4,6]:
                shifted_tensor = torch.roll(mask_generate, shifts=-k)  # 
                shifted_tensor[-k:] = False
                inner_positions = mask_generate & shifted_tensor  # 检查位置i和i+3是否都为True

                inner_positions = torch.nonzero(inner_positions).squeeze()
                if inner_positions is None:
                    continue
                def find_consecutive_groups(tensor):
                    groups = []
                    group = [tensor[0]]
                    for i in range(1, len(tensor)):
                        if tensor[i] == tensor[i-1] + 1:
                            group.append(tensor[i])
                        else:
                            groups.append(group)
                            group = [tensor[i]]
                    groups.append(group)  
                    return groups

                def sample_from_groups(groups):
                    sampled = []
                    for group in groups:
                        num_to_sample = random.randint(0, min(2, len(group)))  # 随机选择1到4个数字
                        sampled+=random.sample(group, num_to_sample)
                    return sampled
                
                groups = find_consecutive_groups(inner_positions)

                sampled_numbers = sample_from_groups(groups)
                if len(sampled_numbers)==0:
                    continue
                inner_positions1 = []
                inner_positions2 = []
                for sampled_number in sampled_numbers:
                    if sampled_number+k>len(mask_generate)-1:
                        continue
                    inner_positions1.append(sampled_number)
                    inner_positions2.append(sampled_number+k)
                inner_positions1 = torch.stack(inner_positions1)
                inner_positions2 = torch.stack(inner_positions2)
                inner_edges = torch.stack([inner_positions1, inner_positions2], dim=0)
                reversed_inner_edges = inner_edges.flip(0)

                sampled_edges.append(inner_edges)
                sampled_edges.append(reversed_inner_edges)
            
            # 获取每个连续True的第一个True的位置
            head_positions = []
            tail_positions = []
            for i in range(1, len(mask_generate)):
                if mask_generate[i] != mask_generate[i-1]:
                    if mask_generate[i]:
                        head_positions.append(i)
                    else:
                        tail_positions.append(i-1)
            tail_positions.append(len(mask_generate)-1)
            head_positions = torch.tensor(head_positions).to(row.device)
            tail_positions = torch.tensor(tail_positions).to(col.device)

            sampled_ht_edges = torch.stack([head_positions, tail_positions], dim=0)
            reversed_edges = sampled_ht_edges.flip(0)
            sampled_edges.append(sampled_ht_edges)
            sampled_edges.append(reversed_edges)
            augmented_edges = torch.cat(sampled_edges, dim=1)
            return ctx_edges, inter_edges,augmented_edges
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

    def forward(self, H_0, X_0, prompt, position_embedding, mask_generate,lengths,atom_embeddings, atom_mask,key_mask,atom_gt, L=None,X_true=None, t=None, sample_structure=True, sample_sequence=True):
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

        ctx_edges, inter_edges,sampled_edges = self._get_edges(mask_generate, batch_ids, lengths,sample=True)
        if hasattr(self, 'dist_rbf'):
            ctx_edge_attr = self._get_edge_dist(self._unnormalize_position(X_noisy, centers, batch_ids, L), ctx_edges, atom_mask)
            inter_edge_attr = self._get_edge_dist(self._unnormalize_position(X_noisy, centers, batch_ids, L), inter_edges, atom_mask)
            guidance_edge_attr = self._get_edge_dist(X_true, sampled_edges, atom_mask)
            ctx_edge_attr = self.dist_rbf(ctx_edge_attr).view(ctx_edges.shape[1], -1)
            inter_edge_attr = self.dist_rbf(inter_edge_attr).view(inter_edges.shape[1], -1)
            guidance_edge_attr = self.guidance_dist_rbf(guidance_edge_attr).view(sampled_edges.shape[1], -1)           
        else:
            ctx_edge_attr, inter_edge_attr = None, None

        beta = self.trans_x.get_timestamp(t)[batch_ids]  # [N]
        atom_full_tmp = torch.zeros((mask_generate.shape[0],atom_gt.shape[1])).to(guidance_edge_attr.device)
        atom_full_tmp[mask_generate] = atom_gt
        
        # 获取所有唯一的数字
        unique_vals = torch.unique(batch_ids)

        # 用于存储抽样结果
        sampled_indices = []

        # 对每个数字进行抽样
        for val in unique_vals:
            # 找出当前数字的位置
            valid_indices = (batch_ids == val) & mask_generate
            indices = valid_indices.nonzero(as_tuple=True)[0]
            
            # 抽样 1-4 个位置
            sampled = indices[torch.randint(0, indices.size(0), (random.randint(1, min(4, len(indices))),))]
            
            sampled_indices+=sampled
        sampled_indices = torch.tensor(sampled_indices).to(guidance_edge_attr.device)
        atom_full = torch.zeros((mask_generate.shape[0],atom_gt.shape[1])).to(guidance_edge_attr.device)
        atom_full[sampled_indices] = atom_full_tmp[sampled_indices]

        guidance_flag = random.random()
        if guidance_flag<0.2:
            # No guidance
            atom_full = torch.zeros_like(atom_full)
            guidance_edge_attr = None
            sampled_edges = None
        elif guidance_flag<0.5:
            # Only distance guidance
            atom_full = torch.zeros_like(atom_full)
        elif guidance_flag<0.8:
            # Only atom guidance
            guidance_edge_attr = None
            sampled_edges = None
        
        eps_H_pred, eps_X_pred = self.eps_net(H_noisy, X_noisy,prompt,position_embedding, ctx_edges, inter_edges, atom_embeddings, atom_mask.float(), mask_generate, beta,atom_gt=atom_full,ctx_edge_attr=ctx_edge_attr, inter_edge_attr=inter_edge_attr,guidance_edge_attr = guidance_edge_attr,guidance_edges = sampled_edges,k_mask=key_mask,batch_ids=batch_ids,text_guidance=True)
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
    def sample(self, H, X, prompt,position_embedding, mask_generate, lengths, atom_embeddings, atom_mask,key_mask,aa_emb_gt,L=None,atom_gt=None,X_true=None, sample_structure=True, sample_sequence=True, pbar=False, energy_func=None, energy_lambda=0.01
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
        self.w = 2
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

            ctx_edges, inter_edges,sampled_edges = self._get_edges(mask_generate, batch_ids, lengths,sample=True)
            if hasattr(self, 'dist_rbf'):
                ctx_edge_attr = self._get_edge_dist(self._unnormalize_position(X_t, centers, batch_ids, L), ctx_edges, atom_mask)
                inter_edge_attr = self._get_edge_dist(self._unnormalize_position(X_t, centers, batch_ids, L), inter_edges, atom_mask)
                guidance_edge_attr = self._get_edge_dist(X_true, sampled_edges, atom_mask)
                
                guidance_edge_attr.fill_(3.8)
                
                ctx_edge_attr = self.dist_rbf(ctx_edge_attr).view(ctx_edges.shape[1], -1)
                inter_edge_attr = self.dist_rbf(inter_edge_attr).view(inter_edges.shape[1], -1)
                guidance_edge_attr = self.guidance_dist_rbf(guidance_edge_attr).view(sampled_edges.shape[1], -1)


                # result = torch.zeros(ctx_edge_attr.shape[0]+inter_edge_attr.shape[0],guidance_edge_attr.shape[1]).to(ctx_edge_attr.device)
                # result[sample_indices] = guidance_edge_attr
                # guidance_edge_attr = result
            
            else:
                ctx_edge_attr, inter_edge_attr = None, None
            
            beta = self.trans_x.get_timestamp(t).view(1).repeat(X_t.shape[0])
            t_tensor = torch.full([X_t.shape[0], ], fill_value=t, dtype=torch.long, device=X_t.device)

            atom_full_tmp = torch.zeros((mask_generate.shape[0],atom_gt.shape[1])).to(guidance_edge_attr.device)
            atom_full_tmp[mask_generate] = atom_gt

            # 获取所有唯一的数字
            unique_vals = torch.unique(batch_ids)

            # 用于存储抽样结果
            sampled_indices = []

            # 对每个数字进行抽样
            for val in unique_vals:
                # 找出当前数字的位置
                valid_indices = (batch_ids == val) & mask_generate
                indices = valid_indices.nonzero(as_tuple=True)[0]
                if len(indices)<4:
                    continue
                sampled = [indices[0],indices[3]]
                sampled_indices+=sampled
            sampled_indices = torch.tensor(sampled_indices).to(guidance_edge_attr.device)
            atom_full = torch.zeros((mask_generate.shape[0],atom_gt.shape[1])).to(guidance_edge_attr.device)
            one_hot_vector = torch.zeros(20).to(guidance_edge_attr.device)
            one_hot_vector[4] = 1
            one_hot_vector = one_hot_vector.unsqueeze(0).repeat(len(sampled_indices),1)
            atom_full[sampled_indices] = one_hot_vector
            
            atom_full,sampled_edges,guidance_edge_attr = self.condition1(atom_gt,batch_ids,mask_generate,X_true,atom_mask)
            prompted_eps_H_pred, prompted_eps_X_pred= self.eps_net(H_t, X_t,prompt,position_embedding, ctx_edges, inter_edges, atom_embeddings, atom_mask.float(), mask_generate, beta,atom_gt=atom_full,ctx_edge_attr=ctx_edge_attr, inter_edge_attr=inter_edge_attr,guidance_edges=sampled_edges,guidance_edge_attr = guidance_edge_attr,k_mask=key_mask,batch_ids=batch_ids,text_guidance=True)

            atom_full = torch.zeros_like(atom_full)
            eps_H, eps_X= self.eps_net(H_t, X_t,prompt,position_embedding, ctx_edges, inter_edges, atom_embeddings, atom_mask.float(), mask_generate, beta,atom_gt=atom_full,ctx_edge_attr=ctx_edge_attr, inter_edge_attr=inter_edge_attr,guidance_edges=None,guidance_edge_attr = None,k_mask=key_mask,batch_ids=batch_ids,text_guidance=False)

            eps_H = (1+self.w)*prompted_eps_H_pred-self.w*eps_H
            eps_X = (1+self.w)*prompted_eps_X_pred-self.w*eps_X
            
            if energy_func is not None:
                with torch.enable_grad():
                    cur_X_state = X_t.clone().double()
                    cur_H_state = H_t.clone().double()

                    cur_X_state.requires_grad = True
                    cur_H_state.requires_grad = True
                    energy = energy_func(
                        X=self._unnormalize_position(cur_X_state, centers.double(), batch_ids, L.double()),
                        H = cur_H_state,atom_gt = aa_emb_gt,mask_generate=mask_generate, batch_ids=batch_ids)
                    print(t,energy)
                    energy_eps_X = grad([energy], [cur_X_state], create_graph=False, retain_graph=False)[0].float()
                    energy_eps_H = grad([energy], [cur_H_state], create_graph=False, retain_graph=False)[0].float()
                # print(energy_lambda, energy / mask_generate.sum())
                energy_eps_X[~mask_generate] = 0
                energy_eps_X = -energy_eps_X

                energy_eps_H[~mask_generate] = 0
                energy_eps_H = -energy_eps_H
                # print(t, 'energy', energy_eps_X[mask_generate][0, 0] * 1000)
            else:
                energy_eps_X = None
                energy_eps_H = None
            
            H_next = self.trans_h.denoise(H_t, eps_H, mask_generate, batch_ids, t_tensor,guidance=energy_eps_H,guidance_weight=energy_lambda)
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
    
    def condition1(self,atom_gt,batch_ids,mask_generate,X_true,atom_mask):
        '''
        k-D/E i-i+3/i+4 distance 4-6.5
        '''
        unique_vals = torch.unique(batch_ids)
        sampled_indices = []
        positions1 = []
        positions2 = []
        for val in unique_vals:
            valid_indices = (batch_ids == val) & mask_generate
            indices = valid_indices.nonzero(as_tuple=True)[0]
            if len(indices)<5:
                continue
            random_indices = indices[(max(indices)-indices>=4)]
            indice = random.choice(random_indices)
            hop = random.choice([3,4])
            sampled = [indice,indice+hop]
            positions1.append(indice)
            positions2.append(indice+hop)
            sampled_indices+=sampled
        
        positions1 = torch.stack(positions1).to(atom_gt.device)
        positions2 = torch.stack(positions2).to(atom_gt.device)
        
        # control the type of K 
        atom_full = torch.zeros((mask_generate.shape[0],atom_gt.shape[1])).to(atom_gt.device)
        one_hot_vector = torch.zeros(20).to(atom_gt.device)
        one_hot_vector[11] = 1 # K
        one_hot_vector = one_hot_vector.unsqueeze(0).repeat(len(positions1),1)
        atom_full[positions1] = one_hot_vector

        # control the type of D/E
        atom_full = torch.zeros((mask_generate.shape[0],atom_gt.shape[1])).to(atom_gt.device)
        one_hot_vector = torch.zeros(20).to(atom_gt.device)
        if random.random()<0.5:
            one_hot_vector[3] = 1 # D
        else:
            one_hot_vector[5] = 1 # E
        one_hot_vector = one_hot_vector.unsqueeze(0).repeat(len(positions2),1)
        atom_full[positions2] = one_hot_vector

        edges = torch.stack([positions1, positions2], dim=0)
        reversed_edges = edges.flip(0)

        sampled_edges = torch.cat([edges,reversed_edges], dim=1)
        guidance_edge_attr = self._get_edge_dist(X_true, sampled_edges, atom_mask)       
        guidance_edge_attr.fill_(4.5)      
        guidance_edge_attr = self.guidance_dist_rbf(guidance_edge_attr).view(sampled_edges.shape[1], -1)
        return atom_full,sampled_edges,guidance_edge_attr
    
    def condition11(self,atom_gt,batch_ids,mask_generate,X_true,atom_mask):
        '''
        k-D/E i-i+3/i+4 distance 4-6.5
        '''
        unique_vals = torch.unique(batch_ids)
        sampled_indices = []
        positions1 = []
        positions2 = []
        for val in unique_vals:
            valid_indices = (batch_ids == val) & mask_generate
            indices = valid_indices.nonzero(as_tuple=True)[0]
            if len(indices)<=10:
                continue
            random_indices = indices[(max(indices)-indices>=10)]
            indice = random.choice(random_indices)
            hop = random.choice([3,4])
            sampled = [indice,indice+hop]
            positions1.append(indice)
            positions2.append(indice+hop)
            sampled_indices+=sampled

            random_indices = indices[(indices>indice+hop)&(max(indices)-indices>=4)]
            indice = random.choice(random_indices)
            hop = random.choice([3,4])
            sampled = [indice,indice+hop]
            positions1.append(indice)
            positions2.append(indice+hop)
            sampled_indices+=sampled

        positions1 = torch.stack(positions1).to(atom_gt.device)
        positions2 = torch.stack(positions2).to(atom_gt.device)
        
        # control the type of K 
        atom_full = torch.zeros((mask_generate.shape[0],atom_gt.shape[1])).to(atom_gt.device)
        one_hot_vector = torch.zeros(20).to(atom_gt.device)
        one_hot_vector[11] = 1 # K
        one_hot_vector = one_hot_vector.unsqueeze(0).repeat(len(positions1),1)
        atom_full[positions1] = one_hot_vector

        # control the type of D/E
        atom_full = torch.zeros((mask_generate.shape[0],atom_gt.shape[1])).to(atom_gt.device)
        one_hot_vector = torch.zeros(20).to(atom_gt.device)
        if random.random()<0.5:
            one_hot_vector[3] = 1 # D
        else:
            one_hot_vector[5] = 1 # E
        one_hot_vector = one_hot_vector.unsqueeze(0).repeat(len(positions2),1)
        atom_full[positions2] = one_hot_vector

        edges = torch.stack([positions1, positions2], dim=0)
        reversed_edges = edges.flip(0)

        sampled_edges = torch.cat([edges,reversed_edges], dim=1)
        guidance_edge_attr = self._get_edge_dist(X_true, sampled_edges, atom_mask)       
        guidance_edge_attr.fill_(4.5)      
        guidance_edge_attr = self.guidance_dist_rbf(guidance_edge_attr).view(sampled_edges.shape[1], -1)
        return atom_full,sampled_edges,guidance_edge_attr
    
    def condition13(self,atom_gt,batch_ids,mask_generate,X_true,atom_mask):
        '''
        k-D/E i-i+3/i+4 distance 4-6.5
        and
        two positions Cys with distance between 3.5-5 A
        '''
        unique_vals = torch.unique(batch_ids)
        sampled_indices = []
        positions1 = []
        positions2 = []
        positions3 = []
        positions4 = []
        for val in unique_vals:
            valid_indices = (batch_ids == val) & mask_generate
            indices = valid_indices.nonzero(as_tuple=True)[0]
            if len(indices)<=10:
                continue
            random_indices = indices[(max(indices)-indices>=10)]
            indice = random.choice(random_indices)
            hop = random.choice([3,4])
            
            positions1.append(indice)
            positions2.append(indice+hop)
            
            head_indices = indices[(indices>indice+hop)&(max(indices)-indices>=3)]
            head_indice = random.choice(head_indices)

            sampled_indices+= [indice,indice+3]
            positions3.append(head_indice)
            positions4.append(head_indice+3)

        # Given the K-D\E guidance
        positions1 = torch.stack(positions1).to(atom_gt.device)
        positions2 = torch.stack(positions2).to(atom_gt.device)
        
        # control the type of K 
        atom_full = torch.zeros((mask_generate.shape[0],atom_gt.shape[1])).to(atom_gt.device)
        one_hot_vector = torch.zeros(20).to(atom_gt.device)
        one_hot_vector[11] = 1 # K
        one_hot_vector = one_hot_vector.unsqueeze(0).repeat(len(positions1),1)
        atom_full[positions1] = one_hot_vector

        # control the type of D/E
        atom_full = torch.zeros((mask_generate.shape[0],atom_gt.shape[1])).to(atom_gt.device)
        one_hot_vector = torch.zeros(20).to(atom_gt.device)
        if random.random()<0.5:
            one_hot_vector[3] = 1 # D
        else:
            one_hot_vector[5] = 1 # E
        one_hot_vector = one_hot_vector.unsqueeze(0).repeat(len(positions2),1)
        atom_full[positions2] = one_hot_vector

        edges = torch.stack([positions1, positions2], dim=0)
        reversed_edges = edges.flip(0)

        sampled_edges1 = torch.cat([edges,reversed_edges], dim=1)

        # Control the 
        sampled_indices = torch.tensor(sampled_indices).to(atom_gt.device)
        one_hot_vector = torch.zeros(20).to(atom_gt.device)
        one_hot_vector[4] = 1
        one_hot_vector = one_hot_vector.unsqueeze(0).repeat(len(sampled_indices),1)
        atom_full[sampled_indices] = one_hot_vector

        positions3 = torch.stack(positions3)
        positions4 = torch.stack(positions4)
        edges = torch.stack([positions3, positions4], dim=0)
        reversed_edges = edges.flip(0)

        sampled_edges2 = torch.cat([edges,reversed_edges], dim=1)

        sampled_edges = torch.cat([sampled_edges1,sampled_edges2],dim=1)

        guidance_edge_attr = self._get_edge_dist(X_true, sampled_edges, atom_mask)       
        guidance_edge_attr.fill_(4.5)      
        guidance_edge_attr = self.guidance_dist_rbf(guidance_edge_attr).view(sampled_edges.shape[1], -1)
        return atom_full,sampled_edges,guidance_edge_attr
    
    def condition2(self,atom_gt,batch_ids,mask_generate,X_true,atom_mask):
        '''
        The distance of head and tail is less than 6 A
        '''
        atom_full = torch.zeros((mask_generate.shape[0],atom_gt.shape[1])).to(atom_gt.device)

        head_positions = []
        tail_positions = []
        for i in range(1, len(mask_generate)):
            if mask_generate[i] != mask_generate[i-1]:
                if mask_generate[i]:
                    head_positions.append(i)
                else:
                    tail_positions.append(i-1)
        tail_positions.append(len(mask_generate)-1)
        head_positions = torch.tensor(head_positions).to(atom_gt.device)
        tail_positions = torch.tensor(tail_positions).to(atom_gt.device)

        sampled_ht_edges = torch.stack([head_positions, tail_positions], dim=0)
        reversed_edges = sampled_ht_edges.flip(0)
        
        sampled_edges = torch.cat([sampled_ht_edges, reversed_edges], dim=1)
        guidance_edge_attr = self._get_edge_dist(X_true, sampled_edges, atom_mask)       
        guidance_edge_attr.fill_(3.8)      
        guidance_edge_attr = self.guidance_dist_rbf(guidance_edge_attr).view(sampled_edges.shape[1], -1)

        return atom_full,sampled_edges,guidance_edge_attr
    
    def condition23(self,atom_gt,batch_ids,mask_generate,X_true,atom_mask):
        '''
        The distance of head and tail is less than 6 A and two positions Cys with distance between 3.5-5 A
        '''

        device = atom_gt.device
        unique_vals = torch.unique(batch_ids)
        sampled_indices = []
        positions1 = []
        positions2 = []
        for val in unique_vals:
            valid_indices = (batch_ids == val) & mask_generate
            indices = valid_indices.nonzero(as_tuple=True)[0]
            
            if len(indices)<4:
                continue
            head_indices = indices[(max(indices)-indices>=3)]
            head_indice = random.choice(head_indices)
            sampled = [head_indice,head_indice+3]
            positions1.append(int(head_indice))
            positions2.append(int(head_indice+3))
            sampled_indices+=sampled
        
        for i in range(1, len(mask_generate)):
            if mask_generate[i] != mask_generate[i-1]:
                if mask_generate[i]:
                    positions1.append(i)
                else:
                    positions2.append(i-1)

        positions2.append(len(mask_generate)-1)
        
        sampled_indices = torch.tensor(sampled_indices).to(atom_gt.device)
        atom_full = torch.zeros((mask_generate.shape[0],atom_gt.shape[1])).to(atom_gt.device)
        one_hot_vector = torch.zeros(20).to(atom_gt.device)
        one_hot_vector[4] = 1
        one_hot_vector = one_hot_vector.unsqueeze(0).repeat(len(sampled_indices),1)
        atom_full[sampled_indices] = one_hot_vector

        positions1 = torch.tensor(positions1).to(device)
        positions2 = torch.tensor(positions2).to(device)
        edges = torch.stack([positions1, positions2], dim=0)
        reversed_edges = edges.flip(0)

        sampled_edges = torch.cat([edges,reversed_edges], dim=1)
        guidance_edge_attr = self._get_edge_dist(X_true, sampled_edges, atom_mask)       
        guidance_edge_attr.fill_(3.8)      
        guidance_edge_attr = self.guidance_dist_rbf(guidance_edge_attr).view(sampled_edges.shape[1], -1)

        return atom_full,sampled_edges,guidance_edge_attr

    
    def condition3(self,atom_gt,batch_ids,mask_generate,X_true,atom_mask):
        '''
        two positions Cys with distance between 3.5-5 A
        '''
        unique_vals = torch.unique(batch_ids)
        sampled_indices = []
        positions1 = []
        positions2 = []
        for val in unique_vals:
            valid_indices = (batch_ids == val) & mask_generate
            indices = valid_indices.nonzero(as_tuple=True)[0]
            
            if len(indices)<4:
                continue
            head_indices = indices[(max(indices)-indices>=3)]
            head_indice = random.choice(head_indices)
            sampled = [head_indice,head_indice+3]
            positions1.append(head_indice)
            positions2.append(head_indice+3)
            sampled_indices+=sampled
        sampled_indices = torch.tensor(sampled_indices).to(atom_gt.device)
        atom_full = torch.zeros((mask_generate.shape[0],atom_gt.shape[1])).to(atom_gt.device)
        one_hot_vector = torch.zeros(20).to(atom_gt.device)
        one_hot_vector[4] = 1
        one_hot_vector = one_hot_vector.unsqueeze(0).repeat(len(sampled_indices),1)
        atom_full[sampled_indices] = one_hot_vector

        positions1 = torch.stack(positions1)
        positions2 = torch.stack(positions2)
        edges = torch.stack([positions1, positions2], dim=0)
        reversed_edges = edges.flip(0)

        sampled_edges = torch.cat([edges,reversed_edges], dim=1)
        guidance_edge_attr = self._get_edge_dist(X_true, sampled_edges, atom_mask)       
        guidance_edge_attr.fill_(3.8)      
        guidance_edge_attr = self.guidance_dist_rbf(guidance_edge_attr).view(sampled_edges.shape[1], -1)

        return atom_full,sampled_edges,guidance_edge_attr
    
    def condition33(self,atom_gt,batch_ids,mask_generate,X_true,atom_mask):
        '''
        two positions Cys with distance between 3.5-5 A
        '''
        unique_vals = torch.unique(batch_ids)
        sampled_indices = []
        positions1 = []
        positions2 = []
        for val in unique_vals:
            valid_indices = (batch_ids == val) & mask_generate
            indices = valid_indices.nonzero(as_tuple=True)[0]
            
            if len(indices)<8:
                continue
            head_indices = indices[(max(indices)-indices>=7)]
            head_indice = random.choice(head_indices)
            sampled = [head_indice,head_indice+3]
            positions1.append(head_indice)
            positions2.append(head_indice+3)
            
            head_indices = indices[(indices>head_indice+3)&(max(indices)-indices>=3)]
            head_indice = random.choice(head_indices)
            sampled += [head_indice,head_indice+3]
            positions1.append(head_indice)
            positions2.append(head_indice+3)

            
            sampled_indices+=sampled
        sampled_indices = torch.tensor(sampled_indices).to(atom_gt.device)
        atom_full = torch.zeros((mask_generate.shape[0],atom_gt.shape[1])).to(atom_gt.device)
        one_hot_vector = torch.zeros(20).to(atom_gt.device)
        one_hot_vector[4] = 1
        one_hot_vector = one_hot_vector.unsqueeze(0).repeat(len(sampled_indices),1)
        atom_full[sampled_indices] = one_hot_vector

        positions1 = torch.stack(positions1)
        positions2 = torch.stack(positions2)
        edges = torch.stack([positions1, positions2], dim=0)
        reversed_edges = edges.flip(0)

        sampled_edges = torch.cat([edges,reversed_edges], dim=1)
        guidance_edge_attr = self._get_edge_dist(X_true, sampled_edges, atom_mask)       
        guidance_edge_attr.fill_(3.8)      
        guidance_edge_attr = self.guidance_dist_rbf(guidance_edge_attr).view(sampled_edges.shape[1], -1)

        return atom_full,sampled_edges,guidance_edge_attr
    
    def condition333(self,atom_gt,batch_ids,mask_generate,X_true,atom_mask):
        '''
        two positions Cys with distance between 3.5-5 A *3
        '''
        unique_vals = torch.unique(batch_ids)
        sampled_indices = []
        positions1 = []
        positions2 = []
        for val in unique_vals:
            valid_indices = (batch_ids == val) & mask_generate
            indices = valid_indices.nonzero(as_tuple=True)[0]
            
            if len(indices)<12:
                continue
            head_indices = indices[(max(indices)-indices>=11)]
            head_indice = random.choice(head_indices)
            sampled = [head_indice,head_indice+3]
            positions1.append(head_indice)
            positions2.append(head_indice+3)
            
            head_indices = indices[(indices>head_indice+3)&(max(indices)-indices>=7)]
            head_indice = random.choice(head_indices)
            sampled += [head_indice,head_indice+3]
            positions1.append(head_indice)
            positions2.append(head_indice+3)
            sampled_indices+=sampled

            head_indices = indices[(indices>head_indice+3)&(max(indices)-indices>=3)]
            head_indice = random.choice(head_indices)
            sampled += [head_indice,head_indice+3]
            positions1.append(head_indice)
            positions2.append(head_indice+3)
            sampled_indices+=sampled
        sampled_indices = torch.tensor(sampled_indices).to(atom_gt.device)
        atom_full = torch.zeros((mask_generate.shape[0],atom_gt.shape[1])).to(atom_gt.device)
        one_hot_vector = torch.zeros(20).to(atom_gt.device)
        one_hot_vector[4] = 1
        one_hot_vector = one_hot_vector.unsqueeze(0).repeat(len(sampled_indices),1)
        atom_full[sampled_indices] = one_hot_vector

        positions1 = torch.stack(positions1)
        positions2 = torch.stack(positions2)
        edges = torch.stack([positions1, positions2], dim=0)
        reversed_edges = edges.flip(0)

        sampled_edges = torch.cat([edges,reversed_edges], dim=1)
        guidance_edge_attr = self._get_edge_dist(X_true, sampled_edges, atom_mask)       
        guidance_edge_attr.fill_(3.8)      
        guidance_edge_attr = self.guidance_dist_rbf(guidance_edge_attr).view(sampled_edges.shape[1], -1)

        return atom_full,sampled_edges,guidance_edge_attr
    
    def condition4(self,atom_gt,batch_ids,mask_generate,X_true,atom_mask):
        '''
        Contruct Bicycle
        '''
        unique_vals = torch.unique(batch_ids)
        sampled_indices = []
        positions1 = []
        positions2 = []
        for val in unique_vals:
            valid_indices = (batch_ids == val) & mask_generate
            indices = valid_indices.nonzero(as_tuple=True)[0]
            if len(indices)<13:
                continue
            bicycle_indices = indices[(max(indices)-indices>=12)]
            bicycle_indice = random.choice(bicycle_indices)
            sampled = [bicycle_indice,bicycle_indice+6,bicycle_indice+12]
            positions1.append(bicycle_indice)
            positions1.append(bicycle_indice)
            positions1.append(bicycle_indice+6)
            positions2.append(bicycle_indice+6)
            positions2.append(bicycle_indice+12)
            positions2.append(bicycle_indice+12)
            sampled_indices+=sampled
        sampled_indices = torch.tensor(sampled_indices).to(atom_gt.device)
        atom_full = torch.zeros((mask_generate.shape[0],atom_gt.shape[1])).to(atom_gt.device)
        one_hot_vector = torch.zeros(20).to(atom_gt.device)
        one_hot_vector[4] = 1
        one_hot_vector = one_hot_vector.unsqueeze(0).repeat(len(sampled_indices),1)
        atom_full[sampled_indices] = one_hot_vector

        positions1 = torch.stack(positions1)
        positions2 = torch.stack(positions2)
        edges = torch.stack([positions1, positions2], dim=0)
        reversed_edges = edges.flip(0)

        sampled_edges = torch.cat([edges,reversed_edges], dim=1)
        guidance_edge_attr = self._get_edge_dist(X_true, sampled_edges, atom_mask)       
        guidance_edge_attr.fill_(8)      
        guidance_edge_attr = self.guidance_dist_rbf(guidance_edge_attr).view(sampled_edges.shape[1], -1)

        return atom_full,sampled_edges,guidance_edge_attr

        

    
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