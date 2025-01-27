#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import torch.nn.functional as F

from data.format import VOCAB
from utils.nn_utils import graph_to_batch


@torch.no_grad()
def continuous_bool(x, k=1000):
    return (x > 0).float()


def _consec_dist_loss(gen_X, gen_X_mask, lb, ub, eps=1e-6):
    consec_dist = torch.norm(gen_X[..., 1:, :] - gen_X[..., :-1, :], dim=-1) # [bs, max_L - 1]
    consec_lb_loss = lb - consec_dist  # [bs, max_L - 1]
    consec_ub_loss = consec_dist - ub  # [bs, max_L - 1]

    consec_lb_invalid = (consec_dist < lb) & gen_X_mask[..., 1:]
    consec_ub_invalid = (consec_dist > ub) & gen_X_mask[..., 1:]
    consec_loss = torch.where(consec_lb_invalid, consec_lb_loss, torch.zeros_like(consec_lb_loss))
    consec_loss = torch.where(consec_ub_invalid, consec_ub_loss, consec_loss)

    consec_loss = consec_loss.sum(-1) / (consec_lb_invalid + consec_ub_invalid + eps).sum(-1)
    consec_loss = torch.sum(consec_loss) # consistent loss scale across different batch size
    return consec_loss


def _inner_clash_loss(gen_X, gen_X_mask, mean, eps=1e-6):
    dist = torch.norm(gen_X[..., :, None, :] - gen_X[..., None, :, :], dim=-1) # [bs, max_L, max_L]
    dist_mask = gen_X_mask[..., :, None] & gen_X_mask[..., None, :] # [bs, max_L, max_L]
    pos = torch.cumsum(torch.ones_like(gen_X_mask, dtype=torch.long), dim=-1) # [bs, max_L]
    non_consec_mask = torch.abs(pos[..., :, None] - pos[..., None, :]) > 1  # [bs, max_L, max_L]

    clash_loss = mean - dist
    clash_loss_mask = (clash_loss > 0) & dist_mask & non_consec_mask # [bs, max_L, max_L]
    clash_loss = torch.where(clash_loss_mask, clash_loss, torch.zeros_like(clash_loss))

    clash_loss = clash_loss.sum(-1).sum(-1) / (clash_loss_mask.sum(-1).sum(-1) + eps)
    clash_loss = torch.sum(clash_loss)  # consistent loss scale across different residue number and batch size
    return clash_loss


def _outer_clash_loss(ctx_X, ctx_X_mask, gen_X, gen_X_mask, mean, eps=1e-6):
    dist = torch.norm(gen_X[..., :, None, :] - ctx_X[..., None, :, :], dim=-1)  # [bs, max_gen_L, max_ctx_L]
    dist_mask = gen_X_mask[..., :, None] & ctx_X_mask[..., None, :] # [bs, max_gen_L, max_ctx_L]
    clash_loss = mean - dist  # [bs, max_gen_L, max_ctx_L]
    clash_loss_mask = (clash_loss > 0) & dist_mask  # [bs, max_gen_L, max_ctx_L]
    clash_loss = torch.where(clash_loss_mask, clash_loss, torch.zeros_like(clash_loss))

    clash_loss = clash_loss.sum(-1).sum(-1) / (clash_loss_mask.sum(-1).sum(-1) + eps)
    clash_loss = torch.sum(clash_loss)  # consistent loss scale across different residue number and batch size
    return clash_loss


def dist_energy(X, mask_generate, batch_ids, mean, std, tolerance=3, **kwargs):
    breakpoint()
    mean, std = round(mean, 4), round(std, 4)
    lb, ub = mean - tolerance * std, mean + tolerance * std

    X = X.clone() # [N, 3]

    ctx_X, ctx_batch_ids = X[~mask_generate], batch_ids[~mask_generate]
    gen_X, gen_batch_ids = X[mask_generate], batch_ids[mask_generate]
    ctx_X = ctx_X[:, VOCAB.ca_channel_idx] # CA (alpha carbon)
    gen_X = gen_X[:, 0] # latent one

    # to batch representation
    ctx_X, ctx_X_mask = graph_to_batch(ctx_X, ctx_batch_ids, mask_is_pad=False) # [bs, max_ctx_L, 3]
    gen_X, gen_X_mask = graph_to_batch(gen_X, gen_batch_ids, mask_is_pad=False) # [bs, max_gen_L, 3]

    # consecutive
    consec_loss = _consec_dist_loss(gen_X, gen_X_mask, lb, ub)

    # inner clash
    inner_clash_loss = _inner_clash_loss(gen_X, gen_X_mask, mean)

    # outer clash
    outer_clash_loss = _outer_clash_loss(ctx_X, ctx_X_mask, gen_X, gen_X_mask, mean)

    return consec_loss + inner_clash_loss + outer_clash_loss

def condition1_guidance(X, H,atom_gt,mask_generate, batch_ids,mean, std, tolerance=3, lb=0, ub=3.5, eps=1e-6,**kwargs):
    device = X.device
    unique_vals = torch.unique(batch_ids)
    sampled_indices = []
    positions1 = []
    positions2 = []
    for val in unique_vals:
        valid_indices = (batch_ids == val) & mask_generate
        indices = valid_indices.nonzero(as_tuple=True)[0]
        if len(indices)<5:
            continue
        indice = indices[0]
        sampled = [indice,indice+3]
        positions1.append(indice)
        positions2.append(indice+3)
        sampled_indices+=sampled
    
    positions1 = torch.stack(positions1).to(device)
    positions2 = torch.stack(positions2).to(device)

    sampled_indices = torch.tensor(sampled_indices).to(device)
    H_source = H[sampled_indices]
    H_target = torch.zeros_like(H,dtype=atom_gt['K'].dtype)
    H_target_K = atom_gt['K'].repeat(positions1.shape[0],1).to(device)
    H_target[positions1] = H_target_K
    H_target_D = atom_gt['D'].repeat(positions2.shape[0],1).to(device)
    H_target[positions2] = H_target_D
    H_target = H_target[sampled_indices]
    
    node_loss = torch.mean((H_source-H_target)**2)

    # 计算首尾距离
    head_coords = X[positions1,0]
    tail_coords = X[positions2,0]
    ht_dist = torch.norm(head_coords - tail_coords, dim=-1)  # [num_batches]

    # 计算损失：超出范围的距离
    ht_loss = torch.zeros_like(ht_dist)
    invalid_lb_mask = ht_dist < lb
    invalid_ub_mask = ht_dist > ub
    ht_loss[invalid_lb_mask] = lb - ht_dist[invalid_lb_mask]
    ht_loss[invalid_ub_mask] = ht_dist[invalid_ub_mask] - ub

    ht_loss = torch.sum(ht_loss)/(invalid_lb_mask+invalid_ub_mask+eps).sum()

    return ht_loss+node_loss


def condition2_guidance(X, H,atom_gt,mask_generate, batch_ids,mean, std, tolerance=3, lb=0, ub=3.5, eps=1e-6,**kwargs):
    """
    约束首尾距离在 [lb, ub] 范围内
    X: Tensor of shape [N, 3], 包含点集的3D坐标
    mask_generate: Bool tensor of shape [N], 表示生成点的掩码
    batch_ids: Tensor of shape [N], 表示点的批次ID
    lb: 距离下界
    ub: 距离上界
    eps: 防止除以零的稳定因子
    """
    device = X.device

    head_positions = []
    tail_positions = []
    for i in range(1, len(mask_generate)):
        if mask_generate[i] != mask_generate[i-1]:
            if mask_generate[i]:
                head_positions.append(i)
            else:
                tail_positions.append(i-1)
    tail_positions.append(len(mask_generate)-1)

    # 转换为张量
    head_positions = torch.tensor(head_positions, device=device)
    tail_positions = torch.tensor(tail_positions, device=device)

    # 计算首尾距离
    head_coords = X[head_positions,0]
    tail_coords = X[tail_positions,0]
    ht_dist = torch.norm(head_coords - tail_coords, dim=-1)  # [num_batches]

    # 计算损失：超出范围的距离
    ht_loss = torch.zeros_like(ht_dist)
    invalid_lb_mask = ht_dist < lb
    invalid_ub_mask = ht_dist > ub
    ht_loss[invalid_lb_mask] = lb - ht_dist[invalid_lb_mask]
    ht_loss[invalid_ub_mask] = ht_dist[invalid_ub_mask] - ub

    ht_loss = torch.sum(ht_loss)/(invalid_lb_mask+invalid_ub_mask+eps).sum()
    return ht_loss

def condition3_guidance(X, H,atom_gt,mask_generate, batch_ids,mean, std, tolerance=3, lb=0, ub=3, eps=1e-6,**kwargs):
    '''
    two positions Cys with distance between 3.5-5 A
    '''

    device = X.device
    unique_vals = torch.unique(batch_ids)
    sampled_indices = []
    positions1 = []
    positions2 = []
    for val in unique_vals:
        valid_indices = (batch_ids == val) & mask_generate
        indices = valid_indices.nonzero(as_tuple=True)[0]
        
        if len(indices)<4:
            continue
        head_indice = indices[0]
        sampled = [head_indice,head_indice+3]
        positions1.append(head_indice)
        positions2.append(head_indice+3)
        sampled_indices+=sampled
    H_target = atom_gt['C'].to(device).repeat(len(sampled_indices),1)
    sampled_indices = torch.tensor(sampled_indices).to(device)
    H_source = H[sampled_indices]
    node_loss = torch.mean((H_source-H_target)**2)

    positions1 = torch.stack(positions1).to(device)
    positions2 = torch.stack(positions2).to(device)

    # 计算首尾距离
    head_coords = X[positions1,0]
    tail_coords = X[positions2,0]
    ht_dist = torch.norm(head_coords - tail_coords, dim=-1)  # [num_batches]

    # 计算损失：超出范围的距离
    ht_loss = torch.zeros_like(ht_dist)
    invalid_lb_mask = ht_dist < lb
    invalid_ub_mask = ht_dist > ub
    ht_loss[invalid_lb_mask] = lb - ht_dist[invalid_lb_mask]
    ht_loss[invalid_ub_mask] = ht_dist[invalid_ub_mask] - ub

    ht_loss = torch.sum(ht_loss)/(invalid_lb_mask+invalid_ub_mask+eps).sum()

    return ht_loss+node_loss

def condition4_guidance(X, H,atom_gt,mask_generate, batch_ids,mean, std, tolerance=3, lb=0, ub=4, eps=1e-6,**kwargs):
    device = X.device
    unique_vals = torch.unique(batch_ids)
    sampled_indices = []
    positions1 = []
    positions2 = []
    for val in unique_vals:
        valid_indices = (batch_ids == val) & mask_generate
        indices = valid_indices.nonzero(as_tuple=True)[0]
        if len(indices)<13:
            continue
        bicycle_indice = indices[0]
        sampled = [bicycle_indice,bicycle_indice+6,bicycle_indice+12]
        positions1.append(bicycle_indice)
        positions1.append(bicycle_indice)
        positions1.append(bicycle_indice+6)
        positions2.append(bicycle_indice+6)
        positions2.append(bicycle_indice+12)
        positions2.append(bicycle_indice+12)
        sampled_indices+=sampled
    H_target = atom_gt['C'].to(device).repeat(len(sampled_indices),1)
    sampled_indices = torch.tensor(sampled_indices).to(device)
    H_source = H[sampled_indices]
    node_loss = torch.mean((H_source-H_target)**2)

    positions1 = torch.stack(positions1).to(device)
    positions2 = torch.stack(positions2).to(device)

    # 计算首尾距离
    head_coords = X[positions1,0]
    tail_coords = X[positions2,0]
    ht_dist = torch.norm(head_coords - tail_coords, dim=-1)  # [num_batches]

    # 计算损失：超出范围的距离
    ht_loss = torch.zeros_like(ht_dist)
    invalid_lb_mask = ht_dist < lb
    invalid_ub_mask = ht_dist > ub
    ht_loss[invalid_lb_mask] = lb - ht_dist[invalid_lb_mask]
    ht_loss[invalid_ub_mask] = ht_dist[invalid_ub_mask] - ub

    ht_loss = torch.sum(ht_loss)/(invalid_lb_mask+invalid_ub_mask+eps).sum()

    return ht_loss+node_loss

