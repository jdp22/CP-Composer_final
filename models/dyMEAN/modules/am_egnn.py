#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn

from utils.decorators import singleton
from torch.nn import MultiheadAttention
import random

from .radial_basis import RadialBasis
from copy import deepcopy


class RadialLinear(nn.Module):
    def __init__(self, n_rbf, cutoff):
        super().__init__()
        self.rbf = RadialBasis(n_rbf, cutoff)
        self.linear = nn.Linear(n_rbf, 1)

    def forward(self, d):
        '''
        args:
            d: distance feature [N, ...]
        returns:
            radial: the same shape with input d, [N, ...]
        '''
        output_shape = d.shape
        radial = self.rbf(d.view(-1))  # [N*d1*d2..., n_rbf]
        radial = self.linear(radial).squeeze(-1)
        return radial.view(*output_shape)


class AMEGNN(nn.Module):

    def __init__(self, in_node_nf, hidden_nf, out_node_nf, n_channel, channel_nf,
                 radial_nf, in_edge_nf=0, act_fn=nn.SiLU(), n_layers=4,
                 residual=True, dropout=0.1, dense=False, n_rbf=0, cutoff=1.0):
        super().__init__()
        '''
        :param in_node_nf: Number of features for 'h' at the input
        :param hidden_nf: Number of hidden features
        :param out_node_nf: Number of features for 'h' at the output
        :param n_channel: Number of channels of coordinates
        :param in_edge_nf: Number of features for the edge features
        :param act_fn: Non-linearity
        :param n_layers: Number of layer for the EGNN
        :param residual: Use residual connections, we recommend not changing this one
        :param dropout: probability of dropout
        :param dense: if dense, then context states will be concatenated for all layers,
                      coordination will be averaged
        '''
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers

        self.dropout = nn.Dropout(dropout)

        self.linear_in = nn.Linear(in_node_nf, self.hidden_nf)

        self.dense = dense
        if dense:
            self.linear_out = nn.Linear(self.hidden_nf * (n_layers + 1), out_node_nf)
        else:
            self.linear_out = nn.Linear(self.hidden_nf, out_node_nf)

        for i in range(0, n_layers):
            self.add_module(f'gcl_{i}', AM_E_GCL(
                self.hidden_nf, self.hidden_nf, self.hidden_nf, n_channel, channel_nf, radial_nf,
                edges_in_d=in_edge_nf, act_fn=act_fn, residual=residual, dropout=dropout, n_rbf=n_rbf, cutoff=cutoff
            ))
            self.add_module(f'guidance_gcl_{i}', AM_E_GCL(
                self.hidden_nf, self.hidden_nf, self.hidden_nf, n_channel, channel_nf, radial_nf,
                edges_in_d=in_edge_nf//2, act_fn=act_fn, residual=residual, dropout=dropout, n_rbf=n_rbf, cutoff=cutoff
            ))
        self.out_layer = AM_E_GCL(
            self.hidden_nf, self.hidden_nf, self.hidden_nf, n_channel, channel_nf,
            radial_nf, edges_in_d=in_edge_nf, act_fn=act_fn, residual=residual, n_rbf=n_rbf, cutoff=cutoff
        )
    
    def forward(self, h, x, edges, channel_attr, channel_weights, ctx_edge_attr=None, x_update_mask=None,guidance_edge_attr=None,guidance_edges=None):
        try:
            h = self.linear_in(h)
        except:
            breakpoint()
        h = self.dropout(h)

        ctx_states, ctx_coords = [], []
        # Add layer-wise attention 
        for i in range(0, self.n_layers):
            h, x = self._modules[f'gcl_{i}'](
                h, edges, x, channel_attr, channel_weights,
                edge_attr=ctx_edge_attr, x_update_mask=x_update_mask)
            if guidance_edges is not None:
                h, x = self._modules[f'guidance_gcl_{i}'](
                    h, guidance_edges, x, channel_attr, channel_weights,
                    edge_attr=guidance_edge_attr, x_update_mask=x_update_mask)
            # cross-attn 
            ctx_states.append(h)
            ctx_coords.append(x)

        h, x = self.out_layer(
            h, edges, x, channel_attr, channel_weights,
            edge_attr=ctx_edge_attr, x_update_mask=x_update_mask)
        ctx_states.append(h)
        ctx_coords.append(x)
        if self.dense:
            h = torch.cat(ctx_states, dim=-1)
            x = torch.mean(torch.stack(ctx_coords), dim=0)
        h = self.dropout(h)
        h = self.linear_out(h)
        return h, x

class Prompt_AMEGNN(nn.Module):

    def __init__(self, in_node_nf, hidden_nf, out_node_nf, n_channel, channel_nf,
                 radial_nf, in_edge_nf=0, act_fn=nn.SiLU(), n_layers=4,
                 residual=True, dropout=0.1, dense=False, n_rbf=0, cutoff=1.0):
        super().__init__()
        '''
        :param in_node_nf: Number of features for 'h' at the input
        :param hidden_nf: Number of hidden features
        :param out_node_nf: Number of features for 'h' at the output
        :param n_channel: Number of channels of coordinates
        :param in_edge_nf: Number of features for the edge features
        :param act_fn: Non-linearity
        :param n_layers: Number of layer for the EGNN
        :param residual: Use residual connections, we recommend not changing this one
        :param dropout: probability of dropout
        :param dense: if dense, then context states will be concatenated for all layers,
                      coordination will be averaged
        '''
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers

        self.dropout = nn.Dropout(dropout)

        self.linear_in = nn.Linear(in_node_nf, self.hidden_nf)

        self.dense = dense
        if dense:
            self.linear_out = nn.Linear(self.hidden_nf * (n_layers + 1), out_node_nf)
        else:
            self.linear_out = nn.Linear(self.hidden_nf, out_node_nf)

        for i in range(0, n_layers):
            self.add_module(f'gcl_{i}', AM_E_GCL(
                self.hidden_nf, self.hidden_nf, self.hidden_nf, n_channel, channel_nf, radial_nf,
                edges_in_d=in_edge_nf, act_fn=act_fn, residual=residual, dropout=dropout, n_rbf=n_rbf, cutoff=cutoff
            ))
            self.add_module(f'attn_1{i}',MultiheadAttention(self.hidden_nf,1,kdim = 768,vdim = 768,batch_first=True))
            self.add_module(f'attn_2{i}',MultiheadAttention(self.hidden_nf,1,kdim = 768,vdim = 768,batch_first=True))
            # self.add_module(f'mlp_{i}',nn.Linear(768,self.hidden_nf))
            self.add_module(f'mix_{i}',nn.Linear(self.hidden_nf*3,self.hidden_nf))
        self.out_layer = AM_E_GCL(
            self.hidden_nf, self.hidden_nf, self.hidden_nf, n_channel, channel_nf,
            radial_nf, edges_in_d=in_edge_nf, act_fn=act_fn, residual=residual, n_rbf=n_rbf, cutoff=cutoff
        )

    def q_padding(self,Nodes, batch_ids):
        """
        将 A[N, H] 张量根据 B[N, 1] 分组，变为 [num_samples, L, H] 张量。
        :param A: 输入特征张量，形状为 (N, H)
        :param B: 样本归属标识张量，形状为 (N, 1)
        :return: 重组后的张量，形状为 (num_samples, max_length, H)
        """

        unique_samples, counts = torch.unique(batch_ids, return_counts=True)
        num_samples = len(unique_samples)
        max_length = counts.max().item()
        N, H = Nodes.shape
        grouped_tensor = torch.full((num_samples, max_length, H),0, dtype=Nodes.dtype, device=Nodes.device)
        attn_mask = torch.zeros((num_samples, max_length),dtype=bool, device=Nodes.device)
        for i, sample_id in enumerate(unique_samples):
            indices = (batch_ids == sample_id).nonzero(as_tuple=True)[0]
            grouped_tensor[i, :len(indices), :] = Nodes[indices]
            attn_mask[sample_id,:len(indices)] = True
        
        return grouped_tensor,attn_mask
    
    def forward(self, h, x,prompt, edges, channel_attr, channel_weights,key_mask_list, ctx_edge_attr=None, x_update_mask=None,batch_ids=None,text_guidance=False,inference=False):
        h = self.linear_in(h)
        h = self.dropout(h)

        ctx_states, ctx_coords = [], []
        # Add layer-wise attention 
        if random.random()<0.5:
            guidance1 = True
        else:
            guidance1 = False
        for i in range(0, self.n_layers):
            h, x = self._modules[f'gcl_{i}'](
                h, edges, x, channel_attr, channel_weights,
                edge_attr=ctx_edge_attr, x_update_mask=x_update_mask)
            # cross-attn 
            if text_guidance:
                h_padding,q_mask = self.q_padding(h,batch_ids)
                q_mask = q_mask.unsqueeze(-1)
                if guidance1 or inference:
                    key_mask = key_mask_list['prompt1_mask']
                    # attn_mask = q_mask&key_mask
                    h_prompt1,_ = self._modules[f'attn_1{i}'](h_padding,prompt['prompt1'],prompt['prompt1'],key_padding_mask = ~key_mask)
                    h_prompt1 = h_prompt1[q_mask.squeeze(-1)]
                else:
                    h_prompt1 = torch.zeros(h.shape[0],self.hidden_nf).to(h.device)
                # if random.random()<0.5 or inference:
                if False:
                    key_mask = key_mask_list['prompt2_mask'].unsqueeze(1)
                    attn_mask = q_mask&key_mask
                    h_prompt2,_ = self._modules[f'attn_2{i}'](h_padding,prompt['prompt2'],prompt['prompt2'],attn_mask = ~attn_mask)
                    h_prompt2 = h_prompt2[q_mask.squeeze(-1)]
                else:
                    h_prompt2 = torch.zeros(h.shape[0],self.hidden_nf).to(h.device)
                h = self._modules[f'mix_{i}'](torch.concat([h,h_prompt1,h_prompt2],dim=-1))
            else:
                h_prompt1 = torch.zeros(h.shape[0],self.hidden_nf).to(h.device)
                h_prompt2 = torch.zeros(h.shape[0],self.hidden_nf).to(h.device)
                h = self._modules[f'mix_{i}'](torch.concat([h,h_prompt1,h_prompt2],dim=-1))
            ctx_states.append(h)
            ctx_coords.append(x)

        h, x = self.out_layer(
            h, edges, x, channel_attr, channel_weights,
            edge_attr=ctx_edge_attr, x_update_mask=x_update_mask)
        ctx_states.append(h)
        ctx_coords.append(x)
        if self.dense:
            h = torch.cat(ctx_states, dim=-1)
            x = torch.mean(torch.stack(ctx_coords), dim=0)
        h = self.dropout(h)
        h = self.linear_out(h)
        return h, x


'''
Below are the implementation of the adaptive multi-channel message passing mechanism
'''

@singleton
class RollerPooling(nn.Module):
    '''
    Adaptive average pooling for the adaptive scaler
    '''
    def __init__(self, n_channel) -> None:
        super().__init__()
        self.n_channel = n_channel
        with torch.no_grad():
            pool_matrix = []
            ones = torch.ones((n_channel, n_channel), dtype=torch.float)
            for i in range(n_channel):
                # i start from 0 instead of 1 !!! (less readable but higher implemetation efficiency)
                window_size = n_channel - i
                mat = torch.triu(ones) - torch.triu(ones, diagonal=window_size)
                pool_matrix.append(mat / window_size)
            self.pool_matrix = torch.stack(pool_matrix)
    
    def forward(self, hidden, target_size):
        '''
        :param hidden: [n_edges, n_channel]
        :param target_size: [n_edges]
        '''
        pool_mat = self.pool_matrix.to(hidden.device).type(hidden.dtype)
        pool_mat = pool_mat[target_size - 1]  # [n_edges, n_channel, n_channel]
        hidden = hidden.unsqueeze(-1)  # [n_edges, n_channel, 1]
        return torch.bmm(pool_mat, hidden)  # [n_edges, n_channel, 1]


class AM_E_GCL(nn.Module):
    '''
    Adaptive Multi-Channel E(n) Equivariant Convolutional Layer
    '''

    def __init__(self, input_nf, output_nf, hidden_nf, n_channel, channel_nf, radial_nf,
                 edges_in_d=0, node_attr_d=0, act_fn=nn.SiLU(), residual=True, attention=False,
                 normalize=False, coords_agg='mean', tanh=False, dropout=0.1, n_rbf=0, cutoff=1.0):
        super(AM_E_GCL, self).__init__()

        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8

        self.dropout = nn.Dropout(dropout)

        input_edge = input_nf * 2
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + radial_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)
        self.radial_linear = nn.Linear(channel_nf ** 2, radial_nf)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + node_attr_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        layer = nn.Linear(hidden_nf, n_channel, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())
            
        if n_rbf > 1:
            self.rbf_linear = RadialLinear(n_rbf, cutoff)

    def edge_model(self, source, target, radial, edge_attr):
        '''
        :param source: [n_edge, input_size]
        :param target: [n_edge, input_size]
        :param radial: [n_edge, d, d]
        :param edge_attr: [n_edge, edge_dim]
        '''
        radial = radial.reshape(radial.shape[0], -1)  # [n_edge, d ^ 2]

        if edge_attr is None:  # Unused.
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        out = self.dropout(out)

        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        '''
        :param x: [bs * n_node, input_size]
        :param edge_index: list of [n_edge], [n_edge]
        :param edge_attr: [n_edge, hidden_size], refers to message from i to j
        :param node_attr: [bs * n_node, node_dim]
        '''
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))  # [bs * n_node, hidden_size]
        # print_log(f'agg1, {torch.isnan(agg).sum()}', level='DEBUG')
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)  # [bs * n_node, input_size + hidden_size]
        # print_log(f'agg, {torch.isnan(agg).sum()}', level='DEBUG')
        out = self.node_mlp(agg)  # [bs * n_node, output_size]
        # print_log(f'out, {torch.isnan(out).sum()}', level='DEBUG')
        out = self.dropout(out)
        if self.residual:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat, channel_attr, channel_weights, x_update_mask=None):
        '''
        coord: [N, n_channel, d]
        edge_index: list of [n_edge], [n_edge]
        coord_diff: [n_edge, n_channel, d]
        edge_feat: [n_edge, hidden_size]
        channel_attr: [N, n_channel, channel_nf]
        channel_weights: [N, n_channel]
        x_update_mask: [N, n_channel], 1 for updating coordinates
        '''
        row, col = edge_index

        # first pooling, then element-wise multiply
        n_channel = channel_weights.shape[-1]
        edge_feat = self.coord_mlp(edge_feat)  # [n_edge, n_channel]
        channel_sum = (channel_weights != 0).long().sum(-1)  # [N]
        pooled_edge_feat = RollerPooling(n_channel)(edge_feat, channel_sum[row])  # [n_edge, n_channel, 1]
        
        trans = coord_diff * pooled_edge_feat  # [n_edge, n_channel, d]

        # aggregate
        if self.coords_agg == 'sum':
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == 'mean':
            agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))  # [N, n_channel, d]
        else:
            raise Exception('Wrong coords_agg parameter' % self.coords_agg)
        if x_update_mask is None:
            coord = coord + agg
        else:
            x_update_mask = x_update_mask.unsqueeze(-1).float()  # [N, n_channel, 1]
            coord = coord + x_update_mask * agg
        return coord

    def forward(self, h, edge_index, coord, channel_attr, channel_weights,
                edge_attr=None, node_attr=None, x_update_mask=None):
        '''
        h: [bs * n_node, hidden_size]
        edge_index: list of [n_row] and [n_col] where n_row == n_col (with no cutoff, n_row == bs * n_node * (n_node - 1))
        coord: [bs * n_node, n_channel, d]
        channel_attr: [bs * n_node, n_channel, channel_nf]
        channel_weights: [bs * n_node, n_channel]
        x_update_mask: [bs * n_node, n_channel], 1 for updating coordinates
        '''
        row, col = edge_index

        radial, coord_diff = coord2radial(edge_index, coord, channel_attr, channel_weights, self.radial_linear, getattr(self, 'rbf_linear', None))

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)  # [n_edge, hidden_size]
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat, channel_attr, channel_weights, x_update_mask)    # [bs * n_node, n_channel, d]
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        return h, coord


def unsorted_segment_sum(data, segment_ids, num_segments):
    '''
    :param data: [n_edge, *dimensions]
    :param segment_ids: [n_edge]
    :param num_segments: [bs * n_node]
    '''
    expand_dims = tuple(data.shape[1:])
    result_shape = (num_segments, ) + expand_dims
    for _ in expand_dims:
        segment_ids = segment_ids.unsqueeze(-1)
    segment_ids = segment_ids.expand(-1, *expand_dims)
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    '''
    :param data: [n_edge, *dimensions]
    :param segment_ids: [n_edge]
    :param num_segments: [bs * n_node]
    '''
    expand_dims = tuple(data.shape[1:])
    result_shape = (num_segments, ) + expand_dims
    for _ in expand_dims:
        segment_ids = segment_ids.unsqueeze(-1)
    segment_ids = segment_ids.expand(-1, *expand_dims)
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


CONSTANT = 1
NUM_SEG = 1  # if you do not have enough memory or you have large attr_size, increase this parameter

def coord2radial(edge_index, coord, attr, channel_weights, linear_map, rbf_linear=None):
    '''
    :param edge_index: tuple([n_edge], [n_edge]) which is tuple of (row, col)
    :param coord: [N, n_channel, d]
    :param attr: [N, n_channel, attr_size], attribute embedding of each channel
    :param channel_weights: [N, n_channel], weights of different channels
    :param linear_map: nn.Linear, map features to d_out
    :param num_seg: split row/col into segments to reduce memory cost
    '''
    row, col = edge_index
    
    radials = []

    seg_size = (len(row) + NUM_SEG - 1) // NUM_SEG

    for i in range(NUM_SEG):
        start = i * seg_size
        end = min(start + seg_size, len(row))
        if end <= start:
            break
        seg_row, seg_col = row[start:end], col[start:end]

        coord_msg = torch.norm(
            coord[seg_row].unsqueeze(2) - coord[seg_col].unsqueeze(1),  # [n_edge, n_channel, n_channel, d]
            dim=-1, keepdim=False)  # [n_edge, n_channel, n_channel]
        if rbf_linear:
            coord_msg = rbf_linear(coord_msg)
        
        coord_msg = coord_msg * torch.bmm(
            channel_weights[seg_row].unsqueeze(2),
            channel_weights[seg_col].unsqueeze(1)
            )  # [n_edge, n_channel, n_channel]
        
        radial = torch.bmm(
            attr[seg_row].transpose(-1, -2),  # [n_edge, attr_size, n_channel]
            coord_msg)  # [n_edge, attr_size, n_channel]
        radial = torch.bmm(radial, attr[seg_col])  # [n_edge, attr_size, attr_size]
        radial = radial.reshape(radial.shape[0], -1)  # [n_edge, attr_size * attr_size]
        if rbf_linear:  # do not need normalization
            radial = linear_map(radial)
        else:
            radial_norm = torch.norm(radial, dim=-1, keepdim=True) + CONSTANT  # post norm
            radial = linear_map(radial) / radial_norm  # [n_edge, d_out]

        radials.append(radial)
    
    radials = torch.cat(radials, dim=0)  # [N_edge, d_out]

    # generate coord_diff by first mean src then minused by dst
    # message passed from col to row
    channel_mask = (channel_weights != 0).long()  # [N, n_channel]
    channel_sum = channel_mask.sum(-1)  # [N]
    pooled_col_coord = (coord[col] * channel_mask[col].unsqueeze(-1)).sum(1)  # [n_edge, d]
    pooled_col_coord = pooled_col_coord / channel_sum[col].unsqueeze(-1)  # [n_edge, d], denominator cannot be 0 since no pad node exists
    coord_diff = coord[row] - pooled_col_coord.unsqueeze(1)  # [n_edge, n_channel, d]

    return radials, coord_diff