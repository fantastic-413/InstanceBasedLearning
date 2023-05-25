#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import average_precision_score

from torch.utils.data import DataLoader

from dataloader import TestDataset
import math
from typing import Union, Iterable
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
torch.autograd.set_detect_anomaly(True)


ACTFN = {
    'none': lambda x: x,
    'tanh': torch.tanh,
}


# 确保输入的张量大于零
def at_least_eps(x: torch.FloatTensor) -> torch.FloatTensor:
    """Make sure a tensor is greater than zero. """
    # get datatype specific epsilon 获取张量数据类型的 epsilon 值（即可以表示数据类型的最小值）
    eps = torch.finfo(x.dtype).eps
    # clamp minimum value 使用 clamp 函数将张量中的值限制在这个 epsilon 值以上，以确保所有的值都不小于 epsilon 值
    return x.clamp(min=eps)


# 确保张量的范数不超过某个阈值。
def clamp_norm(
    x: torch.Tensor,
    maxnorm: float,
    p: Union[str, int] = "fro",
    dim: Union[None, int, Iterable[int]] = None,
) -> torch.Tensor:
    """Ensure that a tensor's norm does not exceeds some threshold. 

    :param x:
        The vector.
    :param maxnorm:
        The maximum norm (>0).
    :param p:
        The norm type.
    :param dim:
        The dimension(s).

    :return:
        A vector with $|x| <= maxnorm$.
    """
    norm = x.norm(p=p, dim=dim, keepdim=True)
    # 如果张量的范数小于等于 maxnorm，那么对应位置的 mask 就是 1，否则为 0。
    mask = (norm < maxnorm).type_as(x)
    # 如果掩码为0，表示张量的范数超过了maxnorm，此时将对张量进行归一化，使得其范数不超过maxnorm。如果norm为0，为避免除以0的错误，应将其设置为一个很小的正数eps。
    return mask * x + (1 - mask) * (x / at_least_eps(norm) * maxnorm)

# 它的主要功能是实现一个基于图卷积网络（GCN）的实体链接模型。给定一个实体和一个关系，模型的目标是预测与该实体相关联的实体。
# 模型使用了两个聚合器（aggregator1和aggregator2），分别在两个方向上执行信息传递（消息传递）操作，以捕捉实体之间的关系。
class IBLE(nn.Module):
    def __init__(self, args, edges, nrelation, nentity, gamma, epsilon, mlp=False, relation_aware=False, entity_dim=None):
        super().__init__()
        # self.config：保存该模型的参数配置。
        # self.use_mlp：用于标识是否使用 MLP 神经网络。
        # self.relation_aware：用于标识是否使用关系感知嵌入。
        # self.m 和 self.n：分别表示边的数量和实体的数量。
        # self.node_edge1、self.edge_node1、self.node_edge2、self.edge_node2：分别表示两个不同的图中的节点和边之间的连接关系。
        # self.r_mask1、self.r_mask2：用于标识与关系相关的边。
        # self.gamma：表示 D 距离的边界。
        # self.graph：用于存储构建的图结构。
        # self.aggregator1、self.aggregator2：分别表示两个 MessagePassing 对象，用于实现图上节点的信息传递和聚合。
        # self.mlp：用于标识是否使用 MLP 神经网络。
        # self.head_r、self.tail_r：表示关系感知嵌入中的可训练参数。
        # self.device：表示模型所在的设备。
        # self.epsilon：表示一个小的值，用于避免分母为零。
        self.config = args
        self.use_mlp = mlp
        self.relation_aware = relation_aware
        self.m = len(edges)
        self.n = nentity
        self.node_edge1 = []
        self.edge_node1 = []
        self.r_mask1 = torch.zeros(nrelation, self.m).cuda()
        self.r_mask2 = torch.zeros(nrelation, self.m).cuda()
        self.node_edge2 = []
        self.edge_node2 = []
        self.gamma = gamma
        self.graph = {}
        # 根据传入的边列表（edges），构建一个图（self.graph）。
        for h,r,t in edges:
            if (r,t) not in self.graph:
                self.graph[(r,t)] = []
            self.graph[(r,t)].append(h)

        # 将每个实体与其相邻的边连接起来，以生成两个不同的图（node_edge1、edge_node1和node_edge2、edge_node2）。
        for i, (h,r,t) in enumerate(edges):
            self.node_edge1.append([h, i])
            self.edge_node1.append([i, t])
            # 对于每个关系，创建一个掩码（r_mask1、r_mask2），以标识在两个不同的方向上与关系相关的边。
            self.r_mask1[r][i] = 1
            self.r_mask2[r][i] = 1

            self.node_edge2.append([t, i])
            self.edge_node2.append([i, h])

        self.node_edge1 = torch.LongTensor(self.node_edge1).permute(1, 0).cuda()
        self.edge_node1 = torch.LongTensor(self.edge_node1).permute(1, 0).cuda()
        self.node_edge2 = torch.LongTensor(self.node_edge2).permute(1, 0).cuda()
        self.edge_node2 = torch.LongTensor(self.edge_node2).permute(1, 0).cuda()

        # 在图神经网络中，MessagePassing用于实现图上节点的信息传递和聚合。每次MessagePassing会将节点的特征向量沿着图的边进行传递，并聚合相邻节点的特征向量。
        # 在这个模型中，self.aggregator1使用add作为聚合方法，
        # 而self.aggregator2使用config中指定的pooling方法作为聚合方法。这两个MessagePassing对象将在模型的前向传播过程中使用。
        self.aggregator1 = MessagePassing(aggr='add')
        self.aggregator2 = MessagePassing(aggr=self.config.pooling)

        if self.use_mlp:
            # 如果self.use_mlp为True，那么IBL模型中会定义一个MLP（多层感知机）神经网络。这个MLP包含两个全连接层（nn.Linear），一个Tanh激活函数以及一个Dropout层（nn.Dropout）。
                # 第一个全连接层的输入维度为self.n，输出维度为min(2self.n, self.config.intermediate_dim).输出大小被限制在2 * self.n和self.config.intermediate_dim之间的较小值，这是为了限制模型的复杂度，防止模型过拟合。
                # 其中 self.config.intermediate_dim 隐藏层的维度或大小/节点数.
                # 第二个全连接层的输入维度为min(2self.n, self.config.intermediate_dim)，输出维度为self.n。
                # dropout 层的丢弃率为 0.1。
            self.mlp = nn.Sequential(
                nn.Linear(self.n, min(2 * self.n, self.config.intermediate_dim)),
                nn.Tanh(),
                nn.Dropout(p=0.1),
                nn.Linear(min(2 * self.n, self.config.intermediate_dim), self.n),
            )

        # 这段代码的作用是定义模型中的关系感知嵌入层
        # 在嵌入层中，通常包含两个关系向量：head_r和tail_r。这些向量表示了每个关系对应的头实体和尾实体，通过学习这些向量，可以更好地捕捉实体与关系之间的交互作用。
        # 首先，根据 entity_dim 和 config.double_entity_embedding 的取值计算出嵌入向量的维度 hidden_dim。
        hidden_dim = entity_dim * 2 if self.config.double_entity_embedding else entity_dim
        # 当 relation_aware 为 'diag' 时，我们使用对角矩阵嵌入方式。
        if self.relation_aware == 'diag':
            # nn.Parameter() 函数可以将嵌入向量标记为需要训练的参数。
            # 即对于每个关系，分别对头实体和尾实体采用一个对角矩阵进行嵌入。
            # 初始化时，采用均匀分布初始化，并乘以一个较小的数bound。
            self.head_r = nn.Parameter(torch.randn(nrelation, hidden_dim))
            bound = 1.0 * 6 / math.sqrt(hidden_dim)
            torch.nn.init.uniform_(self.head_r, -bound, bound)

            self.tail_r = nn.Parameter(torch.randn(nrelation, hidden_dim))
            bound = 1.0 * 6 / math.sqrt(hidden_dim)
            torch.nn.init.uniform_(self.tail_r, -bound, bound)
        elif self.relation_aware == 'matrix':
            # 当 relation_aware 为 'matrix' 时，我们使用矩阵嵌入方式。
            # 即对于每个关系，分别对头实体和尾实体采用一个矩阵进行嵌入。
            # 具体实现和对角矩阵类似，不同的是head_r和tail_r都是hidden_dim x hidden_dim的矩阵。
            self.head_r = nn.Parameter(torch.randn(nrelation, hidden_dim, hidden_dim))
            bound = 1.0 * 6 / math.sqrt(hidden_dim)
            torch.nn.init.uniform_(self.head_r, -bound, bound)

            self.tail_r = nn.Parameter(torch.randn(nrelation, hidden_dim, hidden_dim))
            bound = 1.0 * 6 / math.sqrt(hidden_dim)
            torch.nn.init.uniform_(self.tail_r, -bound, bound)

    # 这是一个用于关系嵌入表示学习的模型的前向传递函数。
    # 输入是实体嵌入（emb，形状为 [batch_size, hidden_dim]）、所有实体嵌入（all_emb，形状为 [num_entities, hidden_dim]）、源实体的索引（source_idx）、关系 ID（relation_ids）和模式（mode）。
    # 模型在这个函数中计算目标实体的嵌入得分，并将其返回。
        # 具体来说，该函数首先根据模式选择正确的变量来使用关系嵌入。
        # 然后，它基于模型的类型（如是对称的还是非对称的）使用关系嵌入更新实体嵌入。
        # 最后，它使用聚合器来计算目标实体的嵌入得分，并应用一个 MLP（多层感知机）进行可选的后处理
    def forward(self, emb, all_emb, source_idx, relation_ids, mode): # bs * dim, nentity * dim
        # 它表示 emb 张量的 batch size，也就是这个张量中有多少个样本。emb 张量的形状为 (bs, dim)，其中 dim 表示每个样本的特征维度，因此 bs 就是这个张量的第一个维度的大小。
        bs = emb.size(0)
        if mode == 'head-batch':
            node_edge = self.node_edge2
            edge_node = self.edge_node2
            r_mask = self.r_mask2
        else:
            node_edge = self.node_edge1
            edge_node = self.edge_node1
            r_mask = self.r_mask1

        # 关系感知嵌入层的作用是将实体和关系嵌入向量连接起来，得到一个新的实体向量表示。
        if self.relation_aware == 'diag':
            if mode == 'head-batch':
                rel = self.head_r[relation_ids]
            else:
                rel = self.tail_r[relation_ids]
            # 对于当前实体的嵌入emb，将其与当前实体对应的关系向量逐元素相乘，然后增加一维，得到嵌入的结果。
            # 对于所有实体的嵌入all_emb，先在第一维增加一维，然后与当前实体对应的关系向量逐元素相乘，得到嵌入的结果。
            emb = (emb * rel).unsqueeze(0) # 1 * bs * dim
            all_emb = all_emb.unsqueeze(1) * rel.unsqueeze(0) # (n * 1 * dim) * (1 * dim) = n * 1 * dim
        elif self.relation_aware == 'matrix':
            if mode == 'head-batch':
                rel = self.head_r[relation_ids].view(-1, self.config.hidden_dim, self.config.hidden_dim)
            else:
                rel = self.tail_r[relation_ids].view(-1, self.config.hidden_dim, self.config.hidden_dim)
            # 将当前实体的嵌入与关系矩阵进行矩阵乘法，得到嵌入的结果。
            # 对于所有实体的嵌入，先在第二维增加一维，然后将其复制 bs 次，即重复 bs 次，最后再与对应的关系矩阵进行矩阵乘法，得到嵌入的结果。
            # 这里的 tensor.repeat 函数，每一个参数代表在哪一维进行重复（如维度数增加，则先增加维度再进行重复）
            emb = torch.bmm(emb.unsqueeze(1), rel)# (bs,1,dim)*(1,dim,dim)=(bs,1,dim)*(n,dim,dim) = bs * 1 * dim
            all_emb = torch.bmm(all_emb.repeat(bs, 1, 1), rel)# (bs,n,dim)*(1,dim,dim)=(bs,n,dim)*(n,dim,dim) = bs * n * dim
        elif self.relation_aware is None:
            pass
            # emb = emb.unsqueeze(0)
            # all_emb = all_emb.unsqueeze(1)


        # 如果self.config.cosine为True，那么就采用余弦相似度计算距离。
        if self.config.cosine:
            # 如果是'diag'，则先将所有实体向量和当前实体向量进行矩阵乘法，再通过激活函数进行激活
            if self.relation_aware == 'diag':
                dis = ACTFN[self.config.activation](torch.bmm(all_emb.permute(1, 0, 2).contiguous(), emb.permute(1, 2, 0).contiguous()).squeeze(2).t()) # (1,n,dim)*(bs,dim,1)=(bs,n,dim)*(bs,dim,1)=(bs,n,1)    .squeeze(2)=(bs,n)        .t()=(n,bs)
            elif self.relation_aware is None:
                dis = ACTFN[self.config.activation](torch.matmul(emb, all_emb.t())).t()# n * bs
            elif self.relation_aware == 'matrix':
                # if self.args.normalize:
                # ensure constraints
                # 在计算余弦相似度时，需要将向量进行标准化处理。
                # 先对所有实体向量和当前实体向量进行规范化（通过clamp_norm函数），再进行矩阵乘法，最后通过激活函数进行激活
                emb = clamp_norm(emb, p=2.0, dim=-1, maxnorm=1.0)
                all_emb = clamp_norm(all_emb, p=2.0, dim=-1, maxnorm=1.0)
                dis = ACTFN[self.config.activation](torch.matmul(all_emb, emb.permute(0, 2, 1).contiguous()).squeeze(2).t())
                # (bs,n,dim)*(bs,dim,1)=(bs,n,1)    .squeeze(2)=(bs,n)        .t()=(n,bs)
        # 如果self.config.cosine为False，则直接计算两个向量之间的欧氏距离，并通过sigmoid函数进行映射，得到一个(0,1)之间的相似度值。
        else:
            dis = (emb - all_emb).norm(p=1, dim=-1)
            
            # 如果source_idx不为空，则将该向量对应位置的距离设置为一个非常大的值，从而在后续处理中排除该向量。最后通过self.gamma对距离进行缩放。
            if source_idx is not None:
                self_mask = torch.ones(self.n, bs).bool().cuda()
                self_mask[source_idx, torch.arange(bs)] = False
                dis = torch.where(self_mask,dis,torch.tensor(1e8).cuda())
            dis = torch.sigmoid(self.gamma-dis)# n * bs

        # edge_score 表示边的得分，即两个实体之间的相似度/距离。为了计算 edge_score，首先将正样本实体向量与全部实体向量计算得到相似度/距离矩阵 dis，然后将该矩阵传递给图卷积层的 propagate 函数，使用 self.aggregator1 对矩阵进行聚合，得到每个实体与其它所有实体之间的相似度/距离。
        # 使用 self.aggregator1 对 dis 向量进行聚合，将得到的结果存储在 edge_score 中。其中，node_edge 是一个二元组的元素，代表着源节点和目标节点之间的边。
        edge_score = self.aggregator1.propagate(node_edge,x=dis, size = (self.n,self.m)) #m * bs
        # 接着，将 relation_ids 中的边的关系类型与 r_mask 中对应的行进行匹配，得到一个关系类型对应的掩码矩阵，然后将该矩阵乘到 edge_score 中，得到新的 edge_score。
        # 接下来，将 edge_score 和 relation_ids (bs x 1) 中的边的关系类型进行匹配，具体来说，使用 torch.index_select() 函数根据 relation_ids 的内容选取 r_mask 中对应位置的行，并使用 permute() 函数将结果进行转置，最后将结果乘到 edge_score 中，得到新的 edge_score 。
        # 这一步的目的是将 edge_score 中没有使用到的关系类型对应的行全部置零，从而排除不相关的关系对实体相似度/距离的影响。
        edge_score *= torch.index_select(r_mask, dim=0, index=relation_ids).permute(1,0) #m*bs
        # 最后，将得到的 edge_score 传递给 self.aggregator2 进行聚合，得到每个实体与其它所有实体之间的相似度/距离得分 (n x batch_size)，并将结果转置，使得结果的维度变为 batch_size x n ，其中 n 是实体的数量，batch_size 是批处理的大小。
        # 最后，使用self.aggregator2对得到的edge_score向量进行聚合，将结果存储在target_score中，并将结果转置，使得结果的维度变为bs*n。其中，edge_node是一个二元组的元素，代表着边的起始节点和结束节点。
        target_score = self.aggregator2.propagate(edge_node,x = edge_score, size = (self.m,self.n) ).permute(1,0) #bs * n

        # 将最后的输出通过一个MLP的作用是将输出进行非线性变换，增加模型的表达能力，并且将输出映射到一个固定的范围内，以便进行后续的评估和比较。
        if self.use_mlp:
            target_score = self.mlp(target_score)
        return target_score

class KGEModel(nn.Module):
    def __init__(self, args, model_name, nentity, nrelation, hidden_dim, gamma, mlp=False, relation_aware=False,
                 double_entity_embedding=False, double_relation_embedding=False, train_triples = None, ible_weight = 0.0):
        super(KGEModel, self).__init__()
        self.config = args
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.ible_weight = ible_weight
        if ible_weight!= 0.0:
            self.ible = IBLE(args, train_triples, nrelation, nentity, gamma, epsilon=2.0, mlp=mlp, relation_aware=relation_aware, entity_dim=hidden_dim)

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]), 
            requires_grad=False
        )
        
        # 这行代码创建了一个模型参数 self.embedding_range，用来表示实体和关系嵌入的范围。这个范围是由 gamma（一个先前设定的参数）和 epsilon（设定为 2.0）计算而来。具体地，这个范围是 [gamma + epsilon] / hidden_dim，其中 hidden_dim 是嵌入的维度。它是一个只读的参数，因为它是由 gamma 和 epsilon 决定的。
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]), 
            requires_grad=False
        )
        
        self.entity_dim = hidden_dim*2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim*2 if double_relation_embedding else hidden_dim
        
        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        
        # 这部分代码是用来初始化实体嵌入矩阵（entity_embedding）的。
        # 初始化实体嵌入矩阵通常采用随机均匀分布，这里使用的是torch.nn.init.uniform_方法，它会将self.entity_embedding中的每个元素初始化为一个在[-bound, bound]范围内的均匀分布随机数。
        # 它乘以1.0是为了将结果转换为浮点数，然后乘以6是为了设置边界的最大值和最小值,其中self.entity_embedding.shape[-1]是实体嵌入的最后一个维度（即每个实体嵌入向量的维度大小），这里用来计算分母。
        bound = 1.0 * 6 / math.sqrt(self.entity_embedding.shape[-1])
        torch.nn.init.uniform_(self.entity_embedding, -bound, bound)

        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))

        bound = 1.0 * 6 / math.sqrt(self.relation_embedding.shape[-1])
        torch.nn.init.uniform_(self.relation_embedding, -bound, bound)
        
        # 如果当前使用的模型是pRotatE，就会初始化一个可训练的变量self.modulus，其值为0.5 * self.embedding_range.item()，这个值在模型的前向传播过程中会被使用；
        if model_name == 'pRotatE':
            self.modulus = nn.Parameter(torch.Tensor([[0.5 * self.embedding_range.item()]]))
        
        #Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'pRotatE']:
            raise ValueError('model %s not supported' % model_name)
        
        # 如果当前使用的模型是RotatE/ComplEx，并且没有开启double_entity_embedding选项或者开启了double_relation_embedding选项，则会抛出一个异常    
        if model_name == 'RotatE' and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('RotatE should use --double_entity_embedding')

        if model_name == 'ComplEx' and (not double_entity_embedding or not double_relation_embedding):
            raise ValueError('ComplEx should use --double_entity_embedding and --double_relation_embedding')
        
    def forward(self, sample, mode='single'):
        # 计算一批三元组得分的正向函数。
        # 在“单一”模式下，样本是一批三元组。
        # 在“head-batch”或“tail-batch”模式下，样本由两部分组成。
        # 第一部分通常是正样本。
        # 第二部分是负样本中的实体。
        # 因为负样本和正样本通常在它们的三元组（（head，relation）或（relation，tail））中共享两个元素。
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements 
        in their triple ((head, relation) or (relation, tail)).
        '''

        if mode == 'single':
            # 如果采样模式为'single'，则表示每个样本只有一个头实体和一个尾实体，因此只需要将这个样本对应的头实体、关系和尾实体的向量从对应的embedding矩阵中取出即可。
            batch_size, negative_sample_size = sample.size(0), 1
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=sample[:,0]
            ).unsqueeze(1)
            
            relation = torch.index_select(
                self.relation_embedding, 
                dim=0, 
                index=sample[:,1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=sample[:,2]
            ).unsqueeze(1)
            
        elif mode == 'head-batch':
            # 如果采样模式为'head-batch'，则表示每个样本只有一个尾实体，但会对同一个头实体进行负采样得到多个负样本。
            # 因此需要将每个样本对应的尾实体、关系和头实体的向量从对应的embedding矩阵中取出。
            # 其中头实体向量会通过把所有负样本张量 head_part 展平成一维来一次性取出 embedding，所以需要根据负采样数目进行reshape。
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            
            relation = torch.index_select(
                self.relation_embedding, 
                dim=0, 
                index=tail_part[:, 1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=tail_part[:, 2]
            ).unsqueeze(1)
            
        elif mode == 'tail-batch':
            # 如果采样模式为'tail-batch'，则表示每个样本只有一个头实体，但会对同一个尾实体进行负采样得到多个负样本。
            # 因此需要将每个样本对应的头实体、关系和尾实体的向量从对应的 embedding 矩阵中取出。
            # 其中尾实体向量会通过把所有负样本张量 tail_part 展平成一维来一次性取出 embedding，所以需要根据负采样数目进行reshape。
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=head_part[:, 0]
            ).unsqueeze(1)
            
            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
     
        else:
            raise ValueError('mode %s not supported' % mode)
        
        # score的计算方式取决于所选择的模型名称，例如TransE、DistMult、ComplEx、RotatE或pRotatE等，可以通过model_name参数进行选择。    
        model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'pRotatE': self.pRotatE
        }
        
        if self.model_name in model_func:
            score = model_func[self.model_name](head, relation, tail, mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)
        
        return score
    
    def TransE(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def DistMult(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim = 2)
        return score

    def ComplEx(self, head, relation, tail, mode):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim = 2)
        return score

    def RotatE(self, head, relation, tail, mode):
        pi = 3.14159265358979323846
        
        # 它首先将头实体和尾实体向量拆分为实部和虚部，并将关系向量的幅值限制在 [-pi, pi] 之间。
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        #Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation/(self.embedding_range.item()/pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        # 头实体得分表示为 (relation x tail) 与 (head) 在复平面上的差值，尾实体得分表示为 (head x relation) 与 (tail) 在复平面上的差值
        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail
        
        # 将头实体得分和尾实体得分的范数相加，并用 gamma 参数减去得分的总和，即为实体对之间的得分。
        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0)

        score = self.gamma.item() - score.sum(dim = 2)
        return score

    def pRotatE(self, head, relation, tail, mode):
        pi = 3.14159262358979323846
        
        #Make phases of entities and relations uniformly distributed in [-pi, pi]

        phase_head = head/(self.embedding_range.item()/pi)
        phase_relation = relation/(self.embedding_range.item()/pi)
        phase_tail = tail/(self.embedding_range.item()/pi)

        if mode == 'head-batch':
            score = phase_head + (phase_relation - phase_tail)
        else:
            score = (phase_head + phase_relation) - phase_tail

        score = torch.sin(score)            
        score = torch.abs(score)

        score = self.gamma.item() - score.sum(dim = 2) * self.modulus
        return score
    
    # 一个训练模型的函数train_step
    # 接收一个模型、一个优化器、一个数据迭代器和一些参数作为输入，并在模型上应用反向传播算法来更新模型参数并返回损失值。
    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        '''
        A single train step. Apply back-propation and return the loss 一个单一的训练 step, 应用反向传播并返回损失值
        '''

        # 设置为训练模式
        model.train()

        # 将模型的所有参数的梯度设置为零，以便进行下一次前向传播和反向传播的操作。
        optimizer.zero_grad()

        # args.gradient_accumulation_steps 表示梯度累积的步数，即在每次更新参数之前累积多少个 batch 的梯度。
        # 当训练时需要处理比较大的 mini-batch 但 GPU 的显存不够时，可以使用梯度累积来解决这个问题。
        # 通过梯度累积，可以在不增加显存消耗的情况下，使 mini-batch 的大小变得更大，从而加速训练。
        for _ in range(args.gradient_accumulation_steps):
            positive_sample, negative_sample, subsampling_weight, mode, label = next(train_iterator)   # this label is deprecated 此标签已弃用

            if args.cuda:
                positive_sample = positive_sample.cuda()
                negative_sample = negative_sample.cuda()
                subsampling_weight = subsampling_weight.cuda()

            # 如果模型配置中的negative_sample_size为-1，表示采用全采样方式，即使用所有负样本进行训练。
            if model.config.negative_sample_size == -1:
                
                # 该参数的值在0和1之间，则需要将模型的输出与IBLE损失函数的输出进行合并得到最终的负样本得分。
                if 0.0 < model.ible_weight < 1.0:
                    negative_score = model((positive_sample, negative_sample), mode=mode)
                    source_idx = positive_sample[:,2] if mode == 'head-batch' else positive_sample[:,0]

                    # 根据source_idx提取对应的嵌入向量。model.entity_embedding是一个二维张量，其中每行都是一个实体的嵌入向量。source_idx是一个一维张量，其中包含正样本中需要提取嵌入向量的实体的索引。
                    emb_batch = torch.index_select(model.entity_embedding,dim=0,index=source_idx)
                    # 由于需要在所有实体的嵌入向量之间计算IBLE分数，因此可以直接将model.entity_embedding复制一份得到一个形状为(num_entities, embedding_dim)的张量emb_all。
                    emb_all = model.entity_embedding

                    ible_score = model.ible(emb_batch,emb_all,source_idx,positive_sample[:,1],mode)
                    negative_score = model.merge_score(negative_score,ible_score)
                # 如果ible_weight为0，则直接使用模型的输出作为负样本得分
                elif model.ible_weight == 0.0:
                    negative_score = model((positive_sample, negative_sample), mode=mode)
                # 如果ible_weight为1，则使用IBLE损失函数的输出作为负样本得分。
                elif model.ible_weight == 1.0:
                    source_idx = positive_sample[:,2] if mode == 'head-batch' else positive_sample[:,0]

                    emb_batch = torch.index_select(model.entity_embedding,dim=0,index=source_idx)
                    emb_all = model.entity_embedding

                    negative_score = model.ible(emb_batch,emb_all,source_idx,positive_sample[:,1],mode)

                labels = positive_sample[:,0] if mode == 'head-batch' else positive_sample[:,2]
                loss = F.cross_entropy(negative_score, labels)
            else:
                if 0.0 < model.ible_weight < 1.0:
                    negative_score = model((positive_sample, negative_sample), mode=mode)
                    positive_score = model(positive_sample)

                    source_idx = positive_sample[:,2] if mode == 'head-batch' else positive_sample[:,0]

                    emb_batch = torch.index_select(model.entity_embedding,dim=0,index=source_idx)
                    emb_all = model.entity_embedding
                    ible_score = model.ible(emb_batch, emb_all, source_idx, positive_sample[:,1], mode)
                    
                    if mode == 'head-batch':
                        positive_ible_score = ible_score[torch.arange(positive_sample.shape[0]), positive_sample[:,0]]
                    else:
                        positive_ible_score = ible_score[torch.arange(positive_sample.shape[0]), positive_sample[:,2]]
                    positive_score = model.merge_score(positive_score, positive_ible_score.unsqueeze(1))
                    
                    negative_ible_score = torch.vstack([ible_score[i, negative_sample[i]] for i in range(negative_sample.shape[0])])
                    negative_score = model.merge_score(negative_score, negative_ible_score)
                elif model.ible_weight == 0.0:
                    negative_score = model((positive_sample, negative_sample), mode=mode)
                    positive_score = model(positive_sample)
                elif model.ible_weight == 1.0:
                    source_idx = positive_sample[:,2] if mode == 'head-batch' else positive_sample[:,0]

                    emb_batch = torch.index_select(model.entity_embedding,dim=0,index=source_idx)
                    emb_all = model.entity_embedding
                    ible_score = model.ible(emb_batch, emb_all, source_idx, positive_sample[:,1], mode)
                    
                    if mode == 'head-batch':
                        positive_score = ible_score[torch.arange(positive_sample.shape[0]), positive_sample[:,0]]
                    else:
                        positive_score = ible_score[torch.arange(positive_sample.shape[0]), positive_sample[:,2]]
                    negative_score = torch.vstack([ible_score[i, negative_sample[i]] for i in range(negative_sample.shape[0])])

                if model.config.loss == 'crossentropy':
                    # 正样本得分和负样本得分拼接在一起，作为模型对当前批次样本的预测得分。拼接的方式是将正样本得分和负样本得分按列拼接（即在列方向上拼接），得到一个形状为(batch_size, 2)的张量。其中第一列为正样本得分，第二列为负样本得分。
                    scores = torch.cat([positive_score, negative_score], dim=1)
                    # 将标签设为全 0 的张量，表示所有的例子都是负例。
                    # 这里，device 会被传递给 labels tensor，以确保 labels 和 scores 存储在相同的设备上，从而避免因设备不匹配而导致的运行错误。
                    labels = torch.zeros(scores.size(0), device=scores.device).long()
                    # 我们将 labels 设为全 0，是因为我们要让神经网络去预测一个二分类任务，其中正例和负例分别对应的标签为 1 和 0。因此，我们将标签设为全 0，表示所有样本都是负例，让神经网络去预测这些负例的概率，并计算损失。
                    loss = F.cross_entropy(scores, target=labels)

                elif model.config.loss == 'margin':
                    if args.negative_adversarial_sampling:
                        #In self-adversarial sampling, we do not apply back-propagation on the sampling weight
                        # 在我对抗采样中，我们不对采样权重应用反向传播
                        negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim = 1).detach() 
                                        * F.logsigmoid(-negative_score)).sum(dim = 1)
                    else:
                        negative_score = F.logsigmoid(-negative_score).mean(dim = 1)

                    positive_score = F.logsigmoid(positive_score).squeeze(dim = 1)

                    if args.uni_weight:
                        positive_sample_loss = - positive_score.mean()
                        negative_sample_loss = - negative_score.mean()
                    else:
                        positive_sample_loss = - (subsampling_weight * positive_score).sum()/subsampling_weight.sum()
                        negative_sample_loss = - (subsampling_weight * negative_score).sum()/subsampling_weight.sum()

                    loss = (positive_sample_loss + negative_sample_loss) / 2


                    # positive_sample_loss = - positive_score.mean()
                    # negative_sample_loss = negative_score.mean()

                    # loss = (positive_sample_loss + negative_sample_loss) / 2

            #loss = loss_fn(negative_score, labels)

            # 。如果模型参数太多，模型可能会过度拟合（overfitting），这时可以通过正则化来约束模型参数的大小，以此提高模型的泛化性能（generalization performance）。
            if args.regularization != 0.0:
                #Use L3 regularization for ComplEx and DistMult 对 ComplEx 和 DistMult 使用 L3 正则化
                # 实体嵌入和关系嵌入的 L3 范数（L3 norm）分别做了三次方，并相加得到正则化项（regularization term），乘以 args.regularization 得到最终的正则化惩罚项（regularization penalty）
                regularization = args.regularization * (
                    model.entity_embedding.norm(p = 3)**3 + 
                    model.relation_embedding.norm(p = 3).norm(p = 3)**3
                )
                # 将正则化惩罚项加到原来的损失（loss）上，得到最终的训练损失（training loss）。
                loss = loss + regularization
                regularization_log = {'regularization': regularization.item()}
            else:
                regularization_log = {}
            
            # 最后，通过将 loss 除以 args.gradient_accumulation_steps（梯度累积步数）来平均损失。然后对平均损失进行反向传播（backpropagation）和参数更新。    
            (loss / args.gradient_accumulation_steps).backward()

        # 执行一步梯度下降优化算法，根据计算出的梯度来更新模型的参数
        optimizer.step()

        log = {
            **regularization_log,
            # 'positive_sample_loss': positive_sample_loss.item(),
            # 'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }

        return log

    def merge_score(self, origin_score, ible_score):
        return ible_score * self.ible_weight + torch.sigmoid(origin_score) * (1 - self.ible_weight)

    @staticmethod
    def test_step(model, test_triples, all_true_triples, args):
        '''
        Evaluate the model on test or valid datasets
        '''
        
        # 进入评估模式，不启用 BatchNormalization 和 Dropout
        model.eval()
        
        if args.countries:
            #Countries S* datasets are evaluated on AUC-PR
            #Process test data for AUC-PR evaluation
            sample = list()
            y_true  = list()
            for head, relation, tail in test_triples:
                for candidate_region in args.regions:
                    y_true.append(1 if candidate_region == tail else 0)
                    sample.append((head, relation, candidate_region))

            sample = torch.LongTensor(sample)
            if args.cuda:
                sample = sample.cuda()

            with torch.no_grad():
                y_score = model(sample).squeeze(1).cpu().numpy()

            y_true = np.array(y_true)

            #average_precision_score is the same as auc_pr
            auc_pr = average_precision_score(y_true, y_score)

            metrics = {'auc_pr': auc_pr}
            
        else:
            #Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics 使用标准的5个评估指标
            #Prepare dataloader for evaluation 准备数据加载器进行评估
            test_dataloader_head = DataLoader(
                TestDataset(
                    test_triples, 
                    all_true_triples, 
                    args.nentity, 
                    args.nrelation, 
                    'head-batch'
                ), 
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num//2), 
                collate_fn=TestDataset.collate_fn
            )

            test_dataloader_tail = DataLoader(
                TestDataset(
                    test_triples, 
                    all_true_triples, 
                    args.nentity, 
                    args.nrelation, 
                    'tail-batch'
                ), 
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num//2), 
                collate_fn=TestDataset.collate_fn
            )
            
            test_dataset_list = [test_dataloader_head, test_dataloader_tail]
            
            # 初始化了一个空列表 logs，用于保存模型在测试集上的表现。
            logs = []

            step = 0
            # 计算了变量 total_steps，该变量表示对于两种类型的测试集，模型总共需要进行的步骤数。这里采用了列表推导式和 sum 函数来计算。
            total_steps = sum([len(dataset) for dataset in test_dataset_list])
            #  初始化了两个空字典 ranks["head-batch"] 和 ranks["tail-batch"]，用于保存不同类型的测试集（头部实体和尾部实体）中每个实体的排名信息。
            ranks = {"head-batch": {}, "tail-batch": {}}
            # 使用 torch.no_grad() 上下文管理器来确保计算模型分数时不会计算梯度。
            with torch.no_grad():
                for test_dataset in test_dataset_list:
                    for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                        if args.cuda:
                            positive_sample = positive_sample.cuda()
                            negative_sample = negative_sample.cuda()
                            filter_bias = filter_bias.cuda()

                        # positive_sample 是一个 tensor ，其大小为 (batch_size, 3) ，其中第一维的大小为 batch size 。因此，positive_sample.size(0) 就是 batch size 的值。
                        batch_size = positive_sample.size(0)

                        # 这行代码是用当前的模型计算一个 batch 的正负样本对的分数，其中输入是正样本和负样本的下标，输出是对应的得分。
                        score = model((positive_sample, negative_sample), mode)
                        # 如果使用了 IBLE ，则还计算 IBLE 分数，并将其与模型分数合并。
                        if model.ible_weight != 0.0:
                            if mode == 'head-batch':
                                source_idx = positive_sample[:, 2]
                            else:
                                source_idx = positive_sample[:, 0]
                            emb_batch = torch.index_select(model.entity_embedding, dim=0, index=source_idx)
                            emb_all = model.entity_embedding
                            ible_score = model.ible(emb_batch, emb_all, source_idx, positive_sample[:,1], mode)
                            score = model.merge_score(score,ible_score)

                        score += filter_bias * 10000

                        #Explicitly sort all the entities to ensure that there is no test exposure bias 明确地对所有实体进行排序，以确保没有测试暴露偏差
                        # 使用 PyTorch 的 argsort 函数对 score 进行排序，以获取得分最高的实体/关系。argsort 函数返回排序后元素的索引，而不是排序后的值。dim=1 表示按行排序，即对每个三元组的得分进行排序。descending=True 表示按照从高到低的顺序进行排序。
                        argsort = torch.argsort(score, dim = 1, descending=True)

                        # 在知识图谱中，一个 triple 通常由三个实体组成：head，relation 和 tail。当进行 link prediction 任务时，需要在给定的 triple 中缺失一个实体，然后通过模型预测该缺失的实体是什么。如果缺失的是 head，那么 positive_arg 就是该 triple 中的 head；如果缺失的是 tail，那么 positive_arg 就是该 triple 中的 tail。
                        if mode == 'head-batch':
                            positive_arg = positive_sample[:, 0]
                        elif mode == 'tail-batch':
                            positive_arg = positive_sample[:, 2]
                        else:
                            raise ValueError('mode %s not supported' % mode)

                        for i in range(batch_size):
                            #Notice that argsort is not ranking  注意 argsort 不是排名
                            # 这里的 (argsort[i, :] == positive_arg[i]) 返回一个布尔值数组，表示每个得分是否等于正例的值。然后使用 nonzero() 函数来查找所有等于正例值的得分的索引。由于这是一个对数值进行相等比较的操作，因此可能存在精度差异。因此，这里的排名实际上不是严格的排名，但是对于评估指标的计算来说是足够的。
                            ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                            assert ranking.size(0) == 1

                            #ranking + 1 is the true ranking used in evaluation metrics ranking + 1 是评估指标中使用的真实排名
                            ranking = 1 + ranking.item()
                            ranks[mode][str(positive_sample[i].tolist())] = ranking
                            logs.append({
                                'MRR': 1.0/ranking,
                                'MR': float(ranking),
                                'HITS@1': 1.0 if ranking <= 1 else 0.0,
                                'HITS@3': 1.0 if ranking <= 3 else 0.0,
                                'HITS@10': 1.0 if ranking <= 10 else 0.0,
                            })

                        if step % args.test_log_steps == 0:
                            logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                        step += 1
                    
                    # sort
                    # ranks[mode] = sorted(ranks[mode].items(), key=lambda x: x[0])

            # 对于 logs 中的每个 log ，它从中提取指标值并将其加到对应指标名称的总和中。最后，对于每个指标，它将总和除以 log 的数量，以得到该指标的平均值。
            metrics = {}
            for metric in logs[0].keys():
                metrics[metric] = sum([log[metric] for log in logs])/len(logs)
            
            import json
            # with open('ranks.json', 'w') as f:
            #     f.write(json.dumps(ranks))

        return metrics
