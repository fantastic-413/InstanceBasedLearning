#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch

from torch.utils.data import Dataset

# 这是一个PyTorch的数据集类，用于处理知识图谱中的三元组数据，以用于训练知识图谱嵌入模型。这个类将原始三元组数据预处理，以获取正例和负例样本，并提供了一些实用的函数用于对数据进行过滤和子采样。
# 这个数据集类将三元组数据转换为张量，并提供了一个collate_fn函数，以将数据样本收集到一个批次中，以用于PyTorch的数据加载器。
class TrainDataset(Dataset):
    # triples：一个包含训练数据三元组的列表。
    # negative_sample_size：每个训练样本所需的负样本数
    def __init__(self, triples, nentity, nrelation, negative_sample_size, mode):
        self.triples = triples
        # 首先将 triples 转换为一个集合，并将其存储在 self.triple_set 中
        self.triple_set = set(triples)
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        # 计算出每个三元组在 triples 中出现的次数，并将结果存储在 self.count 中
        self.count = self.count_frequency(triples)
        # 获取每个关系对应的头部和尾部实体，并将结果分别存储在self.true_head和self.true_tail中。
        self.true_head, self.true_tail = self.get_true_head_and_tail(self.triples)
        
        # 根据mode的不同，它将构造不同的输入列表。
            # 如果mode为'head-batch'，则输入列表是所有三元组的关系和尾部实体的组合；
            # 如果mode为'tail-batch'，则输入列表是所有三元组的头部和关系的组合。
            # 这些输入将在训练模型时使用。
        if self.mode == 'head-batch':
            self.inputs = list(set(torch.tensor(self.triples)[:, 1:]))
        elif self.mode == 'tail-batch':
            self.inputs = list(set(torch.tensor(self.triples)[:, :2]))

        
    def __len__(self):
        return len(self.inputs)
    
    # 该方法返回一个元组，包含正样本（实际存在的三元组）、负样本（通过负采样生成的三元组）、二次采样权重、训练批次的模式以及标签。
    def __getitem__(self, idx):
        positive_sample = self.triples[idx]
        head, relation, tail = positive_sample

        # 计算采样概率，根据每个三元组的出现次数，将其归一化后取倒数，再开根号。
        subsampling_weight = self.count[(head, relation)] + self.count[(tail, -relation-1)]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))
        
        negative_sample_list = []
        negative_sample_size = 0

        # negative_sample_size 是 -1，则采样全部实体，否则需要不断采样，直到采样到足够的数量。
        if self.negative_sample_size == -1:
            negative_sample = list(range(self.nentity))
        else:
            while negative_sample_size < self.negative_sample_size:
                # 生成负样本，使用 np.random.randint 函数生成一个由大小为 self.negative_sample_size*2 的随机整数数组，随机数大小在 [0,self.nentity) 之间
                negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size*2)
                # 根据 mode 的不同，分别选择负样本的头实体或者尾实体。在每次生成负样本时，随机生成两倍于 negative_sample_size 的实体编号，再根据已知三元组中的关系和头尾实体，过滤掉那些不合法的实体编号，最终得到合法的负样本。
                    # 如果 mode 是 'head-batch'，则选择头实体作为三元组中的变量，并过滤掉与关系和尾实体相符的实体编号，保留剩余的实体编号作为负样本。
                    # 如果 mode 是 'tail-batch'，则选择尾实体作为三元组中的变量，并过滤掉与关系和头实体相符的实体编号，保留剩余的实体编号作为负样本。
                if self.mode == 'head-batch':
                    # numpy 库中的 in1d 函数，该函数的作用是判断一个数组中的元素是否属于另一个数组，返回一个布尔型的数组，指示第一个数组中的每个元素是否在第二个数组中出现。
                    mask = np.in1d(
                        negative_sample,
                        self.true_head[(relation, tail)],
                        # assume_unique 参数表示假设输入数组已经是唯一的（即没有重复元素），这可以加速算法。
                        assume_unique=True,
                        # 而 invert 参数则表示要对布尔数组进行逻辑反转。
                        invert=True
                    )
                elif self.mode == 'tail-batch':
                    mask = np.in1d(
                        negative_sample,
                        self.true_tail[(head, relation)],
                        assume_unique=True,
                        invert=True
                    )
                else:
                    raise ValueError('Training batch mode %s not supported' % self.mode)
                # 将这些负样本添加到 negative_sample_list 中，直到所需数量的负样本数目达到设定值。
                negative_sample = negative_sample[mask]
                negative_sample_list.append(negative_sample)
                negative_sample_size += negative_sample.size

            # 将 negative_sample_list 列表中的所有负样本拼接起来，然后从拼接后的样本中随机选取 self.negative_sample_size 个负样本返回。
                # 由于 negative_sample_list 中的每个元素都是一个包含多个负样本的 numpy 数组，所以在拼接时使用了 numpy 的 concatenate 函数。
            negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]

        negative_sample = torch.LongTensor(negative_sample)

        positive_sample = torch.LongTensor([head, relation, tail])

        # 标签是一个张量，其形状为（nentity，），其中1.0表示正样本的位置，0.0表示负样本和未知样本的位置。
        if self.mode == 'head-batch':
            labels = torch.zeros(self.nentity)
            labels[self.true_head[(relation, tail)]] = 1.0
        elif self.mode == 'tail-batch':
            labels = torch.zeros(self.nentity)
            labels[self.true_tail[(head, relation)]] = 1.0
        else:
            raise ValueError('Training batch mode %s not supported' % self.mode)
        return positive_sample, negative_sample, subsampling_weight, self.mode, labels
    
    # 这是一个静态方法，用于将一个batch的数据拼接成一个四元组，以便进行模型训练。
    # 具体来说，这个方法接收一个 batch 的数据列表 data ，该列表中每个元素都是由 TrainDataset 中的 __getitem__ 方法返回的五元组：positive_sample、negative_sample、subsample_weight、mode 和 labels。
    # 这个函数将每个 batch 中所有三元组的正样本、负样本、subsampling权重、模式（head-batch或tail-batch）和标签（1表示真实样本，0表示负样本）分别提取出来，然后用 torch.stack() 函数将它们按照第0维进行拼接。最后将它们以 tuple 的形式返回。
        # 对positive_sample和negative_sample这两个三元组张量列表，分别在第0维上进行拼接，
        # positive_sample 和 negative_sample 分别存储正样本三元组和负样本实体，形状为 (batch_size, 3)和(batch_size, negative_sample_size)。
        # subsample_weight 是一个形状为 (batch_size,) 的张量，用来存储每个三元组的二次采样权重。
        # mode 是一个字符串，表示这个 batch 是以头实体为基准还是以尾实体为基准。
        # labels 是一个形状为 (batch_size, nentity) 的张量，用来存储每个实体是正样本还是负样本的标签。如果一个实体是正样本，则相应位置的值为 1.0，否则为 0.0。
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        labels = torch.stack([_[4] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, subsample_weight, mode, labels
    
    @staticmethod
    def count_frequency(triples, start=4):
        '''
        Get frequency of a partial triple like (head, relation) or (relation, tail)  获取部分三元组的频率，如 (head, relation) 或 (relation, tail)
        The frequency will be used for subsampling like word2vec  频率将用于二次采样，如 word2vec
        '''
        # 由于一个三元组可以有两种方向的表示，例如（Obama, President_of, USA）和（USA, President_of_Reverse, Obama），它们分别代表了从头实体到尾实体和从尾实体到头实体的方向。为了考虑两种方向的表示，我们将头实体和关系对应的出现次数加上尾实体和反向关系对应的出现次数，这样就将两种方向的表示都考虑到了。
        count = {}
        for head, relation, tail in triples:
            if (head, relation) not in count:
                count[(head, relation)] = start
            else:
                count[(head, relation)] += 1

            if (tail, -relation-1) not in count:
                count[(tail, -relation-1)] = start
            else:
                count[(tail, -relation-1)] += 1
        return count
    
    @staticmethod
    def get_true_head_and_tail(triples):
        '''
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling 构建真实三元组的字典，用于过滤这些真实三元组以进行负采样
        '''
        
        true_head = {}
        true_tail = {}

        # 函数先遍历原始数据集中的所有三元组，把符合条件的头实体和尾实体加入到对应的字典中。
        for head, relation, tail in triples:
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = []
            true_tail[(head, relation)].append(tail)
            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = []
            true_head[(relation, tail)].append(head)

        # 对于字典 true_head 中的每个键值对，去重后转换为 numpy 数组；对于字典 true_tail 中的每个键值对，也去重后转换为 numpy 数组。
        for relation, tail in true_head:
            true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
        for head, relation in true_tail:
            true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))                 

        return true_head, true_tail

# 这是一个用于构建PyTorch中数据集的类TestDataset。在这个数据集中，每个样本由一个三元组（头实体，关系，尾实体）以及负样本、过滤偏差和模式组成。    
class TestDataset(Dataset):
    def __init__(self, triples, all_true_triples, nentity, nrelation, mode):
        self.len = len(triples)
        self.triple_set = set(all_true_triples)
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.mode = mode

    def __len__(self):
        return self.len
    
    # 该函数返回给定索引idx的样本。
    # 对于给定三元组（头实体，关系，尾实体），如果mode是'head-batch'，则为了创建负样本，随机选择一个头实体替换原始三元组中的头实体，如果替换后的三元组在所有真实三元组的集合中，则返回-1；如果替换后的三元组不在所有真实三元组的集合中，则返回0。如果mode是'tail-batch'，则执行类似的操作，但是替换尾实体。正样本是原始三元组。
    # 这个函数最后返回一个元组，包含正样本、负样本、过滤偏差和模式。
    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]

        if self.mode == 'head-batch':
            tmp = [(0, rand_head) if (rand_head, relation, tail) not in self.triple_set
                   else (-1, head) for rand_head in range(self.nentity)]
            tmp[head] = (0, head)
        elif self.mode == 'tail-batch':
            tmp = [(0, rand_tail) if (head, relation, rand_tail) not in self.triple_set
                   else (-1, tail) for rand_tail in range(self.nentity)]
            tmp[tail] = (0, tail)
        else:
            raise ValueError('negative batch mode %s not supported' % self.mode)
            
        tmp = torch.LongTensor(tmp)            
        filter_bias = tmp[:, 0].float()
        negative_sample = tmp[:, 1]

        positive_sample = torch.LongTensor((head, relation, tail))
            
        return positive_sample, negative_sample, filter_bias, self.mode
    
    # 这是一个静态方法，用于将一个 batch 的数据拼接成一个四元组。具体来说，这个方法接收一个 batch 的数据列表 data ，该列表中每个元素都是由 TestDataset 中的 __getitem__ 方法返回的四元组：positive_sample、negative_sample、filter_bias和mode。这些四元组表示正样本、负样本、负样本的过滤器和训练模式。此方法将这些四元组中的张量拼接成一个四元组，以便传递给模型进行训练。具体来说，它会执行以下操作：
        # 对positive_sample和negative_sample这两个三元组张量列表，分别在第0维上进行拼接，组成一个包含所有三元组的张量。
        # 对filter_bias这个一维张量列表在第0维上进行拼接，组成一个包含所有张量的一维张量。
        # 将每个四元组中的模式mode都设置为data列表中第一个元素的模式。
        # 最后返回由这些张量和模式组成的四元组。
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        filter_bias = torch.stack([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, filter_bias, mode

# 双向独热生成器
# 通过接受头和尾部数据加载器的参数，它可以在这两个数据迭代器之间来回切换。这种技巧可以确保每个实体和关系都至少有一个正的和负的例子，可以有效减少训练中的负面影响。    
class BidirectionalOneShotIterator(object):
    def __init__(self, dataloader_head, dataloader_tail):
        self.iterator_head = self.one_shot_iterator(dataloader_head)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.step = 0
    
    # 奇数返回尾，偶数返回头    
    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.iterator_head)
        else:
            data = next(self.iterator_tail)
        return data
    
    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data
