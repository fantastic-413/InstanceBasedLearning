#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os
import random

import numpy as np
import torch

from torch.utils.data import DataLoader

from model import KGEModel

from dataloader import TrainDataset
from dataloader import BidirectionalOneShotIterator

import math
from datetime import datetime
from tqdm import tqdm

# 固定随机数种子
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

def parse_args(args=None):
    # argparse 是 python 用于解析命令行参数和选项的标准模块，我们常常可以把 argparse 的使用简化成下面四个步骤
    # 首先导入该模块；
    # 1：import argparse 
    # 然后创建一个解析对象；
    # 2：parser = argparse.ArgumentParser()
    # 然后向该对象中添加你要关注的命令行参数和选项，每一个 add_argument 方法对应一个你要关注的参数或选项；
    # 3：parser.add_argument()
    # 最后调用 parse_args() 方法进行解析；
    # 4：parser.parse_args()
    # 解析成功之后即可使用。
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', action='store_true', help='use GPU')
    
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--evaluate_train', action='store_true', help='Evaluate on training data')
    
    parser.add_argument('--countries', action='store_true', help='Use Countries S1/S2/S3 datasets')
    parser.add_argument('--regions', type=int, nargs='+', default=None, 
                        help='Region Id for Countries S1/S2/S3 datasets, DO NOT MANUALLY SET')
    
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--model', default='TransE', type=str)
    parser.add_argument('-de', '--double_entity_embedding', action='store_true')
    parser.add_argument('-dr', '--double_relation_embedding', action='store_true')
    
    parser.add_argument('-n', '--negative_sample_size', default=128, type=int)
    parser.add_argument('-d', '--hidden_dim', default=500, type=int)
    parser.add_argument('-g', '--gamma', default=12.0, type=float)
    parser.add_argument('-adv', '--negative_adversarial_sampling', action='store_true')
    parser.add_argument('-a', '--adversarial_temperature', default=1.0, type=float)
    parser.add_argument('-b', '--batch_size', default=1024, type=int)
    parser.add_argument('-r', '--regularization', default=0.0, type=float)
    parser.add_argument('--test_batch_size', default=4, type=int, help='valid/test batch size')
    parser.add_argument('--uni_weight', action='store_true', 
                        help='Otherwise use subsampling weighting like in word2vec') # 否则使用 word2vec 中的子采样权重
    
    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=10, type=int)
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('-save', '--save_path', default=None, type=str)
    parser.add_argument('--max_steps', default=100000, type=int)
    parser.add_argument('--warm_up_steps', default=None, type=int)
    
    parser.add_argument('--save_checkpoint_steps', default=10000, type=int)
    parser.add_argument('--valid_steps', default=10000, type=int)
    parser.add_argument('--log_steps', default=100, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')
    
    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--ible_weight', default=0.0, type=float)
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--mlp', default=False, type=bool)
    parser.add_argument('--relation_aware', default=None, type=str)
    parser.add_argument('--pooling', default='mean', type=str)
    parser.add_argument('--loss', default='crossentropy', type=str)
    parser.add_argument('--cosine', default=False, type=bool)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--activation', default='none', type=str)
    parser.add_argument('--intermediate_dim', default=1024, type=int)

    args = parser.parse_args(args)

    if args.relation_aware == 'none':
        args.relation_aware = None
    
    return args

def override_config(args):
    '''
    Override model and data configuration 覆盖模型和数据配置
    '''
    
    with open(os.path.join(args.init_checkpoint, 'config.json'), 'r') as fjson:
        argparse_dict = json.load(fjson)
    
    args.countries = argparse_dict['countries']
    if args.data_path is None:
        args.data_path = argparse_dict['data_path']
    args.model = argparse_dict['model']
    # 这两行代码是用来覆盖原始的配置文件中关于是否需要双重嵌入的设置的。
    # 在知识图谱中，一个实体节点通常由一个嵌入向量表示。但是在一些模型中，一个实体节点可能会有多个嵌入向量表示，例如TransD模型，它会为每个实体节点分别计算一个嵌入向量和一个投影向量。这些代码将从原始的配置文件中读取关于是否需要这种双重嵌入的设置，并将其应用到新的运行配置中。
    args.double_entity_embedding = argparse_dict['double_entity_embedding']
    args.double_relation_embedding = argparse_dict['double_relation_embedding']
    # hidden_dim: 整数类型，表示嵌入向量的维度。
    args.hidden_dim = argparse_dict['hidden_dim']
    args.test_batch_size = argparse_dict['test_batch_size']
    
def save_model(model, optimizer, save_variable_list, args):
    '''
    Save the parameters of the model and the optimizer,             保存模型和优化器的参数，
    as well as some other variables such as step and learning_rate  以及一些其他的变量比如step和learning_rate
    '''
    
    # 将命令行参数 args 转换为字典形式 argparse_dict。
    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    # 使用 torch.save() 函数将模型的状态、优化器的状态、以及一些其他变量打包成一个字典并保存到文件系统中。
    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(args.save_path, 'checkpoint')
    )
    
    # 将训练好的模型中的实体嵌入和关系嵌入保存在Numpy数组中
    # 过 detach() 方法将其从计算图中分离出来，并通过 cpu() 方法将其转移到 CPU 上，最后通过 numpy() 方法将其转换为 Numpy 数组。
    entity_embedding = model.entity_embedding.detach().cpu().numpy()
    # np.save() 方法将这些数组保存在指定路径下的文件中。
    np.save(
        os.path.join(args.save_path, 'entity_embedding'), 
        entity_embedding
    )
    
    relation_embedding = model.relation_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'relation_embedding'), 
        relation_embedding
    )

def read_triple(file_path, entity2id, relation2id):
    '''
    Read triples and map them into ids.  读取三元组并将它们映射到 id。
    '''
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples

def set_logger(args):
    '''
    Write logs to checkpoint and console  设置日志输出到检查点文件和控制台
    '''

    if args.do_train:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'train.log')
    else:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'test.log')

    # logging.basicConfig() 配置日志的基本信息，包括格式、日志级别、日期格式、文件名等。然后再将日志输出到文件中。
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    # logging.StreamHandler() 创建一个输出到控制台的日志处理器。
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # logging.Formatter() 配置日志信息的格式，包括时间、级别和信息内容。
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    # logging.getLogger('') 返回根日志记录器，可以添加日志处理器，指定日志级别等。此处将输出到控制台的处理器添加到根记录器中，从而输出日志信息到控制台。
    logging.getLogger('').addHandler(console)


def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs 打印评估日志
    '''
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))

        
def main(args):
    # 设置随机数种子
    set_seed(args.seed)
    # 开始训练、开始验证、开始测试 至少选择一个
    if (not args.do_train) and (not args.do_valid) and (not args.do_test):
        raise ValueError('one of train/val/test mode must be choosed.')
    
    # 读取配置文件并检查参数的有效性。
        # 如果指定了 init_checkpoint（预训练模型的路径），则通过 override_config(args) 函数读取模型的配置文件并覆盖参数;
        # 如果没有指定 init_checkpoint，则检查是否指定了数据集路径 data_path，如果没有指定则抛出异常。
    if args.init_checkpoint:
        override_config(args)
    elif args.data_path is None:
        raise ValueError('one of init_checkpoint/data_path must be choosed.')

    # 如果用户选择训练模型（args.do_train=True）并且没有指定保存路径（args.save_path=None），则会抛出异常提示用户需要指定保存路径。
    if args.do_train and args.save_path is None:
        raise ValueError('Where do you want to save your trained model?')
    
    args.save_path = f"models/{args.model}-{args.data_path.split('/')[1]}-{datetime.now()}"
    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    # Write logs to checkpoint and console  设置日志输出到检查点文件和控制台
    set_logger(args)
    
    # 从文件中读取实体和关系
    with open(os.path.join(args.data_path, 'entities.dict')) as fin:
        entity2id = dict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)

    with open(os.path.join(args.data_path, 'relations.dict')) as fin:
        relation2id = dict()
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)
    
    # Read regions for Countries S* datasets 读取国家 S* 数据集的区域(本论文中未使用)
    if args.countries:
        regions = list()
        with open(os.path.join(args.data_path, 'regions.list')) as fin:
            for line in fin:
                region = line.strip()
                regions.append(entity2id[region])
        args.regions = regions

    nentity = len(entity2id)
    nrelation = len(relation2id)
    
    args.nentity = nentity
    args.nrelation = nrelation
    
    # 遍历 args 对象将每个属性及其对应的值打印出来，以便于开发者查看。
    for arg in vars(args):
        print(arg, getattr(args, arg))

    logging.info('Model: %s' % args.model)
    logging.info('Data Path: %s' % args.data_path)
    logging.info('#entity: %d' % nentity)
    logging.info('#relation: %d' % nrelation)
    
    train_triples = read_triple(os.path.join(args.data_path, 'train.txt'), entity2id, relation2id)
    logging.info('#train: %d' % len(train_triples))
    valid_triples = read_triple(os.path.join(args.data_path, 'valid.txt'), entity2id, relation2id)
    logging.info('#valid: %d' % len(valid_triples))
    test_triples = read_triple(os.path.join(args.data_path, 'test.txt'), entity2id, relation2id)
    logging.info('#test: %d' % len(test_triples))
    
    #All true triples
    all_true_triples = train_triples + valid_triples + test_triples
    
    # hidden_dim: 整数类型，表示嵌入向量的维度。
    # ible_weight: 浮点数类型，表示ible loss的权重参数。
    # mlp: 布尔类型，表示是否使用基于多层感知器的分数函数。
    # relation_aware: 布尔类型，表示是否使用基于关系嵌入向量的分数函数。
    kge_model = KGEModel(
        args=args,
        model_name=args.model,
        nentity=nentity,
        nrelation=nrelation,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        double_entity_embedding=args.double_entity_embedding,
        double_relation_embedding=args.double_relation_embedding,
        train_triples=train_triples,
        ible_weight=args.ible_weight,
        mlp=args.mlp,
        relation_aware=args.relation_aware,
    )
    
    logging.info('Model Parameter Configuration:')
    for name, param in kge_model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))

    if args.cuda:
        kge_model = kge_model.cuda()

    if args.do_train:
        # Set training dataloader iterator 设置训练数据加载器、迭代器
        train_dataloader_head = DataLoader(
            TrainDataset(train_triples, nentity, nrelation, args.negative_sample_size, 'head-batch'), 
            batch_size=args.batch_size,
            shuffle=True, 
            num_workers=max(1, args.cpu_num//2),
            collate_fn=TrainDataset.collate_fn
        )
        
        train_dataloader_tail = DataLoader(
            TrainDataset(train_triples, nentity, nrelation, args.negative_sample_size, 'tail-batch'), 
            batch_size=args.batch_size,
            shuffle=True, 
            num_workers=max(1, args.cpu_num//2),
            collate_fn=TrainDataset.collate_fn
        )
        
        # 将两个 DataLoader 传递给 BidirectionalOneShotIterator ，创建了一个可迭代对象 train_iterator ，该迭代器会在每个 epoch 中依次遍历两个 DataLoader ，以产生用于训练的数据样本。
        train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)
        
        # Set training configuration 设置训练配置
        current_learning_rate = args.learning_rate
        # 定义优化器为AdamW，并设置优化器对象的优化参数（需要迭代优化的参数或者定义参数组的dicts），学习率
        optimizer = torch.optim.AdamW(
            # 过滤不需要迭代优化（计算梯度）的参数
            filter(lambda p: p.requires_grad, kge_model.parameters()), 
            lr=current_learning_rate
        )
        # create scheduler 设置学习率调度器
        # 设置 warm up steps
        # 创建一个指数衰减学习率调度程序 lr_scheduler，初始学习率为 optimizer 的学习率，衰减因子为0.95或0.1，具体取决于 warm_up_steps 是否存在。
        if args.warm_up_steps:
            warm_up_steps = args.warm_up_steps
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
        else:
            warm_up_steps = args.max_steps // 2
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.1)
        
        # param_norm 计算模型参数的范数，grad_norm 计算梯度的范数。这两个函数在训练过程中用于监测模型参数和梯度的大小。
        param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
        grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))

    # 是否从已有模型检查点——某个 checkpoint 开始训练
    if args.init_checkpoint:
        # Restore model from checkpoint directory 从已有模型检查点目录还原模型
        logging.info('Loading checkpoint %s...' % args.init_checkpoint)
        checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint'))
        init_step = checkpoint['step']
        kge_model.load_state_dict(checkpoint['model_state_dict'])
        # 如果 args.do_train 为 True，则同时恢复 optimizer。
        if args.do_train:
            current_learning_rate = checkpoint['current_learning_rate']
            warm_up_steps = checkpoint['warm_up_steps']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        logging.info('Ramdomly Initializing %s Model...' % args.model)
        init_step = 0
    
    # 是否有预训练好的模型
    if args.pretrained:
        path = os.path.join(args.data_path, f'RotatE_{args.hidden_dim}')
        if not os.path.exists(path):
            raise ValueError(f'Pretrained checkpoint of proposed dimension {args.hidden_dim} does not exist!')

        # 从预训练好的模型中加载实体和关系的嵌入向量，并输出加载的路径和信息
        kge_model.entity_embedding.data = torch.from_numpy(np.load(os.path.join(path, 'entity_embedding.npy'))).cuda()
        kge_model.relation_embedding.data = torch.from_numpy(np.load(os.path.join(path, 'relation_embedding.npy'))).cuda()
        logging.info("Loaded pretrained model from %s" % path)
    
    step = init_step
    best_valid_metrics = {}
    best_test_metrics = {}

    logging.info('Start Training...')
    logging.info('init_step = %d' % init_step)
    logging.info('max_step = %d' % args.max_steps)
    logging.info('valid_steps = %d' % args.valid_steps)
    logging.info('batch_size = %d' % args.batch_size)
    logging.info('negative_adversarial_sampling = %d' % args.negative_adversarial_sampling)
    logging.info('hidden_dim = %d' % args.hidden_dim)
    logging.info('gamma = %f' % args.gamma)
    logging.info('negative_adversarial_sampling = %s' % str(args.negative_adversarial_sampling))
    # 如果使用了负例对抗采样，则还会打印出对抗温度参数
    if args.negative_adversarial_sampling:
        logging.info('adversarial_temperature = %f' % args.adversarial_temperature)
    
    # Set valid dataloader as it would be evaluated during training 设置有效的 dataloader ，因为它将在训练期间进行评估

    # 如果设置了参数 args.do_valid 为 True，那么首次会在训练过程中对验证集和测试集进行评估
    if args.do_valid:
        logging.info('Evaluating on Valid Dataset...')
        metrics = kge_model.test_step(kge_model, valid_triples, all_true_triples, args)
        log_metrics('Valid', step, metrics)
        for k, v in metrics.items():
            if k not in best_valid_metrics:
                best_valid_metrics[k] = v
            else:
                # 如果某个评估指标是 "MR"（平均排名），那么评估结果越小越好；否则，评估结果越大越好。
                if k == "MR":
                    if best_valid_metrics[k] > v:
                        best_valid_metrics[k] = v
                else:
                    if best_valid_metrics[k] < v:
                        best_valid_metrics[k] = v
        log_metrics('Best-Valid', step, best_valid_metrics)

        logging.info('Evaluating on Test Dataset...')
        metrics = kge_model.test_step(kge_model, test_triples, all_true_triples, args)
        log_metrics('Test', step, metrics)
        for k, v in metrics.items():
            if k not in best_test_metrics:
                best_test_metrics[k] = v
            else:
                if k in ["MR", "loss", 'rotate_loss', 'identity_matrix_loss']:
                    if best_test_metrics[k] > v:
                        best_test_metrics[k] = v
                else:
                    if best_test_metrics[k] < v:
                        best_test_metrics[k] = v
        log_metrics('Best-Test', step, best_test_metrics)
    
    # 如果args.do_train为真，则会执行训练过程。
    if args.do_train:
        logging.info('learning_rate = %d' % current_learning_rate)

        training_logs = []
        
        #Training Loop 训练循环从 init_step 开始，执行 args.max_steps - init_step 次迭代
        for step in tqdm(range(init_step, args.max_steps)):
            # 在每个训练步骤中，调用 kge_model 的 train_step 方法来训练模型。
            log = kge_model.train_step(kge_model, optimizer, train_iterator, args)
            # 将返回的训练指标（如损失、准确度等）添加到 training_logs 列表中，用于后续的日志记录和分析。
            training_logs.append(log)
            
            # 如果 step 是 warm_up_steps 的倍数且不为 0，则执行学习率调度器的 step 方法，以便在训练过程中逐渐减小学习率。
            if step % warm_up_steps == 0 and step != 0:
                lr_scheduler.step()
            
            # 如果 step 是 args.save_checkpoint_steps 的倍数，则将当前的 step 、 current_learning_rate 和优化器和模型参数保存到文件中。
            if step % args.save_checkpoint_steps == 0:
                save_variable_list = {
                    'step': step, 
                    'current_learning_rate': current_learning_rate,
                }
                save_model(kge_model, optimizer, save_variable_list, args)
            
            # 如果 step 是 args.log_steps 的倍数，则计算训练集上的平均指标，然后将其记录到日志中。    
            if step % args.log_steps == 0:
                metrics = {'gnorm': grad_norm(kge_model), 'pnorm': param_norm(kge_model)}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)
                log_metrics('Training average', step, metrics)
                training_logs = []
                
            # 如果 args.do_valid 为真， step 是 args.valid_steps 的倍数且不为 0，则执行验证集上的评估。
            if args.do_valid and step % args.valid_steps == 0 and step != 0:
                logging.info('Evaluating on Valid Dataset...')
                metrics = kge_model.test_step(kge_model, valid_triples, all_true_triples, args)
                log_metrics('Valid', step, metrics)
                for k, v in metrics.items():
                    if k not in best_valid_metrics:
                        best_valid_metrics[k] = v
                    else:
                        if k == "MR":
                            if best_valid_metrics[k] > v:
                                best_valid_metrics[k] = v
                        else:
                            if best_valid_metrics[k] < v:
                                best_valid_metrics[k] = v
                log_metrics('Best-Valid', step, best_valid_metrics)

            # 如果 args.do_test 为真， step 是 args.valid_steps 的倍数且不为 0，则执行测试集上的评估。
            if args.do_test and step % args.valid_steps == 0 and step != 0:
                logging.info('Evaluating on Test Dataset...')
                metrics = kge_model.test_step(kge_model, test_triples, all_true_triples, args)
                log_metrics('Test', step, metrics)
                for k, v in metrics.items():
                    if k not in best_test_metrics:
                        best_test_metrics[k] = v
                    else:
                        if k == "MR":
                            if best_test_metrics[k] > v:
                                best_test_metrics[k] = v
                        else:
                            if best_test_metrics[k] < v:
                                best_test_metrics[k] = v
                log_metrics('Best-Test', step, best_test_metrics)

        # 在训练过程结束时，将当前的 step 、current_learning_rate 和 warm_up_steps 保存到文件中。
        save_variable_list = {
            'step': step, 
            'current_learning_rate': current_learning_rate,
            'warm_up_steps': warm_up_steps
        }
        save_model(kge_model, optimizer, save_variable_list, args)
    
    # 这是训练结束后，由 do_valid、do_test 决定最后一次输出对测试集、训练集的评估指标    
    if args.do_valid:
        logging.info('Evaluating on Valid Dataset...')
        metrics = kge_model.test_step(kge_model, valid_triples, all_true_triples, args)
        log_metrics('Valid', step, metrics)

    if args.do_test:
        logging.info('Evaluating on Test Dataset...')
        metrics = kge_model.test_step(kge_model, test_triples, all_true_triples, args)
        log_metrics('Test', step, metrics)

    # if args.evaluate_train:
    #     logging.info('Evaluating on Training Dataset...')
    #     metrics = kge_model.test_step(kge_model, train_triples, all_true_triples, args)
    #     log_metrics('Test', step, metrics)

# 首先解析命令行参数，然后调用 main() 函数来训练和测试知识图谱嵌入模型。如果您在命令行中执行此脚本，则会执行此段代码。        
if __name__ == '__main__':
    args = parse_args()
    main(args)
