from __future__ import division
from __future__ import print_function

import datetime
import json
import logging
import os
import pickle
import time
import dgl.graph_index
import numpy as np
import torch
from config import parser
from utils.data_utils import get_dataset
from utils.train_utils import get_dir_name, format_metrics
import scipy.sparse as sp
import dgl
import statistics

import random
from layers.ham_layers_v1 import HamGraphConvolution
from prettytable import PrettyTable
import sys
from torch_geometric.utils import get_laplacian,to_dense_adj,to_scipy_sparse_matrix,add_remaining_self_loops
import torch.nn.functional as F
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add
from utils.data_utils_lp import load_data
from utils.eval_utils import acc_f1


class HamGNN(torch.nn.Module):
    def __init__(self, args, in_dim, hidden_dim, num_classes, num_layers):
        super(HamGNN, self).__init__()
        self.args = args
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        # self.out_dim = out_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.dropout = args.dropout

        self.linear_layer1 = torch.nn.Linear(in_dim, hidden_dim)
        self.liner_layer2 = torch.nn.Linear(hidden_dim, num_classes, bias=False)
        self.layers = torch.nn.ModuleList()
        self.layers_bn = torch.nn.ModuleList()
        # self.layers.append(HamGraphConvolution(hidden_dim,args))
        for i in range(num_layers):
            self.layers.append(HamGraphConvolution(hidden_dim,args))
            # self.layers_bn.append(torch.nn.BatchNorm1d(hidden_dim))
        # self.layers.append(HamGraphConvolution(hidden_dim, out_dim, num_classes, dropout, use_cuda))

        # self.bn_input = torch.nn.BatchNorm1d(hidden_dim)

        if not args.act:
            self.act = lambda x: x
        elif args.act == 'elu':
            self.act = F.elu
        else:
            self.act = getattr(torch, args.act)
        if args.n_classes > 2:
            self.f1_average = 'micro'
        else:
            self.f1_average = 'binary'

    def forward(self, features, adj):
        # h = F.dropout(features, p=self.dropout, training=self.training)
        h = features
        h = self.linear_layer1(h)
        # h = self.bn_input(h)
        for i, layer in enumerate(self.layers):
            h = layer(h,adj)

            # h = self.layers_bn[i](h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.act(h)

        h = self.liner_layer2(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        return h
    def compute_metrics(self, embeddings, data, split):
        idx = data[f'idx_{split}']
        output = F.log_softmax(embeddings[idx], dim=1)
        loss = F.nll_loss(output, data['labels'][idx])
        acc, f1 = acc_f1(output, data['labels'][idx], average=self.f1_average)
        metrics = {'loss': loss, 'acc': acc, 'f1': f1}
        return metrics

def set_seed(seed=7):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def print_model_params(model):
  print(model)
  table = PrettyTable(["Modules", "Parameters"])
  total_params = 0
  ham_params = 0
  for name, parameter in model.named_parameters():
      if not parameter.requires_grad: continue
      params = parameter.numel()
      table.add_row([name, params])
      total_params += params
  print(table)
  print(f"Total Trainable Params: {total_params}")

def get_optimizer(name, parameters, lr, weight_decay=0):
  if name == 'sgd':
    return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'rmsprop':
    return torch.optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'adagrad':
    return torch.optim.Adagrad(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'adam':
    return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'adamax':
    return torch.optim.Adamax(parameters, lr=lr, weight_decay=weight_decay)
  else:
    raise Exception("Unsupported optimizer: {}".format(name))

def test( model, data):
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index,data.edge_weight)
        accs = []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
            accs.append(acc)

        return accs

def train(args):

    set_seed(args.seed)
    args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'
    opt = vars(args)
    # Load data

    data = load_data(args, os.path.join('./data', args.dataset))



    args.n_nodes, args.feat_dim = data['features'].shape
    args.n_classes = data['labels'].max().item() + 1
    model = HamGNN(args, args.feat_dim, args.hidden, args.n_classes , args.num_layers).to(args.device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    print_model_params(model)
    print(str(model))

    # lap_sym = get_laplacian(data.edge_index, data.edge_weight,normalization='sym')
    # lap_edge_index, lap_edge_weight = get_laplacian(data.edge_index, edge_weight=data.edge_weight, normalization="sym",
    #                                                 num_nodes=data.x.shape[0])
    # lap = to_scipy_sparse_matrix(lap_edge_index, edge_attr=lap_edge_weight)
    # lap = scipy_sparse_to_torch_sparse(lap)
    # print("laplacian shape: ", lap.shape)
    #
    # print("num of train samples: ", len(torch.nonzero(data.train_mask, as_tuple=True)[0]))
    # print("num of val samples: ", len(torch.nonzero(data.val_mask, as_tuple=True)[0]))
    # print("num of test samples: ", len(torch.nonzero(data.test_mask, as_tuple=True)[0]))
    optimizer = get_optimizer(opt['optimizer'], parameters, lr=opt['lr'], weight_decay=opt['decay'])
    criterion = torch.nn.CrossEntropyLoss()
    best_time = best_epoch = train_acc = val_acc = test_acc=counter = 0
    # if data.edge_weight is None:
    #     data.edge_weight = torch.ones((data.edge_index.size(1),), device=data.edge_index.device)
    data['features'] = data['features'].to(args.device)
    data['adj_train_norm'] = data['adj_train_norm'].to(args.device)
    data['labels'] = data['labels'].to(args.device)
    for epoch in range(1, opt['epoch']):
        start_time = time.time()
        model.train()
        optimizer.zero_grad()
        out = model(data['features'], data['adj_train_norm'])
        # loss = criterion(out[data.train_mask], data.y[data.train_mask])
        train_metrics = model.compute_metrics(out, data, 'train')
        train_metrics['loss'].backward()
        optimizer.step()
        train_loss = train_metrics['loss']
        tmp_train_acc = train_metrics['acc']
        val_metrics = model.compute_metrics(out, data, 'val')
        tmp_val_acc = val_metrics['acc']
        test_metrics = model.compute_metrics(out, data, 'test')
        tmp_test_acc = test_metrics['acc']
        print('Epoch: {:04d}'.format(epoch),
              'loss_train: {:.4f}'.format(train_loss.item()),
              'acc_train: {:.4f}'.format(tmp_train_acc),
              'acc_val: {:.4f}'.format(tmp_val_acc),
              'acc_test: {:.4f}'.format(tmp_test_acc),
              'time: {:.4f}s'.format(time.time() - start_time))
        if tmp_val_acc > val_acc:
            best_time = time.time() - start_time
            best_epoch = epoch
            train_acc = tmp_train_acc
            val_acc = tmp_val_acc
            test_acc = tmp_test_acc
            counter = 0
            if args.save:
                # mkdir if not exist
                if not os.path.exists('./model_saved'):
                    os.mkdir('./model_saved')
                model_name = './model_saved/' + args.dataset + '_' + str(args.odemap) + '_' + str(args.num_layers) + '_' + str(args.agg) + '_' + str(args.hidden) + '_'  + '.pt'
                torch.save(model.state_dict(),model_name )
        else:
            counter += 1
        if counter == args.patience:
            print('Early stopping!')
            break
    print("Optimization Finished!")
    print("Best Epoch: {:04d}".format(best_epoch),
            "train_acc= {:.4f}".format(train_acc),
            "val_acc= {:.4f}".format(val_acc),
            "test_acc= {:.4f}".format(test_acc),
            "time= {:.4f}s".format(best_time))

    if args.save:
        #load model and calculate dirichlet energy
        model.load_state_dict(torch.load(model_name))


    return test_acc







if __name__ == '__main__':
    args = parser.parse_args()
    test_acc_list = []
    # create log file
    if not os.path.exists('./log'):
        os.makedirs('./log')
    # add time stamp to log file name to avoid overwriting
    time_stamp = time.strftime("%H%M%S", time.localtime())
    log_file = './log/' + args.dataset + '_' + str(args.odemap) + '_' + str(args.num_layers) + '_' + str(args.agg) + '_' + time_stamp + '.txt'

    # write command line args to log file
    with open(log_file, 'a') as f:
        f.write(' '.join(sys.argv))
        f.write('\n')



    for i in range(args.runtime):
        test_acc = train(args)
        test_acc_list.append(test_acc)
        args.seed += 1
        # print i and test acc
        print("=====================================")
        print("runtime: ", i)
        print("test_acc: ", test_acc)
        # write log
        with open(log_file, 'a') as f:
            f.write('runtime: ' + str(i) + ' test_acc: ' + str(test_acc) + ' ')
            f.write('\n')
        print("test_acc_list: ", test_acc_list)
        print("mean: ", np.mean(test_acc_list))
        print("std: ", np.std(test_acc_list))
    # write log
    with open(log_file, 'a') as f:
        f.write('test_acc_list: ' + str(test_acc_list) + ' ')
        f.write('\n')
        f.write('mean: ' + str(np.mean(test_acc_list)) + ' ')
        f.write('\n')
        f.write('std: ' + str(np.std(test_acc_list)) + ' ')
        f.write('\n')
        #dump args dict  to log
        json.dump(vars(args), f, indent=4)
    # change saved log file name to include mean
    os.rename(log_file, log_file[:-4] + '_mean_' + str(np.mean(test_acc_list)) + '.txt')





