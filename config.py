import argparse

from utils.train_utils import add_flags_from_config

config_args = {
    'training_config': {
        'lr': (0.001, 'learning rate'),
        'dropout': (0.2, 'dropout probability'),
        'dropoutgeo': (0.2, 'dropout probability'),
        'cuda': (3, 'which cuda device to use (-1 for cpu training)'),
        'epoch': (1000, 'maximum number of epochs to train for'),
        'decay': (0.0001, 'l2 regularization strength'),
        'optimizer': ('adam', 'which optimizer to use, can be any of [Adam, RiemannianAdam]'),
        'momentum': (0.999, 'momentum in optimizer'),
        'patience': (200, 'patience for early stopping'),
        'seed': (1234, 'seed for training'),
        'save': (0, '1 to save model and logs and 0 otherwise'),
        'save-dir': ('./experiment3', 'path to save training logs and model weights (defaults to logs/task/date/run/)'),
        'sweep-c': (0, ''),
        'print-epoch': (True, ''),
        'runtime': (1, 'runtime'),
    },
    'model_config': {
        'task': ('nc', 'which tasks to train on, can be any of [lp, nc]'),
        'model': ('GeoGCN', 'which encoder to use, can be any of [Shallow, MLP, HNN, GCN, GAT, HGCN,GeoGCN]'),
        'hidden': (64, 'embedding dimension'),
        'num_layers': (3, 'number of hidden layers in encoder'),
        'act': ('relu', 'which activation function to use (or None for no activation)'),
        'n_heads': (4, 'number of attention heads for graph attention networks, must be a divisor dim'),
        'alpha': (0.2, 'alpha for leakyrelu in graph attention networks'),
        'use-att': (0, 'whether to use hyperbolic attention or not'),
        'odemap': ('h1extend', 'which ode function to use, can be any of [linear,v1, v2, v1learn,v2learn,ricci,riccilearn,v1xlearn,v5,v5learn,v5extend]'),
        'logmethods': ('ode', 'which log to use, can be any of [ode, vanilla,]'),
         'kdim': (8, 'v embedding dimension'),
         'agg': ('GCN', 'which agg to use, can be any of [ GCN, GAT]'),
         'odemethod': ('euler', 'which ode method to use, can be any of [ euler, rk4, dopri5, implicit_adams]'),
         'step_size': (1.0, 'ode step_size'),
         'vt': ('clone', 'which ode method to use, can be any of [ clone, fc, fcorth, qnet,mlp]'),
         'sign': (1, 'ode step_size'),
         'rank': (1, 'rank for geognn'),


    },
    'data_config': {
        'dataset': ('Cora', 'which dataset to use[coauthor, amazoncomputer,amazonphoto,cornell, wisconsin, texas]'),
        'split-seed': (1234, 'seed for data splits (train/test/val)'),
        'random_split': (False, 'random_split for data splits (train/test/val)'),
        'val-prop': (0.05, 'proportion of validation edges for link prediction'),
        'test-prop': (0.1, 'proportion of test edges for link prediction'),
        'use-feats': (1, 'whether to use node features or not'),
        'normalize-feats': (1, 'whether to normalize input node features'),
        'normalize-adj': (1, 'whether to row-normalize the adjacency matrix'),
    }
}

parser = argparse.ArgumentParser()
for _, config_dict in config_args.items():
    parser = add_flags_from_config(parser, config_dict)
