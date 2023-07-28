"""Data utils functions for pre-processing and data loading."""
import os
import pickle as pkl
import sys

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import dgl
from torch_geometric.datasets import WebKB
from torch_geometric.utils import to_scipy_sparse_matrix
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.datasets import Planetoid, Amazon, Coauthor
from scipy import sparse
from torch_geometric.utils.loop import add_remaining_self_loops,remove_self_loops
from torch_geometric.utils import to_undirected, dense_to_sparse,from_scipy_sparse_matrix
import pickle as pkl
import networkx as nx
import scipy.sparse as sp

from .heterophilic import  get_fixed_splits, generate_random_splits
from .heterophilic import WebKB, WikipediaNetwork, Actor



class MyOwnDataset(InMemoryDataset):
  def __init__(self, root, name, transform=None, pre_transform=None):
    super().__init__(None, transform, pre_transform)
def torch_sparse_to_coo(adj):
    m_index = adj._indices().cpu().numpy()
    row = m_index[0]
    col = m_index[1]
    data = adj._values().cpu().numpy()
    sp_matrix = sp.coo_matrix((data, (row, col)), shape=(adj.size()[0], adj.size()[1]))
    return sp_matrix
def get_dataset(opt: dict, data_dir, use_lcc: bool = False, split=0) -> InMemoryDataset:
  ds = opt['dataset']
  path = os.path.join(data_dir, ds)
  if ds in ['Cora', 'Citeseer', 'Pubmed']:
    dataset = Planetoid(path, ds,transform=T.NormalizeFeatures())
    use_lcc = False
    # if opt["random_splits"]:
    #   data = generate_random_splits(dataset.data, train_rate=0.6, val_rate=0.2)
    #   dataset.data = data
    #   print("random_splits with train_rate=0.6, val_rate=0.2")
  elif ds in ['Computers', 'Photo']:
    dataset = Amazon(path, ds)
  elif ds == 'CoauthorCS':
    dataset = Coauthor(path, 'CS')


  elif ds in ['chameleon', 'squirrel']:
    dataset = WikipediaNetwork(root=path, name=ds, transform=T.NormalizeFeatures())
    if opt["random_splits"]:
      data = generate_random_splits(dataset.data, train_rate=0.6, val_rate=0.2)
      dataset.data = data
      print("random_splits with train_rate=0.6, val_rate=0.2")
    else:
      data = get_fixed_splits(dataset.data, ds, path, split)
      dataset.data = data
      print("fixed_splits with splits number: ", split)
    use_lcc = False
  elif ds == 'film':
    dataset = Actor(root=path, transform=T.NormalizeFeatures())
    if opt["random_splits"]:
      data = generate_random_splits(dataset.data, train_rate=0.6, val_rate=0.2)
      dataset.data = data
      print("random_splits with train_rate=0.6, val_rate=0.2")
    else:
      data = get_fixed_splits(dataset.data, ds, path, split)
      dataset.data = data
      print("fixed_splits with splits number: ", split)
    use_lcc = False


  elif ds == 'ogbn-arxiv':
    dataset = PygNodePropPredDataset(name=ds, root=path,
                                     transform=T.ToSparseTensor())
    use_lcc = False  # never need to calculate the lcc with ogb datasets
  elif ds == 'airport':
    dataset = MyOwnDataset(path, name=ds)
    adj, features,labels = load_data_airport('airport', os.path.join('/home/ntu/Documents/zk/GeoGNN/data', 'airport'), return_label=True)

    val_prop, test_prop = 0.15, 0.15
    idx_val, idx_test, idx_train = split_data(labels, val_prop, test_prop, seed=1234)
    train_mask = torch.zeros(features.shape[0], dtype=bool, )
    train_mask[idx_train] = True
    test_mask = torch.zeros(features.shape[0], dtype=bool, )
    test_mask[idx_val] = True
    val_mask = torch.zeros(features.shape[0], dtype=bool, )
    val_mask[idx_test] = True
    adj =adj.tocoo()
    row, col,edge_attr = adj.row,adj.col,adj.data
    row =torch.LongTensor(row)
    col = torch.LongTensor(col)
    edge_attr = torch.FloatTensor(edge_attr)
    labels = torch.LongTensor(labels)
    features = torch.FloatTensor(features)
    edges = torch.stack([row, col], dim=0)
    data = Data(
      x=features,
      edge_index=torch.LongTensor(edges),
      edge_attr=edge_attr,
      y=labels,
      train_mask=train_mask,
      test_mask=test_mask,
      val_mask=val_mask
    )
    use_lcc = False
    dataset.data = data

  elif ds == 'disease':
    dataset = MyOwnDataset(path, name=ds)
    adj, features, labels = load_synthetic_data('disease_nc', 1,os.path.join('/home/ntu/Documents/zk/GeoGNN/data', 'disease_nc'), )
    val_prop, test_prop = 0.10, 0.60

    idx_val, idx_test, idx_train = split_data(labels, val_prop, test_prop, seed=1234)
    train_mask = torch.zeros(features.shape[0], dtype=bool, )
    train_mask[idx_train] = True
    test_mask = torch.zeros(features.shape[0], dtype=bool, )
    test_mask[idx_val] = True
    val_mask = torch.zeros(features.shape[0], dtype=bool, )
    val_mask[idx_test] = True
    adj = adj.tocoo()
    row, col, edge_attr = adj.row, adj.col, adj.data
    row = torch.LongTensor(row)
    col = torch.LongTensor(col)
    edge_attr = torch.FloatTensor(edge_attr)
    labels = torch.LongTensor(labels)
    features = features.toarray()
    features = torch.FloatTensor(features)
    edges = torch.stack([row, col], dim=0)
    data = Data(
      x=features,
      edge_index=torch.LongTensor(edges),
      edge_attr=edge_attr,
      y=labels,
      train_mask=train_mask,
      test_mask=test_mask,
      val_mask=val_mask
    )
    use_lcc = False
    dataset.data = data
  else:
    raise Exception('Unknown dataset.')

  # if use_lcc:
  #   lcc = get_largest_connected_component(dataset)
  #
  #   x_new = dataset.data.x[lcc]
  #   y_new = dataset.data.y[lcc]
  #
  #   row, col = dataset.data.edge_index.numpy()
  #   edges = [[i, j] for i, j in zip(row, col) if i in lcc and j in lcc]
  #   edges = remap_edges(edges, get_node_mapper(lcc))
  #
  #   data = Data(
  #     x=x_new,
  #     edge_index=torch.LongTensor(edges),
  #     y=y_new,
  #     train_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
  #     test_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
  #     val_mask=torch.zeros(y_new.size()[0], dtype=torch.bool)
  #   )
  #   dataset.data = data
  train_mask_exists = True
  try:
    dataset.data.train_mask
  except AttributeError:
    train_mask_exists = False

  if ds == 'ogbn-arxiv':
    split_idx = dataset.get_idx_split()
    ei = to_undirected(dataset.data.edge_index)
    data = Data(
    x=dataset.data.x,
    edge_index=ei,
    y=dataset.data.y,
    train_mask=split_idx['train'],
    test_mask=split_idx['test'],
    val_mask=split_idx['valid'])
    dataset.data = data
    train_mask_exists = True

  # #todo this currently breaks with heterophilic datasets if you don't pass --geom_gcn_splits
  # if (use_lcc or not train_mask_exists) and not opt['geom_gcn_splits']:
  #   dataset.data = set_train_val_test_split(
  #     12345,
  #     dataset.data,
  #     num_development=5000 if ds == "CoauthorCS" else 1500)


  return dataset


def normalize_proteins(adj):
    adj_t = adj.set_diag()
    deg = adj_t.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)


    return adj_t

def get_component(dataset: InMemoryDataset, start: int = 0) -> set:
    visited_nodes = set()
    queued_nodes = set([start])
    row, col = dataset.data.edge_index.numpy()
    while queued_nodes:
        current_node = queued_nodes.pop()
        visited_nodes.update([current_node])
        neighbors = col[np.where(row == current_node)[0]]
        neighbors = [n for n in neighbors if n not in visited_nodes and n not in queued_nodes]
        queued_nodes.update(neighbors)
    return visited_nodes


def get_largest_connected_component(dataset: InMemoryDataset) -> np.ndarray:
    remaining_nodes = set(range(dataset.data.x.shape[0]))
    comps = []
    while remaining_nodes:
        start = min(remaining_nodes)
        comp = get_component(dataset, start)
        comps.append(comp)
        remaining_nodes = remaining_nodes.difference(comp)
    return np.array(list(comps[np.argmax(list(map(len, comps)))]))


# ############### FEATURES PROCESSING ####################################


def process(adj, features, normalize_adj, normalize_feats):
    if sp.isspmatrix(features):
        features = np.array(features.todense())
    if normalize_feats:
        features = normalize(features)
    features = torch.Tensor(features)
    if normalize_adj:
        adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj, features


def normalize(mx):
    """Row-normalize sparse matrix."""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.Tensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def augment(adj, features, normalize_feats=True):
    deg = np.squeeze(np.sum(adj, axis=0).astype(int))
    deg[deg > 5] = 5
    deg_onehot = torch.tensor(np.eye(6)[deg], dtype=torch.float).squeeze()
    const_f = torch.ones(features.size(0), 1)
    features = torch.cat((features, deg_onehot, const_f), dim=1)
    return features


# ############### DATA SPLITS #####################################################

def set_train_val_test_split(
        seed: int,
        data: Data,
        num_development: int = 1500,
        num_per_class: int = 20) -> Data:
  rnd_state = np.random.RandomState(seed)
  num_nodes = data.y.shape[0]
  development_idx = rnd_state.choice(num_nodes, num_development, replace=False)
  test_idx = [i for i in np.arange(num_nodes) if i not in development_idx]

  train_idx = []
  rnd_state = np.random.RandomState(seed)
  for c in range(data.y.max() + 1):
    class_idx = development_idx[np.where(data.y[development_idx].cpu() == c)[0]]
    train_idx.extend(rnd_state.choice(class_idx, num_per_class, replace=False))

  val_idx = [i for i in development_idx if i not in train_idx]

  def get_mask(idx):
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[idx] = 1
    return mask

  data.train_mask = get_mask(train_idx)
  data.val_mask = get_mask(val_idx)
  data.test_mask = get_mask(test_idx)

  return data
def mask_edges(adj, val_prop, test_prop, seed):
    np.random.seed(seed)  # get tp edges
    x, y = sp.triu(adj).nonzero()
    pos_edges = np.array(list(zip(x, y)))
    np.random.shuffle(pos_edges)
    # get tn edges
    x, y = sp.triu(sp.csr_matrix(1. - adj.toarray())).nonzero()
    neg_edges = np.array(list(zip(x, y)))
    np.random.shuffle(neg_edges)

    m_pos = len(pos_edges)
    n_val = int(m_pos * val_prop)
    n_test = int(m_pos * test_prop)
    val_edges, test_edges, train_edges = pos_edges[:n_val], pos_edges[n_val:n_test + n_val], pos_edges[n_test + n_val:]
    val_edges_false, test_edges_false = neg_edges[:n_val], neg_edges[n_val:n_test + n_val]
    train_edges_false = np.concatenate([neg_edges, val_edges, test_edges], axis=0)
    adj_train = sp.csr_matrix((np.ones(train_edges.shape[0]), (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T
    return adj_train, torch.LongTensor(train_edges), torch.LongTensor(train_edges_false), torch.LongTensor(val_edges), \
           torch.LongTensor(val_edges_false), torch.LongTensor(test_edges), torch.LongTensor(
            test_edges_false)  

def split_data_flikr(labels, val_prop, test_prop, seed):
    np.random.seed(seed)
    nb_nodes = labels.shape[0]
    all_idx = np.arange(nb_nodes)
    np.random.shuffle(all_idx)
    all_idx = all_idx.tolist()
    nb_val = round(val_prop * nb_nodes)
    nb_test = round(test_prop * nb_nodes)
    idx_val = all_idx[:nb_val]
    idx_test = all_idx[nb_val:nb_val + nb_test]
    idx_train = all_idx[nb_val + nb_test:]
    print("number of all samples: ", len(all_idx))
    print('val sample: ',len(idx_val) )
    print('test sample: ', len(idx_test))
    print('train sample: ', len(idx_train))

    return idx_val,idx_test,idx_train
def split_data(labels, val_prop, test_prop, seed):
    np.random.seed(seed)
    nb_nodes = labels.shape[0]
    all_idx = np.arange(nb_nodes)
    pos_idx = labels.nonzero()[0]
    neg_idx = (1. - labels).nonzero()[0]
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)
    pos_idx = pos_idx.tolist()
    neg_idx = neg_idx.tolist()
    nb_pos_neg = min(len(pos_idx), len(neg_idx))
    nb_val = round(val_prop * nb_pos_neg)
    nb_test = round(test_prop * nb_pos_neg)
    idx_val_pos, idx_test_pos, idx_train_pos = pos_idx[:nb_val], pos_idx[nb_val:nb_val + nb_test], pos_idx[
                                                                                                   nb_val + nb_test:]
    idx_val_neg, idx_test_neg, idx_train_neg = neg_idx[:nb_val], neg_idx[nb_val:nb_val + nb_test], neg_idx[
                                                                                                   nb_val + nb_test:]

    print("number of total positive samples: ", len(pos_idx))
    print("number of total negetive samples: ", len(neg_idx))
    print("number of training positive samples: ", len(idx_train_pos))
    print("number of training negetive samples: ", len(idx_train_neg))
    print("number of val positive samples: ", len(idx_val_pos))
    print("number of val negetive samples: ", len(idx_val_neg))
    print("number of val positive samples: ", len(idx_test_pos))
    print("number of val negetive samples: ", len(idx_test_neg))


    return idx_val_pos + idx_val_neg, idx_test_pos + idx_test_neg, idx_train_pos + idx_train_neg


def bin_feat(feat, bins):
    digitized = np.digitize(feat, bins)
    return digitized - digitized.min()




# ############### DATASETS ####################################

def remap_edges(edges: list, mapper: dict) -> list:
    row = [e[0] for e in edges]
    col = [e[1] for e in edges]
    row = list(map(lambda x: mapper[x], row))
    col = list(map(lambda x: mapper[x], col))
    return [row, col]


def load_citation_data(dataset_str, use_feats, data_path, split_seed=None):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(os.path.join(data_path, "ind.{}.{}".format(dataset_str, names[i])), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(os.path.join(data_path, "ind.{}.test.index".format(dataset_str)))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended


    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    labels = np.argmax(labels, 1)

    idx_test = test_idx_range.tolist()
    idx_train = list(range(len(y)))
    idx_val = range(len(y), len(y) + 500)

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    if not use_feats:
        features = sp.eye(adj.shape[0])
    return adj, features, labels, idx_train, idx_val, idx_test

def get_node_mapper(lcc: np.ndarray) -> dict:
    mapper = {}
    counter = 0
    for node in lcc:
        mapper[node] = counter
        counter += 1
    return mapper

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_synthetic_data(dataset_str, use_feats, data_path):
    object_to_idx = {}
    idx_counter = 0
    edges = []
    with open(os.path.join(data_path, "{}.edges.csv".format(dataset_str)), 'r') as f:
        all_edges = f.readlines()
    for line in all_edges:
        n1, n2 = line.rstrip().split(',')
        if n1 in object_to_idx:
            i = object_to_idx[n1]
        else:
            i = idx_counter
            object_to_idx[n1] = i
            idx_counter += 1
        if n2 in object_to_idx:
            j = object_to_idx[n2]
        else:
            j = idx_counter
            object_to_idx[n2] = j
            idx_counter += 1
        edges.append((i, j))
    adj = np.zeros((len(object_to_idx), len(object_to_idx)))
    for i, j in edges:
        adj[i, j] = 1.  # comment this line for directed adjacency matrix
        adj[j, i] = 1.
    if use_feats:
        features = sp.load_npz(os.path.join(data_path, "{}.feats.npz".format(dataset_str)))
    else:
        features = sp.eye(adj.shape[0])
    labels = np.load(os.path.join(data_path, "{}.labels.npy".format(dataset_str)))
    return sp.csr_matrix(adj), features, labels


def load_data_airport(dataset_str, data_path, return_label=False):
    graph = pkl.load(open(os.path.join(data_path, dataset_str + '.p'), 'rb'))
    adj = nx.adjacency_matrix(graph)
    features = np.array([graph.nodes[u]['feat'] for u in graph.nodes()])
    if return_label:
        label_idx = 4
        labels = features[:, label_idx]
        features = features[:, :label_idx]
        labels = bin_feat(labels, bins=[7.0/7, 8.0/7, 9.0/7])
        return sp.csr_matrix(adj), features, labels
    else:
        return sp.csr_matrix(adj), features


def load_data_heter(dataset, split=0):
    name = dataset
    path = '../data/' + name
    dataset = WebKB(path, name=name)
    # split = 0
    data = dataset[0]
    splits_file = np.load(f'{path}/{name}/raw/{name}_split_0.6_0.2_{split}.npz')
    train_mask = splits_file['train_mask']
    val_mask = splits_file['val_mask']
    test_mask = splits_file['test_mask']

    data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
    data.val_mask = torch.tensor(val_mask, dtype=torch.bool)
    data.test_mask = torch.tensor(test_mask, dtype=torch.bool)

    features = data.x
    labels = data.y
    adj = to_scipy_sparse_matrix(data.edge_index)
    idx_train  = torch.nonzero(data.train_mask, as_tuple=True)[0]
    idx_val = torch.nonzero(data.val_mask, as_tuple=True)[0]
    idx_test = torch.nonzero(data.test_mask, as_tuple=True)[0]

    return adj, features, labels, idx_train, idx_val, idx_test


def load_new_data(dataset_str, use_feats, data_path):
    adj = sp.load_npz(os.path.join(data_path, "{}.edges.npz".format(dataset_str)))
    labels = np.load(os.path.join(data_path, "{}.labels.npy".format(dataset_str)))
    if use_feats:
        features = sp.load_npz(os.path.join(data_path, "{}.feats.npz".format(dataset_str)))
    else:
        features = sp.eye(adj.shape[0])
    return adj, features, labels



def get_train_val_test_split(random_state,
                             labels,
                             train_examples_per_class=None, val_examples_per_class=None,
                             test_examples_per_class=None,
                             train_size=None, val_size=None, test_size=None):
    random_state = np.random.RandomState(random_state)
    labels = torch.tensor(labels)
    labels = torch.nn.functional.one_hot(labels)
    num_samples, num_classes = labels.shape
    # num_samples, num_classes = labels.shape
    # num_samples = len(labels)
    # num_classes=int(labels.max() + 1)
    remaining_indices = list(range(num_samples))

    if train_examples_per_class is not None:
        train_indices = sample_per_class(random_state, labels, train_examples_per_class)
    else:
        # select train examples with no respect to class distribution
        train_indices = random_state.choice(remaining_indices, train_size, replace=False)

    if val_examples_per_class is not None:
        val_indices = sample_per_class(random_state, labels, val_examples_per_class, forbidden_indices=train_indices)
    else:
        remaining_indices = np.setdiff1d(remaining_indices, train_indices)
        val_indices = random_state.choice(remaining_indices, val_size, replace=False)

    forbidden_indices = np.concatenate((train_indices, val_indices))
    if test_examples_per_class is not None:
        test_indices = sample_per_class(random_state, labels, test_examples_per_class,
                                        forbidden_indices=forbidden_indices)
    elif test_size is not None:
        remaining_indices = np.setdiff1d(remaining_indices, forbidden_indices)
        test_indices = random_state.choice(remaining_indices, test_size, replace=False)
    else:
        test_indices = np.setdiff1d(remaining_indices, forbidden_indices)

    # assert that there are no duplicates in sets
    assert len(set(train_indices)) == len(train_indices)
    assert len(set(val_indices)) == len(val_indices)
    assert len(set(test_indices)) == len(test_indices)
    # assert sets are mutually exclusive
    assert len(set(train_indices) - set(val_indices)) == len(set(train_indices))
    assert len(set(train_indices) - set(test_indices)) == len(set(train_indices))
    assert len(set(val_indices) - set(test_indices)) == len(set(val_indices))
    if test_size is None and test_examples_per_class is None:
        # all indices must be part of the split
        assert len(np.concatenate((train_indices, val_indices, test_indices))) == num_samples

    if train_examples_per_class is not None:
        train_labels = labels[train_indices, :]
        train_labels = train_labels.numpy()
        train_sum = np.sum(train_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(train_sum).size == 1

    if val_examples_per_class is not None:
        val_labels = labels[val_indices, :]
        val_labels = val_labels.numpy()
        val_sum = np.sum(val_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(val_sum).size == 1

    if test_examples_per_class is not None:
        test_labels = labels[test_indices, :]
        test_labels = test_labels.numpy()
        test_sum = np.sum(test_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(test_sum).size == 1

    return train_indices, val_indices, test_indices


def sample_per_class(random_state, labels, num_examples_per_class, forbidden_indices=None):
    # labels = torch.tensor(labels)
    # labels = torch.nn.functional.one_hot(labels)
    num_samples, num_classes = labels.shape
    # num_samples = len(labels)
    # num_classes = int(labels.max() + 1)
    sample_indices_per_class = {index: [] for index in range(num_classes)}

    # get indices sorted by class
    for class_index in range(num_classes):
        for sample_index in range(num_samples):
            if labels[sample_index, class_index] > 0.0:
                if forbidden_indices is None or sample_index not in forbidden_indices:
                    sample_indices_per_class[class_index].append(sample_index)

    # get specified number of indices for each class
    return np.concatenate(
        [random_state.choice(sample_indices_per_class[class_index], num_examples_per_class, replace=False)
         for class_index in range(len(sample_indices_per_class))
         ])


# ############### DATASETS ####################################
def load_data_amazon(dataset_str, use_feats, data_path, split_seed=None):

    names1 = ['adj_matrix.npz', 'attr_matrix.npz']
    names2 = ['label_matrix.npy', 'train_mask.npy', 'val_mask.npy', 'test_mask.npy']
    objects = []
    for tmp_name in names1:
        tmp_path = 'data/{}/{}.{}'.format(dataset_str, dataset_str, tmp_name)
        objects.append(sp.load_npz(tmp_path))
    for tmp_name in names2:
        tmp_path = 'data/{}/{}.{}'.format(dataset_str, dataset_str, tmp_name)
        objects.append(np.load(tmp_path))
    adj, features, label_matrix, train_mask, val_mask, test_mask = tuple(objects)

    labels = np.argmax(label_matrix, 1)

    arr = np.arange(len(train_mask))
    idx_train = list(arr[train_mask])
    idx_val = list(arr[val_mask])
    idx_test = list(arr[test_mask])

    print("features shape: ", features.shape)
    print(" num of idx_train samples: ", len(idx_train))
    print(" num of idx_val samples: ", len(idx_val))
    print(" num of idx_test samples: ", len(idx_test))
    return adj, features, labels, idx_train, idx_val, idx_test