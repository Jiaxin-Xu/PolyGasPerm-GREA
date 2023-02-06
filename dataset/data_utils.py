
from ogb.utils.features import (atom_to_feature_vector,bond_to_feature_vector) 
from rdkit.Chem import AllChem
from rdkit import Chem
import numpy as np
import torch

bd_dict_x = {'CO2_CH4': [1.00E+05, 1.00E-02], 'H2_CH4': [5.00E+04, 2.50E+00], 'O2_N2': [5.00E+04, 1.00E-03], 'H2_N2': [1.00E+05, 1.00E-01], 'CO2_N2':[1.00E+06, 1.00E-04]}
bd_dict_y = {'CO2_CH4': [1.00E+05/2.21E+04, 1.00E-02/4.88E-06], 'H2_CH4': [5.00E+04/8.67E+04, 2.50E+00/5.64E-04], 'O2_N2': [5.00E+04/2.78E+04, 1.00E-03/2.43E-05], 'H2_N2': [1.00E+05/1.02E+05, 1.00E-01/9.21E-06], 'CO2_N2':[1.00E+06/3.05E+05, 1.00E-04/1.05E-08]}

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)
        
def getmorganfingerprint(mol):
    return list(AllChem.GetMorganFingerprintAsBitVect(mol, 2))

def getmaccsfingerprint(mol):
    fp = AllChem.GetMACCSKeysFingerprint(mol)
    return [int(b) for b in fp.ToBitString()]

def log_base(base, x):
    return np.log(x) / np.log(base) 

def smiles2graph(smiles_string):
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """
    mol = Chem.MolFromSmiles(smiles_string)

    # atoms
    atom_features_list = []
    atom_label = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
        atom_label.append(atom.GetSymbol())

    x = np.array(atom_features_list, dtype = np.int64)
    atom_label = np.array(atom_label, dtype = np.str)

    # bonds
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature_vector(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype = np.int64).T

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype = np.int64)

    else:   # mol has no bonds
        edge_index = np.empty((2, 0), dtype = np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype = np.int64)

    graph = dict()
    graph['edge_index'] = edge_index
    graph['edge_feat'] = edge_attr
    graph['node_feat'] = x
    graph['num_nodes'] = len(x)
    mgf = getmorganfingerprint(mol)
    maccs = getmaccsfingerprint(mol)
    mgf_feat = np.array(mgf, dtype="int64")
    maccs_feat = np.array(maccs, dtype="int64")
    graph['mgf']  = np.expand_dims(mgf_feat, axis=0) #2048
    graph['maccs']  = np.expand_dims(maccs_feat,axis=0) #167
    
    return graph 

import copy
import pathlib
import pandas as pd
from tqdm import tqdm
from torch_geometric.data import Data
def labeled2graphs(raw_dir):
    '''
        - raw_dir: the position where property csv stored,  
    '''
    path_suffix = pathlib.Path(raw_dir).suffix
    if path_suffix == '.csv':
        df_full = pd.read_csv(raw_dir, engine='python')
        df_full.set_index('SMILES', inplace=True)
        print(df_full[:5])
    else:
        raise ValueError("Support only csv.")
    graph_list = []
    for smiles_idx in tqdm(df_full.index[:]):
        graph_dict = smiles2graph(smiles_idx)
        props = df_full.loc[smiles_idx]
        for (name,value) in props.iteritems():
            graph_dict[name] = np.array([[value]])
        graph_list.append(graph_dict)
    return graph_list

def unlabel2graphs(raw_dir, property_name=None, drop_property=False):
    '''
        - raw_dir: the position where property csv stored,  
    '''
    path_suffix = pathlib.Path(raw_dir).suffix
    if path_suffix == '.csv':
        df_full = pd.read_csv(raw_dir, engine='python')
        print(df_full[:5])
        # select data without current property
        if drop_property:
            if len(property_name.split('_')) == 2:
                df_full = df_full[df_full['/'.join(property_name.split('_'))].isna()]
            else:
                df_full = df_full[df_full[property_name.split('_')[0]].isna()]
        df_full = df_full.dropna(subset=['SMILES'])
    elif path_suffix == '.txt':
        df_full = pd.read_csv(raw_dir, sep=" ", header=None, names=['SMILES'])
        print(df_full[:5])
    else:
        raise ValueError("Support only csv and txt.")
    graph_list = []
    for smiles_idx in tqdm(df_full['SMILES']):
        graph_dict = smiles2graph(smiles_idx)
        if len(property_name.split('_')) == 2:
            graph_dict['/'.join(property_name.split('_'))] = np.array([[np.nan]])
            graph_dict[property_name.split('_')[0]] = np.array([[np.nan]])
            graph_dict[property_name.split('_')[1]] = np.array([[np.nan]])
        else:
            graph_dict[property_name.split('_')[0]] = np.array([[np.nan]])
        graph_list.append(graph_dict)
    return graph_list
def read_graph_list(raw_dir, property_name=None, drop_property=False, process_labeled=False):
    print('raw_dir', raw_dir)
    if process_labeled:
        graph_list = labeled2graphs(raw_dir)
    else:
        graph_list = unlabel2graphs(raw_dir, property_name=property_name, drop_property=drop_property)
    pyg_graph_list = []
    print('Converting graphs into PyG objects...')
    for graph in graph_list:
        g = Data()
        g.__num_nodes__ = graph['num_nodes']
        g.edge_index = torch.from_numpy(graph['edge_index'])
        del graph['num_nodes']
        del graph['edge_index']
        if property_name.split('_')[-1] == 'all':
            g.y = torch.tensor([[-1.0]])

        elif len(property_name.split('_')) == 2:
            g.y = torch.from_numpy(graph['/'.join(property_name.split('_'))])
            g.y1 = torch.from_numpy(graph[property_name.split('_')[0]])
            g.y2 = torch.from_numpy(graph[property_name.split('_')[1]])
            if not np.isnan(np.sum(graph['/'.join(property_name.split('_'))])):
                x1, x2 = np.log(bd_dict_x[property_name][0]), np.log(bd_dict_x[property_name][1])
                y1, y2 = np.log(bd_dict_y[property_name][0]), np.log(bd_dict_y[property_name][1])
                a = (y1-y2)/(x1-x2)
                b = y1-a*x1
                cy = (np.log(graph['/'.join(property_name.split('_'))]) - (a * np.log(graph[property_name.split('_')[0]]) + b))>0
                cy = cy.astype(float)
                g.cy = torch.from_numpy(cy)
            else:
                g.cy = g.y1
        else:
            g.y = torch.from_numpy(graph[property_name.split('_')[0]])

        if graph['edge_feat'] is not None:
            g.edge_attr = torch.from_numpy(graph['edge_feat'])
            del graph['edge_feat']

        if graph['node_feat'] is not None:
            g.x = torch.from_numpy(graph['node_feat'])
            del graph['node_feat']

        addition_prop = copy.deepcopy(graph)
        for key in ['mgf', 'maccs']:
        # for key in addition_prop.keys():
            g[key] = torch.tensor(graph[key])
            del graph[key]

        pyg_graph_list.append(g)

    return pyg_graph_list

def make_balanced_testset(dataset_name, labels, max_size=150, seed=666, base = None):
    prop_name = dataset_name.split('-')[1]
    print('------ Spliting the dataset: ------')
    # else:
    labels = labels.numpy().reshape(-1)
    if prop_name in ['oxygen']:
        base = 10
        interval = 0.2
        max_size = 8
    elif prop_name in ['density']:
        interval = 0.02
    elif prop_name in ['melting']:
        interval = 10
    elif prop_name in ['glass']:
        interval = 5
    elif prop_name in ['molesol', 'energy']:
        interval = 0.1
    elif prop_name in ['molfreesolv']:
        interval = 0.2
    elif prop_name in ['mollipo']:
        interval = 0.05
    else:
        interval = 1
    if base != None:
        labels = log_base(base, labels)
    max_label_value = np.ceil(max(labels)) + interval
    min_label_value = np.floor(min(labels)) - interval / 2
    bins = np.arange(min_label_value, max_label_value, interval)
    inds = np.digitize(labels, bins)
    u_inds, counts = np.unique(inds, return_counts=True)
    max_size = int(max_size)
    selected_bins = u_inds
    print('--- Split info: ---')
    print('# bin / # valid bin: {} / {} == {}'.format(len(bins), len(u_inds), len(bins) / len(u_inds)))
    train_inds = []
    valid_inds = []
    test_inds = []
    np.random.seed(seed)
    for i in selected_bins:
        candidates_inds = (inds == i).nonzero()[0]
        each_sample_per_bin_val_test = min(len(candidates_inds) // 3, max_size)
        sample_reorder = np.arange(len(candidates_inds))
        np.random.shuffle(sample_reorder)
        test_inds.append(candidates_inds[sample_reorder][:each_sample_per_bin_val_test])
        valid_inds.append(candidates_inds[sample_reorder][each_sample_per_bin_val_test:each_sample_per_bin_val_test*2])
    test_inds = np.concatenate(test_inds)
    valid_inds = np.concatenate(valid_inds)
    train_inds = np.arange(len(labels))
    train_inds = np.setdiff1d(train_inds,test_inds)
    train_inds = np.setdiff1d(train_inds, valid_inds)
    test_num, valid_num = len(test_inds), len(valid_inds)
    train_num = len(labels) - test_num - valid_num

    train_subsampling = np.arange(len(train_inds))
    np.random.shuffle(train_subsampling)
    train_inds = train_inds[train_subsampling[:int(train_num)]]
    new_train_inds = []
    print('first five train sample: ', train_inds[:5])
    print('--- Split Result: ---')
    print('totoal len {} (== len label ? {}) '.format(len(valid_inds)+len(train_inds)+len(test_inds), len(labels)== (len(valid_inds)+len(train_inds)+len(test_inds))))
    print('train len {} / {} train max {} min {}'.format(len(train_inds), len(train_inds)/(test_num + valid_num + train_num), max(labels[train_inds]), min(labels[train_inds])))
    print('valid len {} / {} valid max {} min {}'.format(len(valid_inds), len(valid_inds)/(test_num + valid_num + train_num), max(labels[valid_inds]), min(labels[valid_inds])))
    print('test  len {} / {} test  max {} min {}'.format(len(test_inds), len(test_inds)/(test_num + valid_num + train_num), max(labels[test_inds]), min(labels[test_inds])))
    return train_inds, np.sort(valid_inds, axis=None), np.sort(test_inds, axis=None)

import math
def x_u_split(args, labels):
    label_per_class = args.num_labeled // args.num_classes
    labels = np.array(labels)
    labeled_idx = []
    # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    unlabeled_idx = np.array(range(len(labels)))
    for i in range(args.num_classes):
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class, False)
        labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)
    assert len(labeled_idx) == args.num_labeled

    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)
    return labeled_idx, unlabeled_idx