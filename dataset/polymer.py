import os
import torch
import logging
import pandas as pd
import os.path as osp
from torch_geometric.data import InMemoryDataset

from .data_utils import make_balanced_testset, read_graph_list
logger = logging.getLogger(__name__)

class PolymerRegDataset(InMemoryDataset):
    def __init__(self, name='plym-oxygen', root ='data', transform=None, pre_transform = None, use_valid_file_name='_validv4.csv'):
        '''
            - name (str): name of the dataset: plym-oxygen/melting/glass/density
            - root (str): root directory to store the dataset folder
            - transform, pre_transform (optional): transform/pre-transform graph objects
        ''' 
        self.name = name
        self.dir_name = '_'.join(name.split('-'))
        self.original_root = root
        self.root = osp.join(root, self.dir_name)
        self.processed_root = osp.join(osp.abspath(self.root))
        self.use_valid_file_name = use_valid_file_name

        self.num_tasks = 1
        self.eval_metric = 'rmse'
        self.task_type = 'regression'
        self.__num_classes__ = '-1'
        self.binary = 'False'

        super(PolymerRegDataset, self).__init__(self.processed_root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.total_data_len = len(self.data.y.view(-1))
        self.train_total = self.total_data_len - 4
        self.train_unlabeled = torch.isnan(self.data.y.view(-1)).sum().item()
        self.train_and_validv5 = self.train_total - self.train_unlabeled
        with open(osp.join(self.root, 'raw' ,self.name+'_validv5.csv'), 'r') as fp:
            num_lines = sum(1 for line in fp if line.rstrip())
        with open(osp.join(self.root, 'raw' ,self.name+self.use_valid_file_name), 'r') as fp:
            num_lines_valid = sum(1 for line in fp if line.rstrip())
        self.valid_num = num_lines_valid - 1
        self.train_labeled = self.train_and_validv5 - (num_lines-1)

        print('Name {}, # train label: {}, # valid {}, # train unlabeled: {}, # total: {}'.format(self.name, self.train_labeled, self.valid_num, self.train_unlabeled, self.total_data_len))
    def get_unlabeled_idx(self):
        return torch.arange(self.train_and_validv5, self.train_total, dtype=torch.long)
    def get_labeled_idx(self):
        return torch.arange(self.train_labeled, dtype=torch.long)
    def get_valid_idx(self):
        return torch.arange(self.train_labeled, self.train_labeled+self.valid_num, dtype=torch.long)
    def get_test_idx(self):
        return torch.arange(self.train_total, self.total_data_len, dtype=torch.long)


    @property
    def processed_file_names(self):
        return ['geometric_data_processed.pt']

    def process(self):
        data_list = read_graph_list(osp.join(self.root, 'raw' ,self.name+'_raw.csv'), property_name=self.name, process_labeled=True)
        print(data_list[:3])
        self.train_labeled = len(data_list)
        
        data_list.extend(read_graph_list(osp.join(self.root, 'raw' ,self.name+self.use_valid_file_name), property_name=self.name, process_labeled=True))
        self.train_and_valid = len(data_list)
        print('Labeled (+valid) Finished with length {} + {} '.format(self.train_labeled,self.train_and_valid-self.train_labeled))
        data_list.extend(read_graph_list(osp.join(self.original_root,'plym_all.csv'), property_name=self.name, drop_property=True, process_labeled=False))
        self.train_total = len(data_list)
        print('Label + Unlabeled length', self.train_total)
        data_list.extend(read_graph_list(osp.join(self.original_root,'test_raw.csv'), property_name=self.name, process_labeled=True))
        self.total_data_len = len(data_list)
        print('Label + Unlabeled + Test', self.total_data_len)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
