import os
import torch
import logging
import pandas as pd
import os.path as osp
from torch_geometric.data import InMemoryDataset

from .data_utils import make_balanced_testset, read_graph_list
logger = logging.getLogger(__name__)


class PolymerRegDatasetValid(InMemoryDataset):
    def __init__(self, name='plym-oxygen', root ='data', transform=None, pre_transform = None, use_valid_file_name='_validv5.csv'):
        '''
            - name (str): name of the dataset: plym-oxygen/melting/glass/density
            - root (str): root directory to store the dataset folder
            - transform, pre_transform (optional): transform/pre-transform graph objects
        ''' 
        self.path_abs = '/afs/crc.nd.edu/group/dmsquare/vol2/gliu7/developing/polymer_misc/two_polymers_design/raw_data'
        self.name = name
        self.dir_name = '_'.join(name.split('-'))+use_valid_file_name[:-4]
        self.original_root = root
        self.root = osp.join(root, self.dir_name)
        self.processed_root = osp.join(osp.abspath(self.root))
        # self.use_valid_file_name = '_validv4.csv'
        self.use_valid_file_name = use_valid_file_name

        self.num_tasks = 1
        self.eval_metric = 'rmse'
        self.task_type = 'regression'
        self.__num_classes__ = '-1'
        self.binary = 'False'

        super(PolymerRegDatasetValid, self).__init__(self.processed_root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        with open(osp.join(self.path_abs, '_'.join(self.name.split('-')), 'raw' ,self.name+self.use_valid_file_name), 'r') as fp:
            num_lines_valid = sum(1 for line in fp if line.rstrip())
        self.valid_num = num_lines_valid-1 
        print(' # name {}, # valid {}'.format(self.name, self.valid_num))

    def get_valid_idx(self):
        return torch.arange(self.valid_num, dtype=torch.long)

    @property
    def processed_file_names(self):
        return ['geometric_data_processed.pt']

    def process(self):
        data_list = read_graph_list(osp.join(self.path_abs, '_'.join(self.name.split('-')), 'raw' ,self.name+self.use_valid_file_name), property_name=self.name, process_labeled=True)
        self.valid_num = len(data_list)
        print('Valid finished with length {}'.format(self.valid_num))

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
