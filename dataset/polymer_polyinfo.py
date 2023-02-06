import os
import torch
import logging
import pandas as pd
import os.path as osp
from torch_geometric.data import InMemoryDataset

from .data_utils import make_balanced_testset, read_graph_list
logger = logging.getLogger(__name__)


class PolymerRegDatasetPolyInfo(InMemoryDataset):
    def __init__(self, name='plym-oxygen', root ='data', transform=None, pre_transform = None, use_valid_file_name='_validv5.csv'):
        '''
            - name (str): name of the dataset: plym-oxygen/melting/glass/density
            - root (str): root directory to store the dataset folder
            - transform, pre_transform (optional): transform/pre-transform graph objects
        ''' 
        self.name = 'polyinfo'
        self.path_abs = '/afs/crc.nd.edu/group/dmsquare/vol2/gliu7/developing/polymer_misc/two_polymers_design/raw_data'
        self.dir_name = '_'.join(self.name.split('-'))
        self.original_root = root
        self.root = osp.join(root, self.dir_name)
        self.processed_root = osp.join(osp.abspath(self.root))

        self.num_tasks = 1
        self.eval_metric = 'rmse'
        self.task_type = 'regression'
        self.__num_classes__ = '-1'
        self.binary = 'False'

        super(PolymerRegDatasetPolyInfo, self).__init__(self.processed_root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.total_data_len = len(self.data.y.view(-1))

        print('# example: {} '.format(self.total_data_len))
    def get_polyinfo_idx(self):
        return torch.arange(self.total_data_len, dtype=torch.long)

    @property
    def processed_file_names(self):
        return ['geometric_data_processed.pt']

    def process(self):
        data_list = read_graph_list(osp.join(self.path_abs,'polyinfo_raw.csv'), property_name=self.name, process_labeled=False)
        self.total_data_len = len(data_list)
        print('total_data_len', self.total_data_len)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
