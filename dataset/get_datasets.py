from .polymer import PolymerRegDataset
from .polymer_allset import PolymerRegDataset_AllSet
from .polymer_onlytest import PolymerRegDataset_OnlyTest
from .polymer_onlyvalid import PolymerRegDatasetValid
from .polymer_polyinfo import PolymerRegDatasetPolyInfo
from .ogbg import PygGraphPropPredDataset

DATASET_GETTERS = {
    'plym-oxygen': PolymerRegDataset,
    'plym-density': PolymerRegDataset,
    'plym-melting': PolymerRegDataset,
    'ogbg-molesol': PygGraphPropPredDataset,
    'ogbg-molfreesolv': PygGraphPropPredDataset,
    'ogbg-mollipo': PygGraphPropPredDataset}
def get_dataset(args, load_path, valid_v='v5'):
    return PolymerRegDataset(args.dataset, load_path, use_valid_file_name='_valid{}.csv'.format(valid_v))
def get_dataset_allset(dataset, load_path, valid_v='v5'):
    return PolymerRegDataset_AllSet(dataset, load_path, use_valid_file_name='_valid{}.csv'.format(valid_v))
def get_dataset_onlytest(args, load_path, valid_v='v5'):
    return PolymerRegDataset_OnlyTest(args.dataset, load_path, use_valid_file_name='_valid{}.csv'.format(valid_v))
def get_dataset_validonly(args, load_path, valid_v='v5'):
    return PolymerRegDatasetValid(args.dataset, load_path, use_valid_file_name='_valid{}.csv'.format(valid_v))

def get_dataset_polyinfo(args, load_path, valid_v='v5'):
    return PolymerRegDatasetPolyInfo(args.dataset, load_path, use_valid_file_name='_valid{}.csv'.format(valid_v))


