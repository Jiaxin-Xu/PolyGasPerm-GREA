import math
import torch
import numpy as np
import logging
from torch.optim.lr_scheduler import LambdaLR


__all__ = [ 'get_cosine_schedule_with_warmup', 'seed_torch', 'get_logger']

# def print_info(set_name, perf):
#     output_str = '{}\t\t'.format(set_name)
#     for metric_name in ['mae', 'rmse', 'mse', 'gm']:
#         output_str += '{} all: {:<10.4f} \t'.format(metric_name, perf[metric_name]['all'])
#     print(output_str)

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            elif init_type == 'default':
                pass
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            torch.nn.init.normal_(m.weight.data, 1.0, init_gain)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def log_base(base, x):
    return np.log(x) / np.log(base)

class RegEvaluator(object):
    def __init__(self, dataset, labels, base=None, bin_width=None, medium_t=None, many_t=None):
        self.dataset = dataset
        self.labels = labels.numpy().reshape(-1)
        self.base = base
        self.interval = bin_width 
        self.bins = None
        self.train_label_freqs = None
        self.medium_t = medium_t
        self.many_t = many_t
        self._intialization()
    def get_region_masks(self, labels):
        assert self.train_label_freqs is not None, 'density missing'
        assert self.bins is not None, 'bins missing'
        region_bins = self.bins
        training_labels = self.labels
        if self.base != None:
            labels = log_base(self.base, labels)
            training_labels = log_base(self.base, training_labels)
        _, region_u_inds, bin_counts = self._assign_label_to_bins(training_labels, region_bins)
        region_freqs = np.zeros(region_bins.shape)
        region_freqs[region_u_inds] = bin_counts
        freqs4labels = np.zeros(labels.shape)
        in_train_region_mask = np.logical_and(labels>=min(region_bins), labels<max(region_bins))
        inds4labels = np.digitize(labels[in_train_region_mask], region_bins)
        freqs4labels[in_train_region_mask] = region_freqs[inds4labels]
        max_freq = max(region_freqs)
        many_mask = freqs4labels >= max_freq/self.many_t
        medium_mask = np.logical_and(freqs4labels>=max_freq/self.medium_t, freqs4labels<max_freq/self.many_t)
        few_mask = np.logical_and(freqs4labels>0, freqs4labels<max_freq/self.medium_t)
        zero_mask = freqs4labels == 0
        return many_mask, medium_mask, few_mask, zero_mask
    def _intialization(self):
        labels = self.labels
        if self.base != None:
            labels = log_base(self.base, labels)
        bins = self._get_bins()
        inds, u_inds, counts = self._assign_label_to_bins(labels, bins)
        freqs = np.zeros(bins.shape)
        freqs[u_inds] = counts
        self.train_label_freqs = freqs
        self.bins = bins
    def _get_bins(self):
        training_labels = self.labels
        if self.base is not None:
            training_labels = log_base(self.base, training_labels)
        interval = self.interval
        max_label_value = np.ceil(max(training_labels)) + interval
        min_label_value = np.floor(min(training_labels)) - interval / 2
        bins = np.arange(min_label_value, max_label_value, interval) #[start, end)
        return bins
    def _assign_label_to_bins(self, labels, bins):
        inds = np.digitize(labels, bins)
        u_inds, counts = np.unique(inds, return_counts=True)
        return inds, u_inds, counts


def get_logger(name, logfile=None):
    """ create a nice logger """
    logger = logging.getLogger(name)
    if (logger.hasHandlers()):
        logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if logfile is not None:
        fh = logging.FileHandler(logfile)
        fh.setFormatter(formatter)
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)
    logger.propagate = False
    return logger

def seed_torch(seed=0):
    print('Seed', seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)