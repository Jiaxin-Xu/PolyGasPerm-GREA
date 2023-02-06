import logging
import numpy as np
from scipy.ndimage import convolve1d
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang

logger = logging.getLogger(__name__)

class WeightEstimator(object):
    def __init__(self, dataset, labels, base=None, reweight=None, max_target=None, lds=False, lds_kernel='gaussian', lds_ks=5, lds_sigma=2, bin_width=None, medium_t=None):
        self.dataset = dataset
        self.labels = labels.numpy().reshape(-1)
        self.base = base
        self.interval = bin_width
        self.reweight = reweight
        self.max_target = max_target
        
        self.use_lds = lds
        self.lds_kernel = lds_kernel
        self.lds_ks = lds_ks
        self.lds_sigma = lds_sigma
        
        self.bins = None
        self.train_label_freqs = None
        self.smoothed_train_freqs = None
        if medium_t is None:
            self.medium_t = 10
        else:
            self.medium_t = medium_t
        self.scaling = self._intialize_prepare_weights()
    def get_weights(self, labels):
        labels = labels.reshape(-1)
        if self.scaling is None:
            return None  
        if self.base is not None:
            labels = log_base(self.base, labels)
        inds = np.digitize(labels, self.bins)
        if self.processed_freqs is not None:
            freqs = self.processed_freqs
        else:
            return None
        num_per_label = freqs[inds]
        weights = np.float32(1 / (num_per_label+1))
        weights = self.scaling * weights
        return weights
    def get_region_masks(self, labels):
        assert self.train_label_freqs is not None, 'density missing'
        assert self.bins is not None, 'bins missing'
        if self.dataset.split('-')[1] == 'glass':
            region_bins = self._get_bins(interval=5)
        elif self.dataset.split('-')[1] == 'energy':
            region_bins = self._get_bins(interval=0.1)
        else:
            region_bins = self.bins
        training_labels = self.labels
        if self.base != None:
            labels = log_base(self.base, labels)
            training_labels = log_base(self.base, training_labels)
        _, region_u_inds, bin_counts = self._assign_label_to_bins(training_labels, region_bins, pattern='auto')
        region_freqs = np.zeros(region_bins.shape)
        region_freqs[region_u_inds] = bin_counts
        freqs4labels = np.zeros(labels.shape)
        in_train_region_mask = np.logical_and(labels>=min(region_bins), labels<max(region_bins))
        inds4labels = np.digitize(labels[in_train_region_mask], region_bins)
        freqs4labels[in_train_region_mask] = region_freqs[inds4labels]
        max_freq = max(region_freqs)
        many_mask = freqs4labels >= max_freq/2
        medium_mask = np.logical_and(freqs4labels>=max_freq/self.medium_t, freqs4labels<max_freq/2)
        few_mask = np.logical_and(freqs4labels>0, freqs4labels<max_freq/self.medium_t)
        zero_mask = freqs4labels == 0
        return many_mask, medium_mask, few_mask, zero_mask
    def estimate_correlation(self, y_preds, y_true):
        import pandas as pd
        in_train_region_mask = np.logical_and(y_true>=min(self.bins), y_true<max(self.bins))
        group_ids = np.digitize(y_true[in_train_region_mask], self.bins)
        group_unique_ids, group_count = np.unique(group_ids, return_counts=True)
        freqs = np.zeros(self.bins.shape)
        freqs[group_unique_ids] = group_count
        if self.reweight is not None:
            if  self.reweight == 'sqrt_inv':
                freqs = np.sqrt(freqs)
            elif  self.reweight == 'inverse':
                freqs = freqs
            num_per_group = freqs[group_unique_ids]
        elif self.reweight is None:
            num_per_group = group_count
        if self.reweight is not None and self.use_lds:
            lds_kernel_window = get_lds_kernel_window(self.lds_kernel, self.lds_ks, self.lds_sigma)
            smoothed_freqs = convolve1d(freqs, weights=lds_kernel_window, mode='constant')
            num_per_group = smoothed_freqs[group_unique_ids]
        mae_loss = np.abs(y_preds - y_true)[in_train_region_mask]
        mean_by_groups = pd.DataFrame(mae_loss).groupby(group_ids).mean().values.reshape(-1) 
        return np.corrcoef(mean_by_groups, num_per_group)[0,1]
    def _intialize_prepare_weights(self):
        assert self.reweight in {None, 'inverse', 'sqrt_inv'}
        # assert self.reweight is not None if self.use_lds else True, \
        #     "Set reweight to \'sqrt_inv\' (default) or \'inverse\' when using LDS"
        interval = self.interval
        labels = self.labels
        if self.base != None:
            labels = log_base(self.base, labels)
        bins = self._get_bins(interval)
        inds, u_inds, counts = self._assign_label_to_bins(labels, bins, pattern='auto')
        freqs = np.zeros(bins.shape)
        freqs[u_inds] = counts
        self.train_label_freqs = freqs
        ''''used for double check the train set'''
        # import matplotlib.pyplot as plt
        # plt.bar(np.arange(len(freqs)), freqs)
        # print('imba', max(counts)/min(counts))
        # plt.savefig('{}_testsee.png'.format(prop_name),bbox_inches='tight')
        self.bins = bins
        if self.reweight is None:
            print('No reweighting')
            return None
        else:
            # print(f"Using re-weighting: [{ self.reweight}]")
            if  self.reweight == 'sqrt_inv':
                processed_freqs = np.sqrt(freqs)
            elif  self.reweight == 'inverse':
                processed_freqs = freqs
            num_per_label = processed_freqs[inds]
            self.processed_freqs = processed_freqs
        if self.reweight is not None and self.use_lds:
            lds_kernel_window = get_lds_kernel_window(self.lds_kernel, self.lds_ks, self.lds_sigma)
            # print(f'Using LDS: [{self.lds_kernel.upper()}] ({self.lds_ks}/{self.lds_sigma}) ({lds_kernel_window})')
            smoothed_freqs = convolve1d(processed_freqs, weights=lds_kernel_window, mode='constant')
            num_per_label = smoothed_freqs[inds]
            self.processed_freqs = smoothed_freqs
        weights = np.float32(1 / num_per_label)
        # return len(weights) / np.sum(weights)
        eps = 1e-8
        # print('scaling',np.sum(self.train_label_freqs[1:]) / np.sum((self.train_label_freqs[1:]+eps) / (processed_freqs[1:]+eps)))
        return len(weights) / np.sum(weights)
    def get_train_freqs(self, labels):
        labels = labels.reshape(-1)
        if self.base is not None:
            labels = log_base(self.base, labels)
        inds = np.digitize(labels, self.bins)
        assert self.train_label_freqs is not None, 'density missing'
        return self.train_label_freqs[inds]
    def get_smoothed_freqs(self, labels):
        labels = labels.reshape(-1)
        if self.base is not None:
            labels = log_base(self.base, labels)
        inds = np.digitize(labels, self.bins)
        assert self.smoothed_train_freqs is not None, 'smoothed density missing'
        return self.smoothed_train_freqs[inds]
    def _get_bins(self, interval):
        training_labels = self.labels
        if self.base is not None:
            training_labels = log_base(self.base, training_labels)
        interval = self.interval
        max_label_value = np.ceil(max(training_labels)) + interval
        min_label_value = np.floor(min(training_labels)) - interval / 2
        if self.max_target is not None:
            max_label_value = self.max_target
        bins = np.arange(min_label_value, max_label_value, interval) #[start, end)
        return bins
    def _assign_label_to_bins(self, labels, bins, pattern='auto'):
        # Return the indices of the bins to which each value in input array belongs: return satisfying i: bins[i-1] <= x < bins[i]
        inds = np.digitize(labels, bins)
        u_inds, counts = np.unique(inds, return_counts=True)
        return inds, u_inds, counts
        # return (len(weights) / np.sum(weights)) * weights
        # weights = [np.float32(1 / x) for x in num_per_label]
        # weights = [scaling * x for x in weights]
        # self.weights = weights
        # print(dict(zip(((bins[inds] + bins[inds-1]) / 2).astype(float), weights)))

def log_base(base, x):
    return np.log(x) / np.log(base)

def get_lds_kernel_window(kernel, ks, sigma):
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
    elif kernel == 'triang':
        kernel_window = triang(ks)
    else:
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(map(laplace, np.arange(-half_ks, half_ks + 1)))
    kernel_window = np.array(kernel_window)
    kernel_window = kernel_window / kernel_window.sum()
    return kernel_window




class RegEvaluator(object):
    def __init__(self, dataset, labels, base=None, reweight=None, max_target=None, lds=False, lds_kernel='gaussian', lds_ks=5, lds_sigma=2, bin_width=None, medium_t=None, many_t=None):
        self.dataset = dataset
        self.labels = labels.numpy().reshape(-1)
        self.base = base
        self.interval = bin_width
        self.reweight = reweight
        self.max_target = max_target
        
        self.use_lds = lds
        self.lds_kernel = lds_kernel
        self.lds_ks = lds_ks
        self.lds_sigma = lds_sigma
        
        self.bins = None
        self.train_label_freqs = None
        if medium_t is None:
            self.medium_t = 10
        else:
            self.medium_t = medium_t
        if many_t is None:
            self.many_t = 2
        else:
            self.many_t = many_t
        self._intialization()
    def get_region_masks(self, labels):
        assert self.train_label_freqs is not None, 'density missing'
        assert self.bins is not None, 'bins missing'
        # if self.dataset.split('-')[1] == 'glass':
        #     region_bins = self._get_bins()
        # elif self.dataset.split('-')[1] == 'energy':
        #     region_bins = self._get_bins()
        # else:
        region_bins = self.bins
        training_labels = self.labels
        if self.base != None:
            labels = log_base(self.base, labels)
            training_labels = log_base(self.base, training_labels)
        _, region_u_inds, bin_counts = self._assign_label_to_bins(training_labels, region_bins, pattern='auto')
        region_freqs = np.zeros(region_bins.shape)
        region_freqs[region_u_inds] = bin_counts
        freqs4labels = np.zeros(labels.shape)
        in_train_region_mask = np.logical_and(labels>=min(region_bins), labels<max(region_bins))
        inds4labels = np.digitize(labels[in_train_region_mask], region_bins)
        freqs4labels[in_train_region_mask] = region_freqs[inds4labels]
        max_freq = max(region_freqs)
        many_mask = freqs4labels >= max_freq/self.many_t
        medium_mask = np.logical_and(freqs4labels>=max_freq/self.medium_t, freqs4labels<max_freq/self.many_t)
        # many_mask = freqs4labels >= max_freq/2
        # medium_mask = np.logical_and(freqs4labels>=max_freq/self.medium_t, freqs4labels<max_freq/2)
        # import math
        # print('many >= {}, medium {} ~ {}, few <={}'.format(math.ceil(max_freq/self.many_t), math.ceil(max_freq/self.many_t), math.floor(max_freq/self.medium_t), math.floor(max_freq/self.medium_t)))
        few_mask = np.logical_and(freqs4labels>0, freqs4labels<max_freq/self.medium_t)
        zero_mask = freqs4labels == 0
        return many_mask, medium_mask, few_mask, zero_mask
    def estimate_correlation(self, y_preds, y_true):
        import pandas as pd
        in_train_region_mask = np.logical_and(y_true>=min(self.bins), y_true<max(self.bins))
        group_ids = np.digitize(y_true[in_train_region_mask], self.bins)
        group_unique_ids, group_count = np.unique(group_ids, return_counts=True)
        freqs = np.zeros(self.bins.shape)
        freqs[group_unique_ids] = group_count
        if self.reweight is not None:
            if  self.reweight == 'sqrt_inv':
                freqs = np.sqrt(freqs)
            elif  self.reweight == 'inverse':
                freqs = freqs
            num_per_group = freqs[group_unique_ids]
        elif self.reweight is None:
            num_per_group = group_count
        if self.reweight is not None and self.use_lds:
            lds_kernel_window = get_lds_kernel_window(self.lds_kernel, self.lds_ks, self.lds_sigma)
            smoothed_freqs = convolve1d(freqs, weights=lds_kernel_window, mode='constant')
            num_per_group = smoothed_freqs[group_unique_ids]
        mae_loss = np.abs(y_preds - y_true)[in_train_region_mask]
        mean_by_groups = pd.DataFrame(mae_loss).groupby(group_ids).mean().values.reshape(-1) 
        return np.corrcoef(mean_by_groups, num_per_group)[0,1]
    def _intialization(self):
        prop_name = self.dataset.split('-')[1]
        if self.interval is None:
            if prop_name in ['oxygen']:
                interval = 1
            elif prop_name in ['density']:
                interval = 0.02
            elif prop_name in ['melting']:
                interval = 10
            elif prop_name in ['glass']:
                interval = 5
            elif prop_name in ['molesol','energy']:
                interval = 0.1
            elif prop_name in ['molfreesolv']:
                interval = 0.2
            elif prop_name in ['mollipo']:
                interval = 0.05
            else:
                interval = 1
            self.interval = interval
        # else:
        #     interval = self.interval
        labels = self.labels
        if self.base != None:
            labels = log_base(self.base, labels)
        bins = self._get_bins()
        inds, u_inds, counts = self._assign_label_to_bins(labels, bins, pattern='auto')
        freqs = np.zeros(bins.shape)
        freqs[u_inds] = counts
        self.train_label_freqs = freqs
        self.bins = bins
    
    def get_stat(self, labels):
        labels = labels.numpy().reshape(-1)
        many_mask, medium_mask, few_mask, zero_mask = self.get_region_masks(labels)
        print('Many count {}, medium count {}, few count {}, zero count {}'.format(sum(many_mask), sum(medium_mask), sum(few_mask), sum(zero_mask)))

    def get_train_freqs(self, labels):
        labels = labels.reshape(-1)
        if self.base is not None:
            labels = log_base(self.base, labels)
        inds = np.digitize(labels, self.bins)
        assert self.train_label_freqs is not None, 'density missing'
        return self.train_label_freqs[inds]
    def _get_bins(self):
        training_labels = self.labels
        if self.base is not None:
            training_labels = log_base(self.base, training_labels)
        interval = self.interval
        max_label_value = np.ceil(max(training_labels)) + interval
        min_label_value = np.floor(min(training_labels)) - interval / 2
        if self.max_target is not None:
            max_label_value = self.max_target
        bins = np.arange(min_label_value, max_label_value, interval) #[start, end)
        return bins
    def _assign_label_to_bins(self, labels, bins, pattern='auto'):
        # Return the indices of the bins to which each value in input array belongs: return satisfying i: bins[i-1] <= x < bins[i]
        inds = np.digitize(labels, bins)
        u_inds, counts = np.unique(inds, return_counts=True)
        return inds, u_inds, counts
