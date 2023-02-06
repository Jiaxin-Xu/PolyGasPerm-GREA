import argparse
import math
import os
import shutil
from datetime import datetime
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader

from models.grea import GraphEnvAug
from dataset.get_datasets import get_dataset
from utils import validate, testing, get_cosine_schedule_with_warmup, WeightEstimator
from utils import build_augment_dataset, build_selection_dataset
from utils import train_regular

def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    print('Saved checkpoint to', filepath)
def seed_torch(seed=0):
    print('Seed', seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


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


def get_args():
    # Training settings
    parser = argparse.ArgumentParser(description='Graph ML Models for polymer predictions')
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')
    parser.add_argument('--dataset', default="CO2", type=str, choices=['CO2', 'H2', 'O2','CH4','N2'], help='gas permeability dataset') 
    parser.add_argument('--drop_ratio', type=float, default=0.5, help='dropout ratio (default: 0.5)')
    parser.add_argument('--lr', '--learning-rate', type=float, default=1e-3, help='Learning rate (default: 1e-3)')
    parser.add_argument('--num_layer', type=int, default=5, help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=600, help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=600, help='number of epochs to train (default: 600)')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers (default: 0)')
    parser.add_argument('--add_fp', type=str, default='None',choices=['None','ECFP','MACCS','onlyECFP','onlyMACCS','ECFP_MACCS'])
    parser.add_argument('--out', type=str, default="", help='folder to output result (default: )')
    # Printing control
    parser.add_argument('--no_print', action='store_true', default=False)
    parser.add_argument('--pred_log', action='store_true', default=True)
    args = parser.parse_args()
    args.w_ratio = 1
    return args


def main(args):
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    dataset = get_dataset(args, "./raw_data",valid_v='v4')

    args.num_unlabeled = dataset.train_unlabeled
    args.num_labeled = dataset.train_labeled
    args.num_trained = dataset.train_total
    test_loader = DataLoader(dataset[dataset.get_test_idx()], batch_size=args.batch_size, shuffle=False, num_workers = 0)
    labeled_trainloader = DataLoader(dataset[dataset.get_labeled_idx()], batch_size=args.batch_size, shuffle=True, num_workers = 0)
    valid_loader = DataLoader(dataset[dataset.get_valid_idx()], batch_size=args.batch_size, shuffle=False, num_workers = 0)
    weight_estimator = WeightEstimator(args.dataset, dataset.data.y[dataset.get_labeled_idx()], base=10,reweight='sqrt_inv', max_target=None, lds=True,lds_kernel='gaussian',  lds_ks=3,lds_sigma=2, bin_width=0.1)
    
    add_fp_list  = args.add_fp.split('_')
    model = GraphEnvAug(gnn_type = 'gin-virtual', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, gamma=0.4, add_fp=add_fp_list).to(device)

    unlabeled_set = dataset[dataset.get_unlabeled_idx()]
    unlabeled_trainloader = DataLoader(unlabeled_set, batch_size= args.batch_size, shuffle=True, num_workers = args.num_workers)

    optimizer = optim.Adam(list(model.separator.parameters())+list(model.graph_encoder.parameters())+list(model.predictor.parameters()), lr=args.lr, weight_decay=1e-3) # 1e-5    
    
    if args.out != '':
        testp = args.out+'/{}_{}_{}_U{}_0'.format('grea', args.dataset, args.add_fp, 'T')
        os.makedirs(testp, exist_ok=True)
        args.model_save_path = testp
    else:
        args.model_save_path = False
    if args.model_save_path:
        os.makedirs(args.model_save_path, exist_ok=True)

    train_loaders = {'labeled_iter': iter(labeled_trainloader),'labeled_trainloader': labeled_trainloader,}
    args.steps = args.num_labeled // args.batch_size + 1

    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, args.steps*args.epochs)
    train_loaders['unlabeled_trainloader'] = unlabeled_trainloader
    train_loaders['unlabeled_iter'] = iter(unlabeled_trainloader)
    train_loaders['augmented_reps'] = None
    train_loaders['augmented_labels'] = None

    for epoch in range(0, args.epochs):
        train_loaders = train_regular(args, model, train_loaders, optimizer, scheduler, epoch, weight_estimator)
        if epoch >= 50 and epoch % 30 == 0:
            new_trainloader = build_selection_dataset(args, model, dataset)
            train_loaders['labeled_trainloader'] = new_trainloader
            train_loaders['labeled_iter'] = iter(new_trainloader)
            args.steps = len(new_trainloader.dataset) // args.batch_size + 1
            augmented_reps, augmented_labels = build_augment_dataset(args, model, dataset, including_new=new_trainloader)
            train_loaders['augmented_reps'] = augmented_reps
            train_loaders['augmented_labels'] = augmented_labels
        train_perf = validate(args, model, labeled_trainloader)
        valid_perf = validate(args, model, valid_loader)

        update_test = False
        if epoch == 0:
            update_test = True
        elif valid_perf['mae']['all'] < best_valid_perf['mae']['all']:
            update_test = True
        if update_test:
            best_valid_perf = valid_perf
            best_train_perf = train_perf
            cnt_wait = 0
            best_epoch = epoch
            BPDANP_test, BPDAATC_test = testing(args, model, test_loader)

            if args.model_save_path and epoch >= 0:
                model_to_save = model.module if hasattr(model, "module") else model
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model_to_save.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }, update_test, args.model_save_path, filename='model_best_{}.pth.tar'.format(args.seed))
        else:
            # not update
            if not args.no_print:
                print('train:', {key: '{:<10.4f}'.format(best_train_perf['mae'][key]) for key in best_train_perf['mae'] if key == 'all'})
                print('valid:', {key: '{:<10.4f}'.format(best_valid_perf['mae'][key]) for key in best_valid_perf['mae'] if key == 'all'})
            if epoch > 100: 
                cnt_wait += 1
                if cnt_wait > 50:
                    break

    print('Finished training! Best validation results from epoch {}.'.format(best_epoch))
    print('train:', {key: '{:<10.4f}'.format(best_train_perf['mae'][key]) for key in best_train_perf['mae'] if key != 'Metric'})
    print('valid:', {key: '{:<10.4f}'.format(best_valid_perf['mae'][key]) for key in best_valid_perf['mae'] if key != 'Metric'})
    print('\t\tBPDANP: {:<5.3f} (Train: {:.3f}, Valid: {:.3f}). \t True-Pred={:.3f}-{:.3f}. \t {}<{}<{}'.format(BPDANP_test['error'], best_train_perf['mae']['{}_to_{}'.format(BPDANP_test['lower'], BPDANP_test['upper'])], best_valid_perf['mae']['{}_to_{}'.format(BPDANP_test['lower'], BPDANP_test['upper'])], BPDANP_test['true'], BPDANP_test['pred'], BPDANP_test['lower'], BPDANP_test['true'], BPDANP_test['upper'], ))
    print('\t\tBPDAATC: {:<5.3f} (Train: {:.3f}, Valid: {:.3f}). \t True-Pred={:.3f}-{:.3f}. \t {}<{}<{}'.format(BPDAATC_test['error'], best_train_perf['mae']['{}_to_{}'.format(BPDAATC_test['lower'], BPDAATC_test['upper'])], best_valid_perf['mae']['{}_to_{}'.format(BPDAATC_test['lower'], BPDAATC_test['upper'])], BPDAATC_test['true'], BPDAATC_test['pred'], BPDAATC_test['lower'], BPDAATC_test['true'], BPDAATC_test['upper']))


if __name__ == "__main__":
    args = get_args()
    for i in range(10):
        args.seed = i
        seed_torch(seed=args.seed)
        main(args)
