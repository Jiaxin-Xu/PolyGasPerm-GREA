from functools import reduce
from termios import OFDEL
import time
from tqdm import tqdm
import torch
import math
import random
import numpy as np
import torch.nn.functional as F
from scipy.stats import gmean
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, recall_score, accuracy_score

def weighted_mse_loss(inputs, targets, weights=None):
    loss = (inputs - targets) ** 2
    if weights is not None:
        loss *= weights.view(loss.size())
    return loss

def weighted_l1_loss(inputs, targets, weights=None):
    loss = F.l1_loss(inputs, targets, reduction='none')
    if weights is not None:
        loss *= weights.view(loss.size())
    return loss

def weighted_focal_mse_loss(inputs, targets, activate='sigmoid', beta=.2, gamma=1, weights=None):
    loss = (inputs - targets) ** 2
    loss *= (torch.tanh(beta * torch.abs(inputs - targets))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta * torch.abs(inputs - targets)) - 1) ** gamma
    if weights is not None:
        loss *= weights.view(loss.size())
    return loss


def weighted_focal_l1_loss(inputs, targets, activate='sigmoid', beta=.2, gamma=1, weights=None):
    loss = F.l1_loss(inputs, targets, reduction='none')
    loss *= (torch.tanh(beta * torch.abs(inputs - targets))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta * torch.abs(inputs - targets)) - 1) ** gamma
    if weights is not None:
        loss *= weights.view(loss.size())
    loss = torch.mean(loss)
    return loss


def weighted_huber_loss(inputs, targets, beta=1., weights=None):
    l1_loss = torch.abs(inputs - targets)
    cond = l1_loss < beta
    loss = torch.where(cond, 0.5 * l1_loss ** 2 / beta, l1_loss - 0.5 * beta)
    if weights is not None:
        loss *= weights.view(loss.size())
    return loss

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_regular(args, model, train_loaders, optimizer, scheduler, epoch, weight_estimator):
    reg_criterion = weighted_l1_loss
    torch_softmax = torch.nn.Softmax(dim=0)

    if not args.no_print:
        p_bar = tqdm(range(args.steps))
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_xaug = AverageMeter()
    device = args.device
    model.train()
    if train_loaders['augmented_reps'] is not None and train_loaders['augmented_labels'] is not None:
        aug_reps = train_loaders['augmented_reps']
        aug_targets = train_loaders['augmented_labels']
        random_inds = torch.randperm(aug_reps.size(0))
        aug_reps = aug_reps[random_inds]
        aug_targets = aug_targets[random_inds]
        aug_batch_size = aug_reps.size(0) // args.steps
        aug_inputs = list(torch.split(aug_reps, aug_batch_size))
        aug_outputs = list(torch.split(aug_targets, aug_batch_size))
    else:
        aug_inputs = None
        aug_outputs = None
    for batch_idx in range(args.steps):
        end = time.time()
        model.zero_grad()  
        ### augmentation loss
        if aug_inputs is not None and aug_outputs is not None and aug_inputs[batch_idx].size(0) != 1:
            model._disable_batchnorm_tracking(model)
            pred_aug = model.predictor(aug_inputs[batch_idx])
            model._enable_batchnorm_tracking(model)
            targets_aug = aug_outputs[batch_idx]
            if args.pred_log:
                targets_aug = torch.log10(targets_aug)
            Laug = reg_criterion(pred_aug.view(targets_aug.size()).to(torch.float32), targets_aug, weights=None)
            Laug = Laug.mean()
        else:
            Laug = torch.tensor(0.)  
        ### supervision loss
        try:
            batch_labeled = train_loaders['labeled_iter'].next()
        except:
            train_loaders['labeled_iter'] = iter(train_loaders['labeled_trainloader'])
            batch_labeled = train_loaders['labeled_iter'].next()
        batch_labeled =  batch_labeled.to(device)
        targets = batch_labeled.y.to(torch.float32)
        if args.pred_log:
            targets = torch.log10(batch_labeled.y.to(torch.float32))
        if batch_labeled.x.shape[0] != 1 and batch_labeled.batch[-1] != 0:   
            output = model(batch_labeled)
            pred_labeled, pred_rep = output['pred_rem'], output['pred_rep']

            if weight_estimator.reweight is not None:
                weights = weight_estimator.get_weights(batch_labeled.y.cpu().numpy()) # todo: torch version
                weights = torch.FloatTensor(weights).to(targets.device)
            else:
                weights = None
            Losses_x = reg_criterion(pred_labeled.view(targets.size()).to(torch.float32), targets, weights=weights)
            Lx = Losses_x.mean()
            Lx += output['loss_reg']
            target_rep = targets.repeat_interleave(batch_labeled.batch[-1]+1,dim=0)
            losses_xrep_envs = reg_criterion(pred_rep.view(target_rep.size()).to(torch.float32), target_rep)
            losses_xrep_envs = losses_xrep_envs.view(-1).view(-1,batch_labeled.batch[-1]+1)
            losses_xrep_var, losses_xrep_mean = losses_xrep_envs.var(dim=1), losses_xrep_envs.mean(dim=1)
            Lx += losses_xrep_mean.mean()
        else:
            Lx = torch.tensor(0.)
        loss = Lx + Laug * 1
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.update(loss.item())
        losses_x.update(Lx.item())
        losses_xaug.update(Laug.item())
        batch_time.update(time.time() - end)
        end = time.time()
        if not args.no_print:
            p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_xaug: {losses_xaug:.4f}. ".format(
                epoch=epoch + 1,
                epochs=args.epochs,
                batch=batch_idx + 1,
                iter=args.steps,
                lr=scheduler.get_last_lr()[0],
                data=data_time.avg,
                bt=batch_time.avg,
                loss=losses.avg,
                loss_x=losses_x.avg,
                losses_xaug=losses_xaug.avg,
                ))
            p_bar.update()
    if not args.no_print:
        p_bar.close()
    return train_loaders


def validate(args, model, loader, cls=False):
    y_true = []
    y_pred = []

    if cls:
        acc_dict = {'Metric': 'Accuracy'}
        recall_dict = {'Metric': 'Recall'}
    else:
        rmse_dict = {'Metric': 'RMSE'}
        mse_dict = {'Metric': 'MSE'}
        mae_dict = {'Metric': 'MAE'}
    device = args.device
    model.eval()
    for step, batch in enumerate(loader):
        batch = batch.to(device)
        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                output = model(batch, cls=cls)
                pred = output['pred_rem']
            if cls:
                y_true.append(batch.cy.view(pred.shape).detach().cpu())
                y_pred.append(torch.sigmoid(pred).detach().cpu())
            else:
                y_true.append(batch.y.view(pred.shape).detach().cpu())
                if args.pred_log:
                    pred = torch.clamp(pred, min=None, max=10)
                    y_pred.append(10**pred.detach().cpu())
                else:
                    y_pred.append(pred.detach().cpu())
    y_true = torch.cat(y_true, dim = 0).numpy().reshape(-1)
    y_pred = torch.cat(y_pred, dim = 0).numpy().reshape(-1)
    if cls:
        recall_dict['all'] = recall_score(y_true, y_pred>0.5, average='binary')
        acc_dict['all'] = accuracy_score(y_true, y_pred>0.5)
        perf ={
            'recall': recall_dict,
            'accuracy': acc_dict,
        }
    else:
        mse_dict['all'] = mean_squared_error(y_true, y_pred, squared=True)
        mae_dict['all'] = mean_absolute_error(y_true, y_pred)
        log_range_masks = get_log_range_masks(y_true)
        for key, mask in log_range_masks.items():
            if mask.sum() == 0:
                mse_dict[key] = -1
                mae_dict[key] = -1
            else:        
                mse_dict[key] = mean_squared_error(y_true[mask], y_pred[mask], squared=True)
                mae_dict[key] = mean_absolute_error(y_true[mask], y_pred[mask])
        perf ={
            'mse': mse_dict,
            'mae': mae_dict,
        }
    return perf

def testing(args, model, loader, cls=False):
    y_true = []
    y_pred = []
    device = args.device
    model.eval()
    for step, batch in enumerate(loader):
        batch = batch.to(device)
        if batch.x.shape[0] != 1:
            with torch.no_grad():
                output = model(batch, cls=cls)
                pred = output['pred_rem']
            y_true.append(batch.y.view(pred.shape).detach().cpu())
            if args.pred_log:
                pred = torch.clamp(pred, min=None, max=10)
                y_pred.append(10**pred.detach().cpu())
            else:
                y_pred.append(pred.detach().cpu())
    BPDANP_res = {}
    BPDAATC_res = {}
    BPDANP_res['true'] = torch.cat(y_true, dim = 0).numpy().reshape(-1)[0]
    BPDANP_res['pred'] = torch.cat(y_pred, dim = 0).numpy().reshape(-1)[0]
    BPDANP_res['error'] = BPDANP_res['true'] - BPDANP_res['pred']

    BPDAATC_res['true'] = torch.cat(y_true, dim = 0).numpy().reshape(-1)[1]
    BPDAATC_res['pred'] = torch.cat(y_pred, dim = 0).numpy().reshape(-1)[1]
    BPDAATC_res['error'] = BPDAATC_res['true'] - BPDAATC_res['pred']


    if not cls:
        boundaries = np.array([0, 0.1, 1, 10, 100, 1000, 10000, 100000000])
        boundaries_list = [0, 0.1, 1, 10, 100, 1000, 10000, 100000000]
        BPDANP_upper_idx = (boundaries>BPDANP_res['true']).nonzero()[0].min()
        BPDANP_lower_idx = (boundaries<BPDANP_res['true']).nonzero()[0].max()
        BPDANP_res['lower'], BPDANP_res['upper'] = boundaries_list[BPDANP_lower_idx], boundaries_list[BPDANP_upper_idx]

        BPDAATC_upper_idx = (boundaries>BPDAATC_res['true']).nonzero()[0].min()
        BPDAATC_lower_idx = (boundaries<BPDAATC_res['true']).nonzero()[0].max()
        BPDAATC_res['lower'], BPDAATC_res['upper'] = boundaries_list[BPDAATC_lower_idx], boundaries_list[BPDAATC_upper_idx]
    return BPDANP_res, BPDAATC_res

def get_log_range_masks(y_true):
    y_true = y_true.reshape(-1)
    log_range_masks = {}
    upper_bounds = [0.1, 1, 10, 100, 1000, 10000, 100000000]
    for i, lower_bound in enumerate([0, 0.1, 1, 10, 100, 1000, 10000]):
        upper_bound = upper_bounds[i]
        dict_idx =  '{}_to_{}'.format(lower_bound, upper_bound)
        mask = np.logical_and(y_true >= lower_bound, y_true < upper_bound)
        log_range_masks[dict_idx] = mask
    return log_range_masks



# Copyright (c) 2021-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

def batchwise_ranking_regularizer(features, targets, lambda_val):
    loss = 0

    # Reduce ties and boost relative representation of infrequent labels by computing the 
    # regularizer over a subset of the batch in which each label appears at most once
    batch_unique_targets = torch.unique(targets)
    if len(batch_unique_targets) < len(targets):
        sampled_indices = []
        for target in batch_unique_targets:
            sampled_indices.append(random.choice((targets == target).nonzero()[:,0]).item())
        x = features[sampled_indices]
        y = targets[sampled_indices]
    else:
        x = features
        y = targets

    # Compute feature similarities
    xxt = torch.matmul(F.normalize(x.view(x.size(0),-1)), F.normalize(x.view(x.size(0),-1)).permute(1,0))

    # Compute ranking similarity loss
    for i in range(len(y)):
        label_ranks = rank_normalised(-torch.abs(y[i] - y).transpose(0,1))
        feature_ranks = TrueRanker.apply(xxt[i].unsqueeze(dim=0), lambda_val)
        loss += F.mse_loss(feature_ranks, label_ranks)
    
    return loss

# Copyright (c) 2021-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#######################################################################################################################
# Code is based on the Blackbox Combinatorial Solvers (https://github.com/martius-lab/blackbox-backprop) implementation
# from https://github.com/martius-lab/blackbox-backprop by Marin Vlastelica et al.
#######################################################################################################################

def rank(seq):
    return torch.argsort(torch.argsort(seq).flip(1))


def rank_normalised(seq):
    return (rank(seq) + 1).float() / seq.size()[1]


class TrueRanker(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sequence, lambda_val):
        rank = rank_normalised(sequence)
        ctx.lambda_val = lambda_val
        ctx.save_for_backward(sequence, rank)
        return rank

    @staticmethod
    def backward(ctx, grad_output):
        sequence, rank = ctx.saved_tensors
        assert grad_output.shape == rank.shape
        sequence_prime = sequence + ctx.lambda_val * grad_output
        rank_prime = rank_normalised(sequence_prime)
        gradient = -(rank - rank_prime) / (ctx.lambda_val + 1e-8)
        return gradient, None
