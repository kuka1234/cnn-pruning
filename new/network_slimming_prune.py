import torch
import torch.nn as nn
import numpy as np

def compute_mask_threshold(model, args):
    """Compute mask threshold based on percent of channels to prune"""
    total = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            total += m.weight.data.shape[0]

    bn = torch.zeros(total)
    index = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            size = m.weight.data.shape[0]
            bn[index:(index + size)] = m.weight.data.abs().clone()
            index += size

    y, i = torch.sort(bn)
    thre_index = int(total * args.percent)
    thre = y[thre_index]

    return thre, total


def prune_model(model, threshold, total):
    """Prune the model based on the threshold"""
    pruned = 0
    cfg = []
    cfg_mask = []

    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.BatchNorm2d):
            weight_copy = m.weight.data.abs().clone()
            mask = weight_copy.gt(threshold).float().cuda()
            pruned = pruned + mask.shape[0] - torch.sum(mask)
            m.weight.data.mul_(mask)
            m.bias.data.mul_(mask)
            cfg.append(int(torch.sum(mask)))
            cfg_mask.append(mask.clone())
            print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                  format(k, mask.shape[0], int(torch.sum(mask))))
        elif isinstance(m, nn.MaxPool2d):
            cfg.append('M')

    pruned_ratio = pruned / total
    print(f'Pruned ratio: {pruned_ratio:.2f}')
    print('Pre-processing Successful!')

    return cfg, cfg_mask

def prune(model, args):
    # Compute threshold for pruning
    threshold, total_channels = compute_mask_threshold(model, args)

    # Prune the model
    cfg, cfg_mask = prune_model(model, threshold, total_channels)
    return cfg, cfg_mask


def match_model_weights(model, newmodel, cfg_mask, weight_model=None):
    layer_id_in_cfg = 0
    start_mask = torch.ones(3)
    end_mask = cfg_mask[layer_id_in_cfg]

    if weight_model is None:
        weight_model = model

    for [m0, m1, m_weight] in zip(model.modules(), newmodel.modules(), weight_model.modules()):
        if isinstance(m0, nn.BatchNorm2d):
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            # Use weights from weight_model instead of model
            m1.weight.data = m_weight.weight.data[idx1.tolist()].clone()
            m1.bias.data = m_weight.bias.data[idx1.tolist()].clone()
            m1.running_mean = m_weight.running_mean[idx1.tolist()].clone()
            m1.running_var = m_weight.running_var[idx1.tolist()].clone()
            layer_id_in_cfg += 1
            start_mask = end_mask.clone()
            if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                end_mask = cfg_mask[layer_id_in_cfg]
        elif isinstance(m0, nn.Conv2d):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            # Use weights from weight_model instead of model
            w1 = m_weight.weight.data[:, idx0.tolist(), :, :].clone()
            w1 = w1[idx1.tolist(), :, :, :].clone()
            m1.weight.data = w1.clone()
        elif isinstance(m0, nn.Linear):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            # Use weights from weight_model instead of model
            m1.weight.data = m_weight.weight.data[:, idx0].clone()
            m1.bias.data = m_weight.bias.data.clone()
    
    return newmodel