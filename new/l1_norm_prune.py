import torch
import torch.nn as nn
import numpy as np
import vgg

def prune(model, args):
    # cfg = vgg.defaultcfg[args.depth]
    cfg = [32, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 256, 256, 256, 'M', 256, 256, 256]

    cfg_mask = []
    layer_id = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            out_channels = m.weight.data.shape[0]
            if out_channels == cfg[layer_id]:
                cfg_mask.append(torch.ones(out_channels))
                layer_id += 1
                continue
            weight_copy = m.weight.data.abs().clone()
            weight_copy = weight_copy.cpu().numpy()
            L1_norm = np.sum(weight_copy, axis=(1, 2, 3))
            arg_max = np.argsort(L1_norm)
            arg_max_rev = arg_max[::-1][:cfg[layer_id]]
            assert arg_max_rev.size == cfg[layer_id], "size of arg_max_rev not correct"
            mask = torch.zeros(out_channels)
            mask[arg_max_rev.tolist()] = 1
            cfg_mask.append(mask)
            layer_id += 1
        elif isinstance(m, nn.MaxPool2d):
            layer_id += 1
    
    return cfg, cfg_mask


def match_model_weights(model, newmodel, cfg_mask, weight_model=None):
    start_mask = torch.ones(3)
    layer_id_in_cfg = 0
    end_mask = cfg_mask[layer_id_in_cfg]

    if weight_model is None:
        weight_model = model
    
    for [m0, m1, m_weight] in zip(model.modules(), newmodel.modules(), weight_model.modules()):
        if isinstance(m0, nn.BatchNorm2d):
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            if idx1.size == 1:
                idx1 = np.resize(idx1,(1,))
            # Use weights from weight_model instead
            m1.weight.data = m_weight.weight.data[idx1.tolist()].clone()
            m1.bias.data = m_weight.bias.data[idx1.tolist()].clone()
            m1.running_mean = m_weight.running_mean[idx1.tolist()].clone()
            m1.running_var = m_weight.running_var[idx1.tolist()].clone()
            layer_id_in_cfg += 1
            start_mask = end_mask
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
            # Use weights from weight_model instead
            w1 = m_weight.weight.data[:, idx0.tolist(), :, :].clone()
            w1 = w1[idx1.tolist(), :, :, :].clone()
            m1.weight.data = w1.clone()
        elif isinstance(m0, nn.Linear):
            if layer_id_in_cfg == len(cfg_mask):
                idx0 = np.squeeze(np.argwhere(np.asarray(cfg_mask[-1].cpu().numpy())))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                # Use weights from weight_model instead
                m1.weight.data = m_weight.weight.data[:, idx0].clone()
                m1.bias.data = m_weight.bias.data.clone()
                layer_id_in_cfg += 1
                continue
            # Use weights from weight_model instead
            m1.weight.data = m_weight.weight.data.clone()
            m1.bias.data = m_weight.bias.data.clone()
        elif isinstance(m0, nn.BatchNorm1d):
            # Use weights from weight_model instead
            m1.weight.data = m_weight.weight.data.clone()
            m1.bias.data = m_weight.bias.data.clone()
            m1.running_mean = m_weight.running_mean.clone()
            m1.running_var = m_weight.running_var.clone()
    
    return newmodel