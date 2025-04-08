import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from vgg import vgg


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
    parser.add_argument('--dataset', type=str, default='cifar100',
                        help='training dataset (default: cifar10)')
    parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                        help='input batch size for testing (default: 256)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--depth', type=int, default=19,
                        help='depth of the vgg')
    parser.add_argument('--percent', type=float, default=0.5,
                        help='scale sparse rate (default: 0.5)')
    parser.add_argument('--model', default='', type=str, metavar='PATH',
                        help='path to the model (default: none)')
    parser.add_argument('--save', default='', type=str, metavar='PATH',
                        help='path to save pruned model (default: none)')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args


def load_model(args):
    """Load model and checkpoint if provided"""
    model = vgg(dataset=args.dataset, depth=args.depth)
    if args.cuda:
        model.cuda()

    if args.model:
        if os.path.isfile(args.model):
            print("=> loading checkpoint '{}'".format(args.model))
            checkpoint = torch.load(args.model)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
                  .format(args.model, checkpoint['epoch'], best_prec1))
        else:
            print("=> no checkpoint found at '{}'".format(args.model))

    return model


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


def get_test_loader(args):
    """Get test data loader based on dataset"""
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    if args.dataset == 'cifar10':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    elif args.dataset == 'cifar100':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./data.cifar100', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    else:
        raise ValueError("No valid dataset is given.")

    return test_loader


def test(model, args):
    """Test the model accuracy"""
    test_loader = get_test_loader(args)
    model.eval()
    correct = 0

    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    accuracy = correct / float(len(test_loader.dataset))
    print('\nTest set: Accuracy: {}/{} ({:.1f}%)\n'.format(
        correct, len(test_loader.dataset), 100. * accuracy))

    return accuracy


def match_model_weights(model, newmodel, cfg_mask):
    """Create a new model with the pruned architecture"""
    layer_id_in_cfg = 0
    start_mask = torch.ones(3)
    end_mask = cfg_mask[layer_id_in_cfg]

    for [m0, m1] in zip(model.modules(), newmodel.modules()):
        if isinstance(m0, nn.BatchNorm2d):
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            m1.weight.data = m0.weight.data[idx1.tolist()].clone()
            m1.bias.data = m0.bias.data[idx1.tolist()].clone()
            m1.running_mean = m0.running_mean[idx1.tolist()].clone()
            m1.running_var = m0.running_var[idx1.tolist()].clone()
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
            w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
            w1 = w1[idx1.tolist(), :, :, :].clone()
            m1.weight.data = w1.clone()
        elif isinstance(m0, nn.Linear):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            m1.weight.data = m0.weight.data[:, idx0].clone()
            m1.bias.data = m0.bias.data.clone()

    return newmodel


def save_pruned_model(newmodel, cfg, args, accuracy):
    """Save the pruned model and configuration"""
    # Create directory if it doesn't exist
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    # Save model state
    torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()},
               os.path.join(args.save, 'pruned.pth.tar'))

    # Save pruning info
    num_parameters = sum([param.nelement() for param in newmodel.parameters()])
    savepath = os.path.join(args.save, "prune.txt")
    with open(savepath, "w") as fp:
        fp.write("Configuration: \n" + str(cfg) + "\n")
        fp.write("Number of parameters: \n" + str(num_parameters) + "\n")
        fp.write("Test accuracy: \n" + str(accuracy))

    print(f"Pruned model saved to {args.save}")
    print(f"Total parameters: {num_parameters}")


def main():
    """Main function to run the pruning process"""
    # Parse arguments
    args = parse_args()

    # Load model
    model = load_model(args)
    print(model)

    # Compute threshold for pruning
    threshold, total_channels = compute_mask_threshold(model, args)

    # Prune the model
    cfg, cfg_mask = prune_model(model, threshold, total_channels)

    # Test the model after simple pruning (zeroing out weights)
    accuracy = test(model, args)

    # Create a new model with the pruned architecture
    print("Creating new model with configuration:", cfg)
    new_model = vgg(dataset=args.dataset, cfg=cfg)
    if args.cuda:
        new_model.cuda()

    # Copy weights from original model to new model
    new_model = match_model_weights(model, new_model, cfg_mask)

    # Save the pruned model
    save_pruned_model(new_model, cfg, args, accuracy)

    # Test the final pruned model
    print("Testing final pruned model:")
    print(new_model)
    test(new_model, args)


if __name__ == '__main__':
    main()