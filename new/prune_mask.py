import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from vgg import vgg


# Functions should match vggprune.py, except main and save_pruned_model
# Might be able to collapse it into 1

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
    parser.add_argument('--dataset', type=str, default='cifar100',
                        help='training dataset (default: cifar100)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--percent', type=float, default=0.5,
                        help='scale sparse rate (default: 0.5)')
    parser.add_argument('--model', default='', type=str, metavar='PATH',
                        help='path to raw trained model (default: none)')
    parser.add_argument('--save', default='.', type=str, metavar='PATH',
                        help='path to save prune model (default: none)')
    parser.add_argument('--depth', default=19, type=int,
                        help='depth of resnet and densenet')
    parser.add_argument('--arch', default='vgg', type=str,
                        help='architecture to use')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args


def load_model(args):
    # Changed from original since we are using vgg only
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

    return model


def compute_mask_threshold(model, args):
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


def prune_model(model, threshold):
    pruned = 0
    cfg = []
    cfg_mask = []
    total_channels = 0

    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.BatchNorm2d):
            weight_copy = m.weight.data.abs().clone()
            mask = weight_copy.gt(threshold).float().cuda()
            pruned = pruned + mask.shape[0] - torch.sum(mask)
            total_channels += mask.shape[0]
            m.weight.data.mul_(mask)
            m.bias.data.mul_(mask)
            cfg.append(int(torch.sum(mask)))
            cfg_mask.append(mask.clone())
            print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                  format(k, mask.shape[0], int(torch.sum(mask))))
        elif isinstance(m, nn.MaxPool2d):
            cfg.append('M')

    pruned_ratio = pruned / total_channels
    print(f'Pruned ratio: {pruned_ratio:.2f}')
    print('Pre-processing Successful!')

    return cfg, cfg_mask


def get_test_loader(args):
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


def save_pruned_model(model, cfg, args, accuracy):
    savepath = os.path.join(args.save, "prune.txt")
    with open(savepath, "w") as fp:
        fp.write("Configuration: \n")
        fp.write(str(cfg) + "\n")
        fp.write("Test accuracy: \n")
        fp.write(str(accuracy))

    print(f"Pruned model saved to {args.save}")


def main():
    args = parse_args()

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    model = load_model(args)
    print(model)

    threshold, total_channels = compute_mask_threshold(model, args)

    cfg, cfg_mask = prune_model(model, threshold)

    # This is the difference with vggprune: we save the zeroed model
    torch.save({'cfg': cfg, 'state_dict': model.state_dict()},
               os.path.join(args.save, 'pruned.pth.tar'))

    accuracy = test(model, args)

    save_pruned_model(model, cfg, args, accuracy)


if __name__ == '__main__':
    main()
