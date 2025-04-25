import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from vgg import vgg
import network_slimming_prune
import l1_norm_prune

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
    parser.add_argument('pruning_method', type=str,
                        help='Pruning method to use (e.g., l1_norm, network_slimming)')
    parser.add_argument('weights_init_method', type=str,
                    help='Weights initialisation method in the pruned model (default: unpruned model weights)', default='unpruned')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='training dataset (default: cifar10)')
    parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                        help='input batch size for testing (default: 256)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--depth', type=int, default=16,
                        help='depth of the vgg')
    parser.add_argument('--percent', type=float, default=0.5,
                        help='scale sparse rate (default: 0.5)')
    parser.add_argument('--model', default='', type=str, metavar='PATH',
                        help='path to the model (default: none)')
    parser.add_argument('--initial_model', default='', type=str, metavar='PATH',
                    help='path to the initial model (default: none)')
    parser.add_argument('--save', default='', type=str, metavar='PATH',
                        help='path to save pruned model (default: none)')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args


def load_model(args, model_path):
    """Load model and checkpoint if provided"""
    model = vgg(dataset=args.dataset, depth=args.depth) if args.pruning_method=="l1_norm" else vgg(dataset=args.dataset, depth=args.depth, simple_classifier=True)
    if args.cuda:
        model.cuda()

    if model_path:
        if os.path.isfile(model_path):
            print("=> loading checkpoint '{}'".format(model_path))
            checkpoint = torch.load(model_path)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
                  .format(model_path, checkpoint['epoch'], best_prec1))
        else:
            raise Warning("=> no checkpoint found at '{}'".format(model_path))

    return model


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

def save_pruned_model(newmodel, cfg, args, accuracy):
    """Save the pruned model and configuration"""
    model_file_name = os.path.splitext(os.path.basename(args.model))[0]

    # Create directory if it doesn't exist
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    # Save model state
    torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()},
               os.path.join(args.save, f'{model_file_name}_pruned_{args.pruning_method}_{args.weights_init_method}.pth.tar'))

    # Save pruning info
    num_parameters = sum([param.nelement() for param in newmodel.parameters()])
    savepath = os.path.join(args.save, f"{model_file_name}_pruned_{args.pruning_method}_{args.weights_init_method}.txt")
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
    model = load_model(args, args.model)

    if args.pruning_method == 'l1_norm':
        cfg, cfg_mask = l1_norm_prune.prune(model, args)
    elif args.pruning_method == 'network_slimming':
        cfg, cfg_mask = network_slimming_prune.prune(model, args)
    else:
        raise Warning("No valid pruning method is given.")

    # Test the model after simple pruning (zeroing out weights)
    accuracy = test(model, args)

    # Create a new model with the pruned architecture
    print("Creating new model with configuration:", cfg)
    new_model = vgg(dataset=args.dataset, cfg=cfg, simple_classifier=True) if args.pruning_method=="network_slimming" else vgg(dataset=args.dataset, cfg=cfg)
    if args.cuda:
        new_model.cuda()

    # Copy weights from original model to new model
    if args.pruning_method == 'l1_norm':
        if args.weights_init_method == 'unpruned':
            new_model = l1_norm_prune.match_model_weights(model, new_model, cfg_mask)
        elif args.weights_init_method == 'random':
            new_model = new_model
        elif args.weights_init_method == 'initial':
            new_model = l1_norm_prune.match_model_weights(model, new_model, cfg_mask, weight_model=load_model(args, args.initial_model))
        else: raise Warning("No valid weight initialisation method is given.")
    elif args.pruning_method == 'network_slimming':
        if args.weights_init_method == 'unpruned':
            new_model = network_slimming_prune.match_model_weights(model, new_model, cfg_mask)
        elif args.weights_init_method == 'random':
            new_model = new_model
        elif args.weights_init_method == 'initial':
            new_model = network_slimming_prune.match_model_weights(model, new_model, cfg_mask, weight_model=load_model(args, args.initial_model))
        else: raise Warning("No valid weight initialisation method is given.")

    # Save the pruned model
    save_pruned_model(new_model, cfg, args, accuracy)

    # Test the final pruned model
    print("Testing final pruned model:")
    print(new_model)
    test(new_model.cuda(), args)


if __name__ == '__main__':
    main()