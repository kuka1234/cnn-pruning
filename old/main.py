from __future__ import print_function
import os
import argparse
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from collections import defaultdict
import models
import wandb
import matplotlib.pyplot as plt
wandb.login()

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')
    parser.add_argument('--dataset', type=str, default='cifar100',
                        help='training dataset (default: cifar100)')
    parser.add_argument('--sparsity-regularization', '-sr', dest='sr', action='store_true',
                        help='train with channel sparsity regularization')
    parser.add_argument('--s', type=float, default=0.0001,
                        help='scale sparse rate (default: 0.0001)')
    parser.add_argument('--refine', default='', type=str, metavar='PATH',
                        help='path to the pruned model to be fine tuned')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 256)')
    parser.add_argument('--epochs', type=int, default=160, metavar='N',
                        help='number of epochs to train (default: 160)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save', default='./logs', type=str, metavar='PATH',
                        help='path to save prune model (default: current directory)')
    parser.add_argument('--arch', default='vgg', type=str, 
                        help='architecture to use')
    parser.add_argument('--depth', default=19, type=int,
                        help='depth of the neural network')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    if args.dataset == 'cifar10':
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data.cifar10', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.Pad(4),
                            transforms.RandomCrop(32),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                        ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                        ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    else:
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./data.cifar100', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.Pad(4),
                            transforms.RandomCrop(32),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                        ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./data.cifar100', train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                        ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)

    if args.refine:
        checkpoint = torch.load(args.refine)
        model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth, cfg=checkpoint['cfg'])
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth)

    if args.cuda:
        model.cuda()


    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
                .format(args.resume, checkpoint['epoch'], best_prec1))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # additional subgradient descent on the sparsity-induced penalty term
    def updateBN():
        # all_weights = []
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                # all_weights.append(m.weight.detach().cpu().numpy())  # Collect weights
                # m.weight.grad.data.add_(args.s*m.weight.data)  # L2
                m.weight.grad.data.add_(args.s*m.weight.data + args.s*torch.sign(m.weight.data))  # L2 + L1

                # m.weight.grad.data.add_(args.s*torch.sign(m.weight.data))  # L1

        
        # if all_weights:  # Combine and log
        #     wandb.log({"BatchNorm_Weights_Histogram": wandb.Histogram(np.concatenate(all_weights))})

    def reset_bn_and_conv(threshold=0.05):
        print("Resetting layers")
        # MIGHT NEED TO RESET OPTIMIZER
        with torch.no_grad():
            prev_conv = None  # To keep track of the last Conv2D layer
            for layer in model.modules():
                if isinstance(layer, nn.Conv2d):
                    # Update the tracker for the previous Conv2D layer
                    prev_conv = layer
                
                elif isinstance(layer, nn.BatchNorm2d):
                    # Ensure there's a Conv2D layer before the BN
                    if prev_conv is None:
                        print(f"No preceding Conv2D layer found for {layer}")
                        continue
                    
                    # Check BN weights (gamma)
                    for channel, gamma in enumerate(layer.weight):
                        if abs(gamma) < threshold:
                            layer.weight[channel] = 0.5  # Default gamma
                            layer.bias[channel] = 0.0   # Default beta
                            
                            # Reset corresponding Conv2D channel
                            nn.init.kaiming_uniform_(prev_conv.weight[channel].unsqueeze(0), a=0, mode='fan_in', nonlinearity='relu')
                            if prev_conv.bias is not None:
                                prev_conv.bias[channel] = 0.0  # Default Conv bias initialization
                            print(f"Reset BN channel {channel} and corresponding Conv channel")

                    

    def train(epoch, optimizer):
        model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            # for m in model.modules():
            #     if isinstance(m, nn.BatchNorm2d):
            #         loss += 10*torch.sum(((torch.sigmoid(torch.pow(1 * m.weight.data, 2))-0.5) + (0.1 * torch.abs(m.weight.data))))

            total_loss += loss
            pred = output.data.max(1, keepdim=True)[1]
            loss.backward()
            if args.sr:
                updateBN()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print(f'Train Epoch: {epoch} {100 * batch_idx/ len(train_loader)} Loss: {loss}')
                # print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                #     epoch, batch_idx * len(data), len(train_loader.dataset),
                #     100. * batch_idx / len(train_loader), loss.data[0]))

        total_loss /= len(test_loader.dataset)
        wandb.log({
            "training_loss_total": total_loss,
            "training_loss_no_regularisation": total_loss,
            "epoch": epoch,
            "lr": optimizer.param_groups[-1]['lr'],
            "beta": args.s}
        , step=epoch)

    def test():
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            # data, target = Variable(data, volatile=True), Variable(target)
            output = model(data)
            # test_loss += F.cross_entropy(output, target, size_average=False).data[0] # sum up batch loss
            test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(test_loader.dataset)
        print(f'Test set: Average loss: {test_loss}, Accuracy: {100 * correct /  len(test_loader.dataset)}')

        wandb.log({
            "validation_loss_total": test_loss, 
            "validation_loss_no_regularisation": test_loss,
            "validation_accuracy": 100 * correct /  len(test_loader.dataset),
            "epoch": epoch}
        , step=epoch)
        return correct / float(len(test_loader.dataset))


    def save_checkpoint(state, is_best, filepath):
        torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))
        if is_best:
            shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, 'model_best.pth.tar'))

    with wandb.init(project="compression_cnns", config={}):
        wandb.watch(model, log="all", log_freq = 100)

        # # Dictionary to store outputs during forward pass
        # bn_outputs = defaultdict(list)

        # # Hook to collect outputs
        # def collect_bn_outputs(module, input, output):
        #     bn_outputs["BatchNorm_Outputs"].append(output.detach().cpu().numpy())

        # # Attach hooks to all BatchNorm layers
        # for module in model.modules():
        #     if isinstance(module, nn.BatchNorm2d):
        #         module.register_forward_hook(collect_bn_outputs)

        # Log histogram of outputs at intervals (e.g., per epoch)
        # def log_bn_outputs_histogram():
        #     if bn_outputs["BatchNorm_Outputs"]:
        #         all_outputs = np.concatenate(bn_outputs["BatchNorm_Outputs"])
        #         wandb.log({"BatchNorm_Outputs_Histogram": wandb.Histogram(all_outputs)})
        #         bn_outputs["BatchNorm_Outputs"] = []  # Reset after logging
        scalers = np.array([])
        
        def log_bn_weights_histogram(scalers):
            with torch.no_grad():
                all_weights = []
                ind = 0
                scaler_this_epoch = []
                for module in model.modules():
                    if isinstance(module, nn.BatchNorm2d):
                        for channel, gamma in enumerate(module.weight):
                            scaler_this_epoch.append(gamma.detach().cpu())
                            ind += 1

                        all_weights.append(module.weight.detach().cpu().numpy())  # Collect weights

                scalers = np.vstack((scalers, np.array(scaler_this_epoch))) if len(scalers) > 0  else np.array([scaler_this_epoch])
                if all_weights:  # Combine and log
                    wandb.log({"BatchNorm_Weights_Histogram": wandb.Histogram(np.concatenate(all_weights))})
                
                return scalers
        
        def plot_hist(data):
            plt.imshow(scalers, cmap='viridis', interpolation='nearest', aspect=20)

            # Add a color bar
            plt.colorbar()

            # Add labels (optional)
            plt.title("Heatmap Example")
            plt.xlabel("Column Index")
            plt.ylabel("Row Index")

            # Display the heatmap
            plt.savefig('scalars.png')
            # plt.show()

        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        best_prec1 = 0.

        for epoch in range(args.start_epoch, args.epochs):
            if epoch in [args.epochs*0.5, args.epochs*0.75]:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.1
            train(epoch, optimizer)
            prec1 = test()
            scalers = log_bn_weights_histogram(scalers)

            if epoch % 10 == 0 and epoch <= 70:
                reset_bn_and_conv(threshold=0.2)
                optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
                args.s = 0
                plot_hist(scalers)

            if epoch < 80:
                args.s += 0.000005

            if epoch == 80:
                args.s = 0

            # log_bn_outputs_histogram()
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best, filepath=args.save)

        print("Best accuracy: "+str(best_prec1))

if __name__=='__main__':
    main()


# TRAIN COMPLETLEY NORMALLY
# TRAIN WITH L1 + L2 REG
# TRAIN WITH INCREASING BETA and resetting
# TRAIN WITH BOTH.