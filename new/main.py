import os
import argparse
import random
import torch
from torchvision import datasets, transforms
import shutil
from vgg import vgg
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import wandb

# Default cifar-10 from old code.
def get_cifar10(train_batch_size, test_batch_size, kwargs):
    '''
    :arg train_batch_size : training set batch size
    :arg test_batch_size : test set batch size
    :arg kwargs : cuda arguments

    :return train_loader, test_loader : train and test data loader
    '''
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.cifar10', train=True, download=True,
                         transform=transforms.Compose([
                             transforms.Pad(4),
                             transforms.RandomCrop(32),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                         ])),
        batch_size=train_batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])),
        batch_size=test_batch_size, shuffle=True, **kwargs)

    return train_loader, test_loader


def main():
    # Training Settings
    parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')
    parser.add_argument('run_name', type=str, help='Name of the log run')
    parser.add_argument('--sparsity-regularization', '-sr', dest='sr', action='store_true',
                        help='train with channel sparsity regularization')
    parser.add_argument('--s', type=float, default=0.0001,
                        help='scale sparse rate (default: 0.0001)')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='training dataset (default: cifar10)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 256)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 160)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--fine_tuning',default='', type=str, metavar='PATH',
                        help='path to the pruned model to be fine tuned')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save', type=str, metavar='PATH',
                        help='path to save prune model (default: current directory)')
    parser.add_argument('--depth', default=16, type=int,
                        help='depth of the neural network')
    parser.add_argument('--simple-classifier', action='store_true', default=False,
                        help='Simple classifier used for network slimming')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    train_loader, test_loader = get_cifar10(args.batch_size, args.test_batch_size, kwargs)

    if args.fine_tuning:
        checkpoint = torch.load(args.fine_tuning)
        print(checkpoint['cfg'])
        model = vgg(dataset=args.dataset, depth=args.depth, cfg=checkpoint['cfg']) if not args.simple_classifier else vgg(dataset=args.dataset, depth=args.depth, cfg=checkpoint['cfg'], simple_classifier=True)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model = vgg(dataset=args.dataset, depth=args.depth) if not args.simple_classifier else vgg(dataset=args.dataset, depth=args.depth, simple_classifier=True)

    if args.cuda:
        model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # if args.resume:
    #     if os.path.isfile(args.resume):
    #         print("=> loading checkpoint '{}'".format(args.resume))
    #         checkpoint = torch.load(args.resume)
    #         args.start_epoch = checkpoint['epoch']
    #         best_prec1 = checkpoint['best_prec1']
    #         model.load_state_dict(checkpoint['state_dict'])
    #         optimizer.load_state_dict(checkpoint['optimizer'])
    #         print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
    #             .format(args.resume, checkpoint['epoch'], best_prec1))
    #     else:
    #         print("=> no checkpoint found at '{}'".format(args.resume))

    def updateBN():
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.grad.data.add_(args.s*torch.sign(m.weight.data))  # L1

    def train(epoch):
        model.train()
        avg_loss = 0.
        train_acc = 0.
        for batch_idx, (data, target) in enumerate(train_loader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            avg_loss += loss.data.item()
            pred = output.data.max(1, keepdim=True)[1]
            train_acc += pred.eq(target.data.view_as(pred)).cpu().sum()
            loss.backward()
            if args.sr:
                updateBN()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data.item()))

        wandb.log({
            "training_loss": avg_loss,
            "training_accuracy": train_acc / len(train_loader.dataset),
            "epoch": epoch,
            "lr": optimizer.param_groups[-1]['lr']
            }
        , step=epoch)

    def test():
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).data.item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        
        wandb.log({
            "validation_loss": test_loss,
            "validation_accuracy": correct / len(test_loader.dataset),
            "epoch": epoch,
            "lr": optimizer.param_groups[-1]['lr']
            }
        , step=epoch)

        return correct / float(len(test_loader.dataset))

    def save_checkpoint(state, is_best, filename, filepath=args.save):
        torch.save(state, os.path.join(filepath, f'{filename}.pth.tar'))
        if is_best:
            shutil.copyfile(os.path.join(filepath, f'{filename}.pth.tar'), os.path.join(filepath, f'{filename}_best.pth.tar'))

    # args.run_name = args.run_name + str(random.getrandbits(16))  # Add random bits to the run name to avoid overwriting
    args.run_name = args.run_name
    with wandb.init(
        project="network_pruning", 
        name=args.run_name, 
        config={"epochs": args.epochs, "lr": args.lr, "fine tuning": True if args.fine_tuning != "" else False} + {"sr": args.s} if args.sr else {},
        tags=["vgg", "pruning"],
        mode="offline",
        dir="./wandb_logs"
    ):
        wandb.watch(model, log="all", log_freq = 100)
        save_checkpoint({
            'epoch': 0,
            'state_dict': model.state_dict(),
            'best_prec1': 0,
            'optimizer': optimizer.state_dict(),
            'cfg': model.cfg
        }, False, filename=args.run_name+"_initial_model")

        best_prec1 = 0.
        for epoch in range(1, args.epochs):
            if epoch in [args.epochs*0.5, args.epochs*0.75]:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.1
            train(epoch)
            prec1 = test()
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
                'cfg': model.cfg
            }, is_best, filename=args.run_name)

if __name__=='__main__':
    main()