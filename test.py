import argparse
import os
import torch.nn.functional as F
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from calflops import calculate_flops
from numpy import mean

import torchvision.transforms as transforms
import torchvision.datasets as datasets

from load_dataset import load_data
from utils import print_nonzeros


num_gpus = torch.cuda.device_count()
print(f'Number of available GPUs: {num_gpus}')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
global test_loader, args


def r2_loss(output, target):
    return torch.mean(torch.sum(torch.square(target - output), dim=0) / torch.sum(torch.square(target - torch.mean(target, dim=0)), dim=0))


def l1_loss(output, target):
    return torch.mean(torch.abs(output - target))


def load_datas(args):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    if args.dataset == "mnist":
        traindataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
        testdataset = datasets.MNIST('../data', train=False, transform=transform)
        from archs.mnist import AlexNet, LeNet5, fc1, vgg, resnet

    elif "CFD" in args.dataset:
        args.dataset = 'CFD'
        traindataset, testdataset = load_data(args)
        from archs.CFD import fc1

    elif args.dataset == "fluidanimation":
        traindataset, testdataset = load_data(args)
        from archs.fluidanimation import fc1

    elif args.dataset == "puremd":
        traindataset, testdataset = load_data(args)
        from archs.puremd import fc1

    elif args.dataset == "cosmoflow":
        traindataset, testdataset = load_data(args)
        from archs.cosmoflow import fc1

    elif args.dataset == "dimenet":
        traindataset, testdataset = load_data(args)

    else:
        print("\nWrong Dataset choice \n")
        exit()

    train_loader = torch.utils.data.DataLoader(traindataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                                               drop_last=False)
    # train_loader = cycle(train_loader)
    test_loader = torch.utils.data.DataLoader(testdataset, batch_size=len(testdataset), shuffle=False, num_workers=0,
                                              drop_last=True)
    return train_loader, test_loader


def cal_performance(model_path, input_shape, classfication=False):
    model = torch.load(model_path)
    ratio = print_nonzeros(model)
    print("Sparsity in model: {:.2f}%".format(ratio))

    flops, macs, params = calculate_flops(model=model,
                                          input_shape=input_shape,
                                          output_as_string=True,
                                          output_precision=4)
    print("FLOPs:%s   MACs:%s   Params:%s \n" % (flops, macs, params))
    # turn str flops to float. (eg. 100.20 KFLOPS -> 100200.0)
    flops = float(flops.split()[0]) * 1000 if 'KFLOPS' in flops else float(flops.split()[0])
    flops = flops * ratio/100
    print("Remain Flops: %s" % flops)

    times = []
    test_losses = 0
    relative_errors = 0
    model = model.to(device)
    train_loader, test_loader = load_datas(args)
    correct = 0
    with torch.no_grad():
        for i, (datas) in enumerate(test_loader):
            if isinstance(datas, dict):
                data = datas
                target = datas['targets'][0]
            else:
                data, target = datas
                data, target = data.to(device), target.to(device)
            start = time.time()
            output = model(data)
            end = time.time()
            times.append(end - start)
            if classfication:
                test_loss = F.cross_entropy(output, target).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            else:
                test_loss = r2_loss(output, target).item()
                relative_error = torch.mean(torch.abs(output - target) / torch.abs(target + 1e-8))
                relative_errors += relative_error.item()
            test_losses += test_loss

    average_time = mean(times)
    average_test_loss = test_losses / len(test_loader)
    average_relative_error = relative_errors / len(test_loader)
    if classfication:
        accuracy = 100. * correct / len(test_loader.dataset)
        average_relative_error = accuracy

    print("Time cost: %s" % average_time)
    print("Test loss: %s" % average_test_loss)
    print("Relative error: %s" % average_relative_error)
    return flops, average_time, average_test_loss, average_relative_error, 1-ratio/100


def plot_results(x, y, y_label, x_label='Pruning Level', title=None, savefig=None):
    plt.figure()
    plt.plot(x, y, '-o', linewidth=2)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(savefig)


def read_results(root):
    with open(f'{root}/results.txt', 'r') as f:
        lines = f.readlines()
        flops = lines[0].split('[')[1].split(']')[0].split(',')
        flops = [float(x.strip()) for x in flops]
        times = lines[1].split('[')[1].split(']')[0].split(',')
        times = [float(x.strip()) for x in times]
        test_losses = lines[2].split('[')[1].split(']')[0].split(',')
        test_losses = [float(x.strip()) for x in test_losses]
        relative_errors = lines[3].split('[')[1].split(']')[0].split(',')
        relative_errors = [float(x.strip()) for x in relative_errors]
        pruned_ratios = lines[4].split('[')[1].split(']')[0].split(',')
        pruned_ratios = [float(x.strip()) for x in pruned_ratios]
    return flops, times, test_losses, relative_errors, pruned_ratios


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=1.2e-3, type=float, help="Learning rate")
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--start_iter", default=0, type=int)
    parser.add_argument("--end_iter", default=50, type=int)  # 100
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--valid_freq", default=1, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prune_type", default="lt", type=str, help="lt | reinit")
    parser.add_argument("--gpu", default="0", type=str)
    parser.add_argument("--dataset", default="puremd", type=str,
                        help="mnist | cifar10 | fashionmnist | cifar100 | "
                             "CFD | fluidanimation | puremd | cosmoflow | dimenet")
    parser.add_argument("--arch_type", default="fc1", type=str,
                        help="fc1 | lenet5 | alexnet | vgg16 | resnet18 | densenet121")
    parser.add_argument("--prune_percent", default=20, type=int, help="Pruning percent")
    parser.add_argument("--prune_iterations", default=10, type=int, help="Pruning iterations count")
    parser.add_argument("--device", default="cuda", type=str, help="cuda | cpu")

    args = parser.parse_args()

    root = './saves/puremd-fs/01-10-13'
    # root = './saves/fc1/puremd'
    # root = './saves/CFD/01-08-20/'
    # root = './saves/CFD-fs/01-09-18'
    # root = './saves/cosmoflow/01-08-20'
    roots = [os.path.join(root, x) for x in os.listdir(root) if 'model_lt' in x]
    # sort the roots
    roots.sort(key=lambda x: int(x.split('_')[0].split('/')[-1]))
    flops = []
    average_times = []
    average_test_losses = []
    average_relative_errors = []
    pruned_ratios = []
    roots = ['saves/puremd-fs/01-10-13/9_model_lt.pth.tar']
    for path in roots:
        print(path)
        # flops_, average_time, average_test_loss, average_relative_error, pruned_ratio = cal_performance(path, (1, 4, 128, 128, 128))
        flops_, average_time, average_test_loss, average_relative_error, pruned_ratio = cal_performance(path, (1, 9))
        # flops_, average_time, average_test_loss, average_relative_error, pruned_ratio = cal_performance(path, (1, 28, 28), classfication=True)

        flops.append(flops_)
        average_times.append(average_time)
        average_test_losses.append(average_test_loss)
        average_relative_errors.append(average_relative_error)
        pruned_ratios.append(pruned_ratio)

    # write done the results
    with open(f'{root}/results.txt', 'w') as f:
        f.write(f'FLOPs: {flops}\n')
        f.write(f'Time: {average_times}\n')
        f.write(f'Test loss: {average_test_losses}\n')
        f.write(f'Relative error: {average_relative_errors}\n')
        f.write(f'Pruned ratio: {pruned_ratios}\n')

    # plot the results in a plot
    flops = np.arange(0, len(flops))
    plot_results(flops, average_times, 'Time', title='Time cost', savefig=f'{root}/time-{args.dataset}.png')
    plot_results(flops, average_test_losses, 'Loss', title='Test loss', savefig=f'{root}/loss-{args.dataset}.png')
    plot_results(flops, average_relative_errors, 'Relative Error', title='Relative error', savefig=f'{root}/relative_error-{args.dataset}.png')



