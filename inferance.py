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

import utils
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


# def load_datas(args):
#     transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
#     if args.dataset == "mnist":
#         traindataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
#         testdataset = datasets.MNIST('../data', train=False, transform=transform)
#
#     elif args.dataset == "cifar10":
#         traindataset = datasets.CIFAR10('../data', train=True, download=True, transform=transform)
#         testdataset = datasets.CIFAR10('../data', train=False, transform=transform)
#
#     elif "CFD" in args.dataset:
#         args.dataset = 'CFD'
#         traindataset, testdataset = load_data(args)
#         from archs.CFD import fc1
#
#     elif args.dataset == "fluidanimation":
#         traindataset, testdataset = load_data(args)
#         from archs.fluidanimation import fc1
#
#     elif args.dataset == "puremd":
#         traindataset, testdataset = load_data(args)
#         from archs.puremd import fc1
#
#     elif args.dataset == "cosmoflow":
#         traindataset, testdataset = load_data(args)
#         from archs.cosmoflow import fc1
#
#     elif args.dataset == "dimenet":
#         traindataset, testdataset = load_data(args)
#
#     else:
#         print("\nWrong Dataset choice \n")
#         exit()
#
#     train_loader = torch.utils.data.DataLoader(traindataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
#                                                drop_last=False)
#     # train_loader = cycle(train_loader)
#     test_loader = torch.utils.data.DataLoader(testdataset, batch_size=len(testdataset), shuffle=False, num_workers=0,
#                                               drop_last=True)
#     return train_loader, test_loader


def cal_performance(model_path, input_shape, classfication=False):
    model = torch.load(model_path)
    ratio = print_nonzeros(model)
    print("Sparsity in model: {:.2f}%".format(ratio))

    for name, param in model.named_parameters():
        print(name, param.shape)

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
    train_loader, test_loader, _ = utils.get_essentials(args)
    correct = 0
    r_error = utils.AverageMeter()
    test_losses = utils.AverageMeter()
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
                r2 = utils.accuracy(output.data, target, topk=(1,))[0].item()
                r_error.update(r2, data.size(0))
            else:
                test_loss = r2_loss(output, target).item()
                relative_error = torch.mean(torch.abs(output - target) / torch.abs(target + 1e-8))
                relative_errors += relative_error.item()
            test_losses.update(test_loss)

    average_time = mean(times)
    average_test_loss = test_losses.avg
    average_relative_error = relative_errors / len(test_loader)
    if classfication:
        accuracy = 100. * correct / len(test_loader.dataset)
        average_relative_error = accuracy.item()

    print(average_relative_error, r_error.avg)
    print("Time cost: %s" % average_time)
    print("Test loss: %s" % average_test_loss)
    print("Relative error: %s" % average_relative_error)
    return flops, average_time, average_test_loss, r_error.avg, 1-ratio/100


def plot_results(x, y, y_label, x_label='Pruning Level', title=None, savefig=None, td=False):
    colors = ['skyblue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta']
    if td:
        plt.figure()
        for i in range(len(y)):
            #plt.plot(x[i], y[i], '-o', linewidth=2, color=colors[i], label=y_label[i])
            # plot the fitting line of x and y
            z = np.polyfit(x[i], y[i], 20)
            p = np.poly1d(z)
            plt.plot(x[i], p(x[i]), '-o', linewidth=2, color=colors[i], label=y_label[i])
        plt.legend()
        plt.xlim(-0.05, 1)
    else:
        plt.figure()
        plt.plot(x, y, '-o', linewidth=2)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

    plt.xlabel(x_label)
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


def read_ML_results():
    with open('/aul/homes/sgao014/Projects/AI4Science/ML-sorrogate/logs/cifar10/bz200-N7194360-wd0.0005-regTrue-31-18-18-la-0.1-lr0.001-temp0.67-noema/results.txt', 'r') as f:
        lines = f.readlines()
        pruned_ratios = lines[0].split('[')[1].split(']')[0].split(',')
        pruned_ratios = [float(x.strip()) for x in pruned_ratios]
        quality = lines[1].split('[')[1].split(']')[0].split(',')
        quality = [float(x.strip()) for x in quality]
    return pruned_ratios, quality


def inferance(roots, args):
    flops = []
    average_times = []
    average_test_losses = []
    average_relative_errors = []
    pruned_ratios = []
    for path in roots:
        print(path)
        # flops_, average_time, average_test_loss, average_relative_error, pruned_ratio = cal_performance(path, (1, 4, 128, 128, 128))
        # flops_, average_time, average_test_loss, average_relative_error, pruned_ratio = cal_performance(path, (1, 15))
        flops_, average_time, average_test_loss, average_relative_error, pruned_ratio = cal_performance(path, (1, 28, 28), classfication=True)

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
    # flops = np.arange(0, len(flops))
    plot_results(pruned_ratios, average_times, 'Time', title='Time cost', savefig=f'{root}/time-{args.dataset}.png')
    plot_results(pruned_ratios, average_test_losses, 'Loss', title='Test loss', savefig=f'{root}/loss-{args.dataset}.png')
    plot_results(pruned_ratios, average_relative_errors, 'Relative Error',
                 title='Relative error', savefig=f'{root}/relative_error-{args.dataset}.png')


def read_draw(roots):
    for path in roots:
        flops, average_times, average_test_losses, average_relative_errors, pruned_ratios = read_results(path)

        # plot the results in a plot
        # flops = np.arange(0, len(flops))
        plot_results(pruned_ratios, average_times, 'Time', title='Time cost', savefig=f'{root}/time-{args.dataset}.png')
        plot_results(pruned_ratios, average_test_losses, 'Loss', title='Test loss', savefig=f'{root}/loss-{args.dataset}.png')
        plot_results(pruned_ratios, average_relative_errors, 'Relative Error / Acc', title='Relative error / Acc',
                     savefig=f'{root}/relative_error or Acc-{args.dataset}.png')


def draw_compare_acc(root1, root2):
    _, _, _, average_relative_errors1, pruned_ratios1 = read_results(root1[0])
    _, _, _, average_relative_errors2, pruned_ratios2 = read_results(root2[0])
    plot_results([pruned_ratios1, pruned_ratios2], [average_relative_errors1, average_relative_errors2],
                 y_label=['prune_neuron', 'gradually_prune_neuron'], title='Compare Acc',
                 savefig=f'{root2[0]}/Compare Acc.png', td=True)


def draw_compare_acc1(root1, root2):
    _, _, _, average_relative_errors1, pruned_ratios1 = read_results(root1[0])
    _, _, _, average_relative_errors2, pruned_ratios2 = read_results(root2[0])
    pruned_ratios3, quality = read_ML_results()
    plot_results([pruned_ratios1, pruned_ratios2, pruned_ratios3], [average_relative_errors1, average_relative_errors2, quality],
                 y_label=['prune_weights', 'prune_neuron', 'L0 prune'], title='Compare Acc',
                 savefig=f'{root2[0]}/Compare Acc.png', td=True)


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
    parser.add_argument("--dataset", default="cifar10", type=str,
                        help="mnist | cifar10 | fashionmnist | cifar100 | "
                             "CFD | fluidanimation | puremd | cosmoflow | dimenet")
    parser.add_argument("--arch_type", default="lenet5", type=str,
                        help="fc1 | lenet5 | alexnet | vgg16 | resnet18 | densenet121")
    parser.add_argument("--prune_percent", default=20, type=int, help="Pruning percent")
    parser.add_argument("--prune_iterations", default=10, type=int, help="Pruning iterations count")
    parser.add_argument("--device", default="cuda", type=str, help="cuda | cpu")

    args = parser.parse_args()

    # root = './saves/puremd/01-11-17'
    # root = './saves/fc1/puremd'
    # root = './saves/fluidanimation/01-12-16'
    # root = './saves/CFD-fs/01-09-18'
    # root = './saves/mnist/02-01-19-gradually-prune-neuron'
    root = './saves/cifar10/02-02-10-prune-neuron'
    roots = [os.path.join(root, x) for x in os.listdir(root) if 'model_lt' in x]
    # sort the roots
    roots.sort(key=lambda x: int(x.split('_')[0].split('/')[-1]))

    inferance(roots, args)
    # read_draw([root])
    # draw_compare_acc(['./saves/mnist/01-29-13'], ['./saves/mnist/02-02-09-gradually-prune-neuron'])
    # draw_compare_acc1(['./saves/cifar10/01-31-18-prune-weights'], ['./saves/cifar10/01-31-18-prune-neuron'])
