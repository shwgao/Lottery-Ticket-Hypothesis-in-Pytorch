# Importing Libraries
import argparse
import copy
import datetime
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import os
from tensorboardX import SummaryWriter
import seaborn as sns
import torch.nn.init as init
import pickle

# Custom Libraries
import utils
from load_dataset import load_data


def r2_loss(output, target):
    return torch.mean(torch.sum(torch.square(target - output), dim=0) / torch.sum(torch.square(target - torch.mean(target, dim=0)), dim=0))


def mae_loss(output, target):
    return torch.mean(torch.abs(output - target))


def log_cosh_loss(output, target):
    return torch.mean(torch.log(torch.cosh(output - target)))


def print_nonzeros(model):
    model_mask = model.mask.data.cpu().numpy()
    print(model_mask)
    return np.count_nonzero(model_mask)


now_time = datetime.datetime.now().strftime("%m-%d-%H")

# Plotting Style
sns.set_style('darkgrid')
debugging = False
# criterion = nn.CrossEntropyLoss() # Default was F.nll_loss
criterion_train = log_cosh_loss
criterion_test = r2_loss
# if args.dataset == 'dimenet':
#     criterion = nn.MSELoss


# Main
def main(args, ITE=0):
    writer = SummaryWriter(f'runs/{args.dataset}/{now_time}/')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    reinit = True if args.prune_type == "reinit" else False

    # Load Dataset
    global model, LeNet5
    dt = args.dataset
    args.dataset = args.dataset.replace('-fs', '')
    train_loader, test_loader, model = utils.get_essentials(args)
    args.dataset = dt
    # from archs.CFD import fc1
    from archs.puremd import fc1
    model = fc1.fc1_mask().to(device)
    # Importing Network Architecture

    # Weight Initialization
    if args.dataset == "dimenet":
        pass
    else:
        model.apply(weight_init)

    # Copying and Saving Initial State
    initial_state_dict = copy.deepcopy(model.state_dict())
    utils.checkdir(f"{os.getcwd()}/saves/{args.dataset}/{now_time}/")
    if args.dataset == "dimenet":
        torch.save(model.state_dict(),
                   f"{os.getcwd()}/saves/{args.dataset}/{now_time}/initial_state_dict_{args.prune_type}.pth")
    else:
        torch.save(model,
                   f"{os.getcwd()}/saves/{args.dataset}/{now_time}/initial_state_dict_{args.prune_type}.pth.tar")

    # Making Initial Mask
    make_mask(model)

    # Optimizer and Loss
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-4)

    # Layer Looper
    for name, param in model.named_parameters():
        print(name, param.size())

    # Pruning
    # NOTE First Pruning Iteration is of No Compression
    bestr2 = 100
    best_relative_error = 100000
    ITERATION = args.prune_iterations
    comp = np.zeros(ITERATION, float)
    bestre = np.zeros(ITERATION, float)
    step = 0
    all_loss = np.zeros(args.end_iter, float)
    all_relative_error = np.zeros(args.end_iter, float)

    for _ite in range(args.start_iter, ITERATION):
        if not _ite == 0:
            prune_by_percentile(args.prune_percent, resample=resample, reinit=reinit)
            if reinit:
                model.apply(weight_init)
                step = 0
                for name, param in model.named_parameters():
                    if 'weight' in name:
                        weight_dev = param.device
                        param.data = torch.from_numpy(param.data.cpu().numpy() * mask[step]).to(weight_dev)
                        step = step + 1
                step = 0
            else:
                original_initialization(mask, initial_state_dict)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        print(f"\n--- Pruning Level [{ITE}:{_ite}/{ITERATION}]: ---")

        # Print the table of Nonzeros in each layer
        comp1 = print_nonzeros(model)
        comp[_ite] = comp1
        pbar = tqdm(range(args.end_iter))

        for iter_ in pbar:

            # Frequency for Testing
            if iter_ % args.valid_freq == 0:
                test_loss, relative_error = test(model, test_loader, criterion_test)
                writer.add_scalars('test_loss', {f'{_ite}': test_loss}, iter_)
                writer.add_scalars('relative_error', {f'{_ite}': relative_error}, iter_)

                # Save Weights
                if relative_error < best_relative_error:
                    best_relative_error = relative_error
                    utils.checkdir(f"{os.getcwd()}/saves/{args.dataset}/{now_time}/")
                    if args.dataset == "dimenet":
                        torch.save(model.state_dict(),
                                   f"{os.getcwd()}/saves/{args.dataset}/{now_time}/{_ite}_model_{args.prune_type}.pth")
                    else:
                        torch.save(model,
                                   f"{os.getcwd()}/saves/{args.dataset}/{now_time}/{_ite}_model_{args.prune_type}.pth.tar")
                if test_loss < bestr2:
                    bestr2 = test_loss

            # Training
            loss = train(model, train_loader, optimizer, criterion_train)
            all_loss[iter_] = loss
            all_relative_error[iter_] = relative_error
            # writer['train_loss'].add_scalar(f'/Loss/{_ite}/train', loss, iter_)
            writer.add_scalars('train_loss', {f'{_ite}': loss}, iter_)

            # Frequency for Printing relative_error and Loss
            if iter_ % args.print_freq == 0:
                pbar.set_description(
                    f'Train Epoch: {iter_}/{args.end_iter} Loss: {loss:.6f} best_test_loss: {bestr2:.8f} '
                    f'relative_error: {relative_error*100:.4f}% Best relative error: {best_relative_error*100:.4f}%')

        bestre[_ite] = best_relative_error

        # Plotting Loss (Training), relative_error (Testing), Iteration Curve
        # NOTE Loss is computed for every iteration while relative_error is computed only for every {args.valid_freq}
        # iterations. Therefore relative_error saved is constant during the uncomputed iterations.
        # NOTE Normalized the relative_error to [0,100] for ease of plotting.
        plt.plot(np.arange(1, (args.end_iter) + 1),
                 100 * (all_loss - np.min(all_loss)) / np.ptp(all_loss).astype(float), c="blue", label="Loss")
        plt.plot(np.arange(1, (args.end_iter) + 1), all_relative_error, c="red", label="relative_error")
        plt.title(f"Loss Vs relative_error Vs Iterations ({args.dataset},{args.arch_type})")
        plt.xlabel("Iterations")
        plt.ylabel("Loss and relative_error")
        plt.legend()
        plt.grid(color="gray")
        utils.checkdir(f"{os.getcwd()}/plots/lt/{args.dataset}/{now_time}/")
        plt.savefig(
            f"{os.getcwd()}/plots/lt/{args.dataset}/{now_time}/{args.prune_type}_LossVsRelative_error_{comp1}.png",
            dpi=1200)
        plt.close()

        # Dump Plot values
        utils.checkdir(f"{os.getcwd()}/dumps/lt/{args.dataset}/{now_time}/")
        all_loss.dump(f"{os.getcwd()}/dumps/lt/{args.dataset}/{now_time}/{args.prune_type}_all_loss_{comp1}.dat")
        all_relative_error.dump(
            f"{os.getcwd()}/dumps/lt/{args.dataset}/{now_time}/{args.prune_type}_all_relative_error_{comp1}.dat")

        # Dumping mask
        utils.checkdir(f"{os.getcwd()}/dumps/lt/{args.dataset}/{now_time}/")
        with open(f"{os.getcwd()}/dumps/lt/{args.dataset}/{now_time}/{args.prune_type}_mask_{comp1}.pkl",
                  'wb') as fp:
            pickle.dump(mask, fp)

        # Making variables into 0
        best_relative_error = 100000
        bestr2 = 100
        all_loss = np.zeros(args.end_iter, float)
        all_relative_error = np.zeros(args.end_iter, float)

    # Dumping Values for Plotting
    utils.checkdir(f"{os.getcwd()}/dumps/lt/{args.dataset}/{now_time}/")
    comp.dump(f"{os.getcwd()}/dumps/lt/{args.dataset}/{now_time}/{args.prune_type}_compression.dat")
    bestre.dump(f"{os.getcwd()}/dumps/lt/{args.dataset}/{now_time}/{args.prune_type}_bestrelativeerror.dat")

    # Plotting
    a = np.arange(args.prune_iterations)
    plt.plot(a, bestre, c="blue", label="Winning tickets")
    plt.title(f"Test relative error vs Unpruned Weights Percentage ({args.dataset},{args.arch_type})")
    plt.xlabel("Unpruned Weights Percentage")
    plt.ylabel("test relative error")
    plt.xticks(a, comp, rotation="vertical")
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(color="gray")
    utils.checkdir(f"{os.getcwd()}/plots/lt/{args.dataset}/{now_time}/")
    plt.savefig(f"{os.getcwd()}/plots/lt/{args.dataset}/{now_time}/{args.prune_type}_reeVsWeights.png", dpi=1200)
    plt.close()


# Function for Training
def train(model, train_loader, optimizer, criterion):
    EPS = 1e-6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_loss = 0
    model.train()
    for batch_idx, (datas) in enumerate(train_loader):
        if isinstance(datas, dict):
            data = datas
            target = datas['targets'][0]
        else:
            data, target = datas
    # for batch_idx, (imgs, targets) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        train_loss = criterion(output, target)
        train_loss.backward()
        total_loss += train_loss.item() / len(train_loader)

        # Freezing Pruned weights by making their gradients Zero
        p = model.mask
        tensor = p.data.cpu().numpy()
        grad_tensor = p.grad.data.cpu().numpy()
        grad_tensor = np.where(tensor < EPS, 0, grad_tensor)
        p.grad.data = torch.from_numpy(grad_tensor).to(device)
        optimizer.step()
        if debugging:
            print(f'train loss: {train_loss.item()}')
    mean_loss = total_loss

    if debugging:
        print(f'mean: train loss: {mean_loss}, len: {len(train_loader)}')
    return mean_loss


# Function for Testing
def test(model, test_loader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_losses = 0
    relative_errors = 0

    with torch.no_grad():
        for i, (datas) in enumerate(test_loader):
            if isinstance(datas, dict):
                data = datas
                target = datas['targets'][0]
            else:
                data, target = datas
            # for data, target in test_loader:
                data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss = criterion(output, target).item()
            test_losses += test_loss
            relative_error = torch.mean(torch.abs(output - target) / torch.abs(target + 1e-8))
            relative_errors += relative_error
            if debugging:
                print(f'test loss: {test_loss}, relative error: {relative_error}')
        test_losses /= len(test_loader)
        relative_errors /= len(test_loader)
        if debugging:
            print(f'mean: test loss: {test_losses}, relative error: {relative_errors}, len: {len(test_loader)}')
    return test_losses, relative_errors


def test1(model, test_loader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0
    output = None
    target = None
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
        test_loss /= len(test_loader.dataset)
        if output is not None and target is not None:
            relative_error = 100. * torch.mean(torch.abs(output - target) / torch.abs(target + 1e-8))
        else:
            relative_error = torch.tensor(0)  # or any other default value
    return test_loss, relative_error


# Prune by Percentile module
def prune_by_percentile(percent, resample=False, reinit=False, **kwargs):
    global step
    global mask
    global model

    # Calculate percentile value
    tensor = model.mask.data.cpu().numpy()
    alive = tensor[np.nonzero(tensor)]  # flattened array of nonzero values
    percentile_value = np.percentile(abs(alive), percent)

    # Convert Tensors to numpy and calculate
    weight_dev = model.mask.device
    new_mask = np.where(abs(tensor) < percentile_value, 0, mask)

    # Apply new weight and mask
    model.mask.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
    mask = new_mask


# Function to make an empty mask of the same size as the model
def make_mask(model):
    global mask
    tensor = model.mask.data.cpu().numpy()
    mask = np.ones_like(tensor)


def original_initialization(mask_temp, initial_state_dict):
    global model

    step = 0
    for name, param in model.named_parameters():
        if "weight" in name:
            weight_dev = param.device
            param.data = torch.from_numpy(mask_temp[step] * initial_state_dict[name].cpu().numpy()).to(weight_dev)
            step = step + 1
        if "bias" in name:
            param.data = initial_state_dict[name]


# Function for Initialization
def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


if __name__ == "__main__":
    # from gooey import Gooey
    # @Gooey

    # Arguement Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=1.2e-3, type=float, help="Learning rate")
    parser.add_argument("--batch_size", default=30000, type=int)
    parser.add_argument("--start_iter", default=0, type=int)
    parser.add_argument("--end_iter", default=50, type=int)  # 100
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--valid_freq", default=1, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prune_type", default="lt", type=str, help="lt | reinit")
    parser.add_argument("--gpu", default="1", type=str)
    parser.add_argument("--dataset", default="puremd-fs", type=str,
                        help="mnist | cifar10 | fashionmnist | cifar100 | "
                             "CFD | fluidanimation | puremd | cosmoflow | dimenet")
    parser.add_argument("--arch_type", default="fc1", type=str,
                        help="fc1 | lenet5 | alexnet | vgg16 | resnet18 | densenet121")
    parser.add_argument("--prune_percent", default=20, type=int, help="Pruning percent")
    parser.add_argument("--prune_iterations", default=10, type=int, help="Pruning iterations count")
    parser.add_argument("--device", default="cuda", type=str, help="cuda | cpu")

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # FIXME resample
    resample = False
    if args.dataset == 'cosmoflow':
        args.batch_size = 64
        args.lr = 0.0002
        criterion_test = log_cosh_loss
        criterion_train = log_cosh_loss
    elif args.dataset == 'dimenet':
        args.batch_size = 1

    # for debugging
    debugging = False
    if debugging:
        args.end_iter = 3

    # Looping Entire process
    # for i in range(0, 5):
    main(args, ITE=1)
