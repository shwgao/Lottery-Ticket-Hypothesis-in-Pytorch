# Importing Libraries
import argparse
import copy
import datetime
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
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


now_time = datetime.datetime.now().strftime("%m-%d-%H")+'-prune-neuron'

# Plotting Style
sns.set_style('darkgrid')
debugging = False
# criterion = nn.CrossEntropyLoss() # Default was F.nll_loss
# criterion_train = nn.MSELoss()
# criterion_test = nn.MSELoss()
criterion_train = nn.CrossEntropyLoss()
criterion_test = nn.CrossEntropyLoss()
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
    train_loader, test_loader, model = utils.get_essentials(args)
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
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Layer Looper
    for name, param in model.named_parameters():
        print(name, param.size())

    # Pruning
    # NOTE First Pruning Iteration is of No Compression
    best_r2_loss = 100
    best_mean_r2_score = -100
    ITERATION = args.prune_iterations
    comp = np.zeros(ITERATION, float)
    best_r2_scores = np.zeros(ITERATION, float)
    step = 0
    all_loss = np.zeros(args.end_iter, float)
    all_r2_loss = np.zeros(args.end_iter, float)

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
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        print(f"\n--- Pruning Level [{ITE}:{_ite}/{ITERATION}]: ---")

        # Print the table of Nonzeros in each layer
        comp1 = utils.print_nonzeros(model)
        comp[_ite] = comp1
        pbar = tqdm(range(args.end_iter))

        for iter_ in pbar:

            # Frequency for Testing
            if iter_ % args.valid_freq == 0:
                test_loss, r2_loss, r2 = test(model, test_loader, criterion_test)
                writer.add_scalars('loss', {f'{_ite}_test_loss': test_loss}, iter_)
                writer.add_scalars('r2_loss', {f'{_ite}_r2_loss': test_loss}, iter_)
                writer.add_scalars('r2_score', {f'{_ite}_r2_score': np.mean(r2)}, iter_)

                # Save Weights
                if test_loss < best_r2_loss:
                    best_r2_loss = test_loss
                    utils.checkdir(f"{os.getcwd()}/saves/{args.dataset}/{now_time}/")
                    if args.dataset == "dimenet":
                        torch.save(model.state_dict(),
                                   f"{os.getcwd()}/saves/{args.dataset}/{now_time}/{_ite}_model_{args.prune_type}.pth")
                    else:
                        torch.save(model,
                                   f"{os.getcwd()}/saves/{args.dataset}/{now_time}/{_ite}_model_{args.prune_type}.pth.tar")
                if np.mean(r2) > best_mean_r2_score:
                    best_mean_r2_score = np.mean(r2)

            # Training
            loss = train(model, train_loader, optimizer, criterion_train)

            all_loss[iter_] = loss
            all_r2_loss[iter_] = best_r2_loss
            # writer['train_loss'].add_scalar(f'/Loss/{_ite}/train', loss, iter_)
            writer.add_scalars('loss', {f'{_ite}_train_loss': loss}, iter_)

            # Frequency for Printing relative_error and Loss
            if iter_ % args.print_freq == 0:
                pbar.set_description(
                    f'Train Epoch: {iter_}/{args.end_iter} Train Loss: {loss:.6f} test loss: {test_loss:.8f} '
                    f'test r2 loss: {r2_loss:.6f}  test r2 scores: {r2} beast mean r2: {best_mean_r2_score:.4f}')

        best_r2_scores[_ite] = np.mean(r2)

        # Dumping mask
        utils.checkdir(f"{os.getcwd()}/dumps/lt/{args.dataset}/{now_time}/")
        with open(f"{os.getcwd()}/dumps/lt/{args.dataset}/{now_time}/{args.prune_type}_mask_{comp1}.pkl",
                  'wb') as fp:
            pickle.dump(mask, fp)

        # Making variables into 0
        best_r2_loss = 100
        best_mean_r2_score = -100
        all_loss = np.zeros(args.end_iter, float)


# Function for Training
def train(model, train_loader, optimizer, criterion):
    EPS = 1e-28
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

        # Freezing Pruned weights by making their gradients Zero
        for name, p in model.named_parameters():
            if 'weight' in name:
                tensor = p.data.cpu().numpy()
                grad_tensor = p.grad.data.cpu().numpy()
                grad_tensor = np.where(tensor < EPS, 0, grad_tensor)
                p.grad.data = torch.from_numpy(grad_tensor).to(device)
        optimizer.step()
        total_loss += train_loss.item() / len(train_loader)
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
    r2 = []

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
            if isinstance(criterion, nn.CrossEntropyLoss):
                r2_l = test_loss
                r2 += [utils.accuracy(output, target, topk=(1,))[0].item()]
            else:
                r2_l = r2_loss(output, target).item()
                r2 = r2_score(target.cpu(), output.cpu(), multioutput='raw_values')
            test_losses += test_loss
            if debugging:
                print(f'test loss: {test_loss}, Quality loss: {r2_l}, Quality matrix: {r2}.(Quality means r2 or acc ')
        test_losses /= len(test_loader)
        if debugging:
            print(f'mean: test loss: {test_losses}, r2 score: {r2}, len: {len(test_loader)}')
    return test_losses, r2_l, r2


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
    step = 0
    for name, param in model.named_parameters():

        # We do not prune bias term
        if 'weight' in name:
            params = param.data.cpu().numpy()
            if len(params.shape) <= 2:
                tensor = np.sum(np.abs(params), axis=0)
            else:
                tensor = np.sum(np.abs(params), axis=(0, 2, 3))
            alive = tensor[np.nonzero(tensor)]  # flattened array of nonzero values
            percentile_value = np.percentile(abs(alive), percent)

            # Convert Tensors to numpy and calculate
            weight_dev = param.device
            new_mask = np.where(abs(tensor) < percentile_value, 0, mask[step])

            # Apply new weight and mask
            if len(params.shape) <= 2:
                new_mask_mul = np.expand_dims(new_mask, axis=0).astype(np.int32)
                new_mask_mul = np.repeat(new_mask_mul, repeats=params.shape[0], axis=0)
            else:
                new_mask_mul = np.expand_dims(new_mask, axis=(0, 2, 3)).astype(np.int32)
                new_mask_mul = np.repeat(new_mask_mul, repeats=params.shape[0], axis=0)
                new_mask_mul = np.repeat(new_mask_mul, repeats=params.shape[2], axis=2)
                new_mask_mul = np.repeat(new_mask_mul, repeats=params.shape[3], axis=3)
            param.data = torch.from_numpy(params * new_mask_mul).to(dtype=torch.float32, device=weight_dev)
            mask[step] = new_mask
            step += 1
    step = 0


# Function to make an empty mask of the same size as the model
def make_mask(model):
    global step
    global mask
    step = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            step = step + 1
    mask = [None] * step
    step = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            tensor = param.data.cpu().numpy()
            mask[step] = np.ones(tensor.shape[1])
            step = step + 1
    step = 0


def original_initialization(mask_temp, initial_state_dict):
    global model

    step = 0
    for name, param in model.named_parameters():
        if "weight" in name:
            weight_dev = param.device
            # Apply new weight and mask
            new_mask = mask_temp[step]
            if len(param.shape) <= 2:
                new_mask_mul = np.expand_dims(new_mask, axis=0).astype(np.int32)
                new_mask_mul = np.repeat(new_mask_mul, repeats=param.shape[0], axis=0)
            else:
                new_mask_mul = np.expand_dims(new_mask, axis=(0, 2, 3)).astype(np.int32)
                new_mask_mul = np.repeat(new_mask_mul, repeats=param.shape[0], axis=0)
                new_mask_mul = np.repeat(new_mask_mul, repeats=param.shape[2], axis=2)
                new_mask_mul = np.repeat(new_mask_mul, repeats=param.shape[3], axis=3)
            param.data = torch.from_numpy(new_mask_mul * initial_state_dict[name].cpu().numpy()).to(dtype=torch.float32, device=weight_dev)
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
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate")
    parser.add_argument("--batch_size", default=200, type=int)
    parser.add_argument("--start_iter", default=0, type=int)
    parser.add_argument("--end_iter", default=100, type=int)  # 100
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
    parser.add_argument("--prune_percent", default=10, type=int, help="Pruning percent")
    parser.add_argument("--prune_iterations", default=35, type=int, help="Pruning iterations count")
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

