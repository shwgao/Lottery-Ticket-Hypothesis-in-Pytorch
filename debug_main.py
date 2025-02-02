"""
This file is used to train the tensorflow model like dimenet_pp.
"""
# Importing Libraries
import argparse
import datetime
import numpy as np
import yaml
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import os
from tensorboardX import SummaryWriter
import seaborn as sns
import pickle

# Custom Libraries
import utils
from archs.puremd.dimenet_pp import get_trainer, extract_dense_layers

from load_dataset import load_data

now_time = datetime.datetime.now().strftime("%m-%d-%H")
# Tensorboard initialization
# writer = SummaryWriter()
# writer = {'train_loss': SummaryWriter(f'runs/{now_time}/train_loss'),
#           'test_loss': SummaryWriter(f'runs/{now_time}/test_loss'),
#           'relative_error': SummaryWriter(f'runs/{now_time}/relative_error')}

# Plotting Style
sns.set_style('darkgrid')

with open('./archs/puremd/config_pp.yaml', 'r') as c:
    config = yaml.safe_load(c)


# Main
def main(args, ITE=0):
    writer = SummaryWriter(f'runs/{args.dataset}/{now_time}/')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    reinit = True if args.prune_type == "reinit" else False

    # Data Loader
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    if args.dataset == "mnist":
        traindataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
        testdataset = datasets.MNIST('../data', train=False, transform=transform)
        from archs.mnist import AlexNet, LeNet5, fc1, vgg, resnet
        # If you want to add extra datasets paste here
    elif args.dataset == "puremd":
        train_loader, validation = load_data(args)
        from archs.puremd import dimenet_pp

    else:
        print("\nWrong Dataset choice \n")
        exit()

    # Importing Network Architecture
    global model, LeNet5
    if args.arch_type == "fc1":
        if args.dataset == "puremd":
            model = dimenet_pp.DimeNetPP()
        else:
            model = fc1.fc1().to(device)
    elif args.arch_type == "lenet5":
        model = LeNet5.LeNet5().to(device)
    elif args.arch_type == "alexnet":
        model = AlexNet.AlexNet().to(device)
    elif args.arch_type == "vgg16":
        model = vgg.vgg16().to(device)
    elif args.arch_type == "resnet18":
        model = resnet.resnet18().to(device)
    elif args.arch_type == "densenet121":
        model = densenet.densenet121().to(device)

    else:
        print("\nWrong Model choice\n")
        exit()

    # Weight Initialization
    # model.apply(weight_init) #  model were initialized during the model definition

    # Copying and Saving Initial State
    # initial_state_dict = copy.deepcopy(model.state_dict())
    initial_state_dict = [layer.get_weights() for layer in model.layers if len(layer.get_weights()) > 0]
    utils.checkdir(f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/")
    # model.save(f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/initial_state_dict_{args.prune_type}", save_format="tf")

    # Making Initial Mask
    inputs, targets = next(train_loader['dataset_iter'])
    model(inputs)
    make_mask(model)

    trainer = get_trainer(model)

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
            original_initialization(mask, initial_state_dict)
        print(f"\n--- Pruning Level [{ITE}:{_ite}/{ITERATION}]: ---")

        # Print the table of Nonzeros in each layer
        comp1 = utils.print_nonzeros_tf(model)
        comp[_ite] = comp1
        pbar = tqdm(range(args.end_iter))

        for iter_ in pbar:

            # Frequency for Testing
            if iter_ % args.valid_freq == 0:
                test_loss, relative_error = test(validation, trainer)
                # writer['test_loss'].add_scalar(f'/Loss/{_ite}/test_loss', test_loss, iter_)
                # writer['relative_error'].add_scalar(f'/Loss/{_ite}/relative_error', relative_error, iter_)
                writer.add_scalars('test_loss', {f'{_ite}': test_loss}, iter_)
                writer.add_scalars('relative_error', {f'{_ite}': relative_error}, iter_)

                # Save Weights
                if relative_error < best_relative_error:
                    best_relative_error = relative_error
                    utils.checkdir(f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/")
                    model.save(f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/{_ite}_model_{args.prune_type}",
                               save_format="tf")
                if test_loss < bestr2:
                    bestr2 = test_loss

            # Training
            loss = train(train_loader, trainer)
            all_loss[iter_] = loss
            all_relative_error[iter_] = relative_error
            # writer['train_loss'].add_scalar(f'/Loss/{_ite}/train', loss, iter_)
            writer.add_scalars('train_loss', {f'{_ite}': loss}, iter_)

            # Frequency for Printing relative_error and Loss
            if iter_ % args.print_freq == 0:
                pbar.set_description(
                    f'Train Epoch: {iter_}/{args.end_iter} Loss: {loss:.6f} R2: {bestr2:.8f} relative_error: {relative_error:.4f}% Best relative error: {best_relative_error:.4f}%')

        bestre[_ite] = best_relative_error

        # Plotting Loss (Training), relative_error (Testing), Iteration Curve
        # NOTE Loss is computed for every iteration while relative_error is computed only for every {args.valid_freq} iterations. Therefore relative_error saved is constant during the uncomputed iterations.
        # NOTE Normalized the relative_error to [0,100] for ease of plotting.
        plt.plot(np.arange(1, (args.end_iter) + 1),
                 100 * (all_loss - np.min(all_loss)) / np.ptp(all_loss).astype(float), c="blue", label="Loss")
        plt.plot(np.arange(1, (args.end_iter) + 1), all_relative_error, c="red", label="relative_error")
        plt.title(f"Loss Vs relative_error Vs Iterations ({args.dataset},{args.arch_type})")
        plt.xlabel("Iterations")
        plt.ylabel("Loss and relative_error")
        plt.legend()
        plt.grid(color="gray")
        utils.checkdir(f"{os.getcwd()}/plots/lt/{args.arch_type}/{args.dataset}/")
        plt.savefig(
            f"{os.getcwd()}/plots/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_LossVsRelative_error_{comp1}.png",
            dpi=1200)
        plt.close()

        # Dump Plot values
        utils.checkdir(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/")
        all_loss.dump(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_all_loss_{comp1}.dat")
        all_relative_error.dump(
            f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_all_relative_error_{comp1}.dat")

        # Dumping mask
        utils.checkdir(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/")
        with open(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_mask_{comp1}.pkl",
                  'wb') as fp:
            pickle.dump(mask, fp)

        # Making variables into 0
        best_relative_error = 100
        bestr2 = 100
        all_loss = np.zeros(args.end_iter, float)
        all_relative_error = np.zeros(args.end_iter, float)

    # Dumping Values for Plotting
    utils.checkdir(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/")
    comp.dump(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_compression.dat")
    bestre.dump(f"{os.getcwd()}/dumps/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_bestrelativeerror.dat")

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
    utils.checkdir(f"{os.getcwd()}/plots/lt/{args.arch_type}/{args.dataset}/")
    plt.savefig(f"{os.getcwd()}/plots/lt/{args.arch_type}/{args.dataset}/{args.prune_type}_reeVsWeights.png", dpi=1200)
    plt.close()


# Function for Training
def train_torch(model, train_loader, optimizer, criterion):
    EPS = 1e-6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    for batch_idx, (imgs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        # imgs, targets = next(train_loader)
        imgs, targets = imgs.to(device), targets.to(device)
        output = model(imgs)
        train_loss = criterion(output, targets)
        train_loss.backward()

        # Freezing Pruned weights by making their gradients Zero
        for name, p in model.named_parameters():
            if 'weight' in name:
                tensor = p.data.cpu().numpy()
                grad_tensor = p.grad.data.cpu().numpy()
                grad_tensor = np.where(tensor < EPS, 0, grad_tensor)
                p.grad.data = torch.from_numpy(grad_tensor).to(device)
        optimizer.step()
    return train_loss.item()


def train(train_provider, trainer):
    losses = []
    sample_numbers = []
    relative_errors = []
    for i in range(int(np.ceil(config['num_train'] / config['batch_size']))):
        loss, nsamples = trainer.train_on_batch(train_provider['dataset_iter'], mask)
        losses.append(loss)
        sample_numbers.append(nsamples)
    return np.average(losses, weights=sample_numbers)


# Function for Testing
def test_torch(model, test_loader, criterion):
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
        relative_error = 100. * torch.mean(torch.abs(output - target) / torch.abs(target + 1e-8))
    return test_loss, relative_error


def test(validation, trainer):
    losses = []
    sample_numbers = []
    relative_errors = []
    # Save backup variables and load averaged variables
    trainer.save_variable_backups()
    trainer.load_averaged_variables()

    for i in range(int(np.ceil(config['num_valid'] / config['batch_size']))):
        loss, nsamples, relative_error = trainer.test_on_batch(validation['dataset_iter'])
        losses.append(loss)
        sample_numbers.append(nsamples)
        relative_errors.append(relative_error)

    trainer.restore_variable_backups()
    return np.average(np.array(losses), weights=np.array(sample_numbers)), np.average(np.array(relative_errors))
    # return np.mean(losses), np.mean(relative_errors)

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
def prune_by_percentile_torch(percent, resample=False, reinit=False, **kwargs):
    global step
    global mask
    global model

    # Calculate percentile value
    step = 0
    for name, param in model.named_parameters():

        # We do not prune bias term
        if 'weight' in name:
            tensor = param.data.cpu().numpy()
            alive = tensor[np.nonzero(tensor)]  # flattened array of nonzero values
            percentile_value = np.percentile(abs(alive), percent)

            # Convert Tensors to numpy and calculate
            weight_dev = param.device
            new_mask = np.where(abs(tensor) < percentile_value, 0, mask[step])

            # Apply new weight and mask
            param.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
            mask[step] = new_mask
            step += 1
    step = 0


def prune_by_percentile(percent, model, mask):
    global step
    step = 0
    for layer in model.dense_layers:
        weights = layer.get_weights()[0]
        alive = weights[np.nonzero(weights)]
        percentile_value = np.percentile(abs(alive), percent)

        # Create new mask
        new_mask = np.where(abs(weights) < percentile_value, 0, mask[layer.name])

        # Apply new mask to the weights
        layer.set_weights([weights * new_mask, layer.get_weights()[1]])
        mask[layer.name] = new_mask
        step += 1
    step = 0


# Function to make an empty mask of the same size as the model
def make_mask_torch(model):
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
            mask[step] = np.ones_like(tensor)
            step = step + 1
    step = 0


def make_mask(model):
    global mask
    mask = {}
    for layer in model.dense_layers:
        weights = layer.get_weights()[0]
        mask[layer.name] = np.ones_like(weights)
    return mask


def original_initialization_torch(mask_temp, initial_state_dict):
    global model

    step = 0
    for name, param in model.named_parameters():
        if "weight" in name:
            weight_dev = param.device
            param.data = torch.from_numpy(mask_temp[step] * initial_state_dict[name].cpu().numpy()).to(weight_dev)
            step = step + 1
        if "bias" in name:
            param.data = initial_state_dict[name]


def original_initialization(mask, initial_weights):
    for layer in model.dense_layers:
        layer.set_weights([mask[layer.name] * initial_weights[layer.name], layer.get_weights()[1]])


def r2_loss(output, target):
    return torch.sum(torch.square(target - output)) / torch.sum(torch.square(target - torch.mean(target)))


if __name__ == "__main__":
    # from gooey import Gooey
    # @Gooey

    # Arguement Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=1.2e-3, type=float, help="Learning rate")
    parser.add_argument("--batch_size", default=3000, type=int)
    parser.add_argument("--start_iter", default=0, type=int)
    parser.add_argument("--end_iter", default=50, type=int)  # 100
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--valid_freq", default=1, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prune_type", default="lt", type=str, help="lt | reinit")
    parser.add_argument("--gpu", default="2", type=str)
    parser.add_argument("--dataset", default="puremd", type=str,
                        help="mnist | cifar10 | fashionmnist | cifar100 | CFD | fluidanimation | puremd | cosmoflow")
    parser.add_argument("--arch_type", default="fc1", type=str,
                        help="fc1 | lenet5 | alexnet | vgg16 | resnet18 | densenet121")
    parser.add_argument("--prune_percent", default=20, type=int, help="Pruning percent")
    parser.add_argument("--prune_iterations", default=10, type=int, help="Pruning iterations count")
    parser.add_argument("--device", default="cuda", type=str, help="cuda | cpu")

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    # FIXME resample
    resample = False
    if args.dataset == 'cosmoflow':
        args.batch_size = 64

    # Looping Entire process
    # for i in range(0, 5):
    main(args, ITE=1)


# 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