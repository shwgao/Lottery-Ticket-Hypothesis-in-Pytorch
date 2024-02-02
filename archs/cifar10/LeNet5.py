import numpy as np
from copy import deepcopy
import torch.nn as nn
import torch.nn.functional as func


class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)
        self.fc1 = nn.Linear(50*5*5, 500)
        self.fc2 = nn.Linear(500, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = func.relu(self.conv1(x))
        x = func.max_pool2d(x, 2)
        x = func.relu(self.conv2(x))
        x = func.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def get_flat_fts(in_size, fts, device='cuda:0'):
    dummy_input = torch.ones(1, *in_size)
    dummy_input = dummy_input.to(device)
    f = fts(torch.autograd.Variable(dummy_input))
    print('conv_out_size: {}'.format(f.size()))
    return int(np.prod(f.size()[1:]))


class L0LeNet5(nn.Module):
    def __init__(self, num_classes, input_size=(3, 32, 32), conv_dims=(20, 50), fc_dims=500, use_reg=True,
                 N=50000, beta_ema=0., weight_decay=1, lambas=(1., 1., 1., 1.), local_rep=False,
                 temperature=2./3., device='cuda:0'):
        super(L0LeNet5, self).__init__()
        self.N = N
        assert(len(conv_dims) == 2)
        self.conv_dims = conv_dims
        self.fc_dims = fc_dims
        self.beta_ema = beta_ema
        self.weight_decay = weight_decay

        convs = [L0Conv2d(input_size[0], conv_dims[0], 5, droprate_init=0.5, temperature=temperature,
                          weight_decay=self.weight_decay, lamba=lambas, local_rep=local_rep, device=device, use_reg=use_reg),
                 nn.ReLU(), nn.MaxPool2d(2),
                 L0Conv2d(conv_dims[0], conv_dims[1], 5, droprate_init=0.5, temperature=temperature,
                          weight_decay=self.weight_decay, lamba=lambas, local_rep=local_rep, device=device, use_reg=use_reg),
                 nn.ReLU(), nn.MaxPool2d(2)]
        self.convs = nn.Sequential(*convs)

        self.convs = self.convs.to(device)

        flat_fts = get_flat_fts(input_size, self.convs, device=device)
        fcs = [L0Dense(flat_fts, 500, droprate_init=0.5, weight_decay=self.weight_decay, device=device,
                       lamba=lambas, local_rep=local_rep, temperature=temperature, use_reg=use_reg), nn.ReLU(),
               L0Dense(500, 128, droprate_init=0.5, weight_decay=self.weight_decay, device=device,
                       lamba=lambas, local_rep=local_rep, temperature=temperature, use_reg=use_reg), nn.ReLU(),
               L0Dense(128, num_classes, droprate_init=0.5, weight_decay=self.weight_decay, device=device,
                       lamba=lambas, local_rep=local_rep, temperature=temperature, use_reg=use_reg)]
        self.fcs = nn.Sequential(*fcs)

        self.layers = []
        for m in self.modules():
            if isinstance(m, L0Dense) or isinstance(m, L0Conv2d):
                self.layers.append(m)

        if beta_ema > 0.:
            print('Using temporal averaging with beta: {}'.format(beta_ema))
            self.avg_param = deepcopy(list(p.data for p in self.parameters()))
            if torch.cuda.is_available():
                self.avg_param = [a.cuda() for a in self.avg_param]
            self.steps_ema = 0.

    def forward(self, x):
        o = self.convs(x)
        o = o.view(o.size(0), -1)
        return self.fcs(o)

    def constrain_parameters(self):
        for layer in self.layers:
            layer.constrain_parameters()

    def regularization(self):
        regularization = 0.
        for layer in self.layers:
            regularization += - (1. / self.N) * layer.regularization()
        if torch.cuda.is_available():
            regularization = regularization.cuda()
        return regularization

    def get_exp_flops_l0(self):
        expected_flops, expected_l0 = 0., 0.
        for layer in self.layers:
            e_fl, e_l0 = layer.count_expected_flops_and_l0()
            expected_flops += e_fl
            expected_l0 += e_l0
        return expected_flops, expected_l0

    def update_ema(self):
        self.steps_ema += 1
        for p, avg_p in zip(self.parameters(), self.avg_param):
            avg_p.mul_(self.beta_ema).add_((1 - self.beta_ema) * p.data)

    def load_ema_params(self):
        for p, avg_p in zip(self.parameters(), self.avg_param):
            p.data.copy_(avg_p / (1 - self.beta_ema**self.steps_ema))

    def load_params(self, params):
        for p, avg_p in zip(self.parameters(), params):
            p.data.copy_(avg_p)

    def get_params(self):
        params = deepcopy(list(p.data for p in self.parameters()))
        return params


import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair as pair
from torch.nn.modules.utils import _triple as triple
from torch.autograd import Variable
from torch.nn import init
limit_a, limit_b, epsilon = -0.1, 1.1, 1e-6


class L0Dense(Module):
    """Implementation of L0 regularization for the input units of a fully connected layer"""

    def __init__(
        self,
        in_features,
        out_features,
        device="cuda:0",
        bias=True,
        weight_decay=1.0,
        droprate_init=0.5,
        temperature=2.0 / 3.0,
        lamba=1.0,
        local_rep=False,
        use_reg=True,
        **kwargs
    ):
        """
        :param in_features: Input dimensionality
        :param out_features: Output dimensionality
        :param bias: Whether we use a bias
        :param weight_decay: Strength of the L2 penalty
        :param droprate_init: Dropout rate that the L0 gates will be initialized to
        :param temperature: Temperature of the concrete distribution
        :param lamba: Strength of the L0 penalty
        :param local_rep: Whether we will use a separate gate sample per element in the minibatch
        """
        super(L0Dense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_prec = weight_decay
        self.weights = Parameter(torch.Tensor(in_features, out_features))
        self.qz_loga = Parameter(torch.Tensor(in_features))
        self.temperature = temperature
        self.droprate_init = droprate_init if droprate_init != 0.0 else 0.5
        self.lamba = lamba
        self.use_bias = False
        self.local_rep = local_rep
        self.device = device
        self.trained_z = None
        self.use_reg = use_reg
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
            self.use_bias = True
        self.floatTensor = (
            torch.FloatTensor
            if not torch.cuda.is_available()
            else torch.cuda.FloatTensor
        )
        self.reset_parameters()
        print(self)

    def reset_parameters(self):
        init.kaiming_normal_(self.weights, mode="fan_out")

        self.qz_loga.data.normal_(
            math.log(1 - self.droprate_init) - math.log(self.droprate_init), 1e-2
        )

        if self.use_bias:
            self.bias.data.fill_(0)

    def constrain_parameters(self, **kwargs):
        self.qz_loga.data.clamp_(min=math.log(1e-2), max=math.log(1e2))

    def cdf_qz(self, x):
        """Implements the CDF of the 'stretched' concrete distribution"""
        xn = (x - limit_a) / (limit_b - limit_a)  # scale
        logits = math.log(xn) - math.log(1 - xn)
        return torch.sigmoid(logits * self.temperature - self.qz_loga).clamp(
            min=epsilon, max=1 - epsilon
        )

    def quantile_concrete(self, x):
        """Implements the quantile, aka inverse CDF, of the 'stretched' concrete distribution"""
        x = x.to(self.device)
        y = torch.sigmoid(
            (torch.log(x) - torch.log(1 - x) + self.qz_loga) / self.temperature
        )
        return y * (limit_b - limit_a) + limit_a

    def _reg_w(self):
        """Expected L0 norm under the stochastic gates, takes into account and re-weights also a potential L2 penalty"""
        logpw_col = torch.sum(-(0.5 * self.prior_prec * self.weights.pow(2)) - self.lamba, 1)
        logpw = torch.sum((1 - self.cdf_qz(0)) * logpw_col)
        logpb = (
            0
            if not self.use_bias
            else -torch.sum(0.5 * self.prior_prec * self.bias.pow(2))
        )
        # print("logpw:",logpw.item(),"logpb:",logpb.item())
        return logpw + logpb

    def _reg_w2(self):
        budget = 0.2
        logpw_col = torch.sum(-(0.5 * self.prior_prec * self.weights.pow(2)) - self.lamba, 1)
        qz = 1-self.cdf_qz(0)
        bd_qz = nn.ReLU()(qz-budget)
        logpw = torch.sum(bd_qz * logpw_col)
        logpb = (
            0
            if not self.use_bias
            else -torch.sum(0.5 * self.prior_prec * self.bias.pow(2))
        )
        # print("logpw:",logpw.item(),"logpb:",logpb.item())
        return logpw + logpb

    def regularization(self):
        if self.use_reg:
            return self._reg_w()
        else:
            return 0

    def count_expected_flops_and_l0(self):
        """Measures the expected floating point operations (FLOPs) and the expected L0 norm"""
        # dim_in multiplications and dim_in - 1 additions for each output neuron for the weights
        # + the bias addition for each neuron
        # total_flops = (2 * in_features - 1) * out_features + out_features
        if self.use_reg:
            # ppos = torch.sum(1 - self.cdf_qz(0))
            z = self.sample_z(1, sample=False)
            ppos = torch.sum(z).item()
        else:
            ppos = self.in_features
        expected_flops = (2 * ppos - 1) * self.out_features
        expected_l0 = ppos * self.out_features
        if self.use_bias:
            expected_flops += self.out_features
            expected_l0 += self.out_features
        # return expected_flops.data[0], expected_l0.data[0]
        return expected_flops, expected_l0

    def get_eps(self, size):
        """Uniform random numbers for the concrete distribution"""
        eps = torch.FloatTensor(size).to(self.device).uniform_(epsilon, 1 - epsilon)
        eps = Variable(eps)
        return eps

    def sample_z(self, batch_size, sample=True):
        """Sample the hard-concrete gates for training and use a deterministic value for testing"""
        if sample:
            eps = self.get_eps((batch_size, self.in_features))
            z = self.quantile_concrete(eps)
            return F.hardtanh(z, min_val=0, max_val=1)
        else:  # mode
            pi = (
                torch.sigmoid(self.qz_loga)
                .view(1, self.in_features)
                # .expand(batch_size, self.in_features)
            )
            return F.hardtanh(pi * (limit_b - limit_a) + limit_a, min_val=0, max_val=1)

    def sample_weights(self):
        z = self.quantile_concrete(
            self.get_eps((self.in_features))
        )
        mask = F.hardtanh(z, min_val=0, max_val=1)
        return mask.view(self.in_features, 1) * self.weights

    def update_z(self, sample=False):
        z = self.sample_z(1, sample)
        z[z < 0.1] = 0
        self.trained_z = z

    def forward(self, input_):
        if self.use_reg:
            if self.local_rep or not self.training:
                z = self.sample_z(input_.size(0), sample=self.training)
                xin = input_.mul(z)
                # xin = input.mul(self.trained_z)
                output = torch.matmul(xin, self.weights)  # output = xin.mm(self.weights)
            else:
                weights = self.sample_weights()
                output = torch.matmul(input_, weights)  # output = input_.mm(weights)

        else:
            output = input_.mm(self.weights)

        if self.use_bias:
            output.add_(self.bias)
        return output

    def __repr__(self):
        s = (
            "{name}({in_features} -> {out_features}, droprate_init={droprate_init}, "
            "lamba={lamba}, temperature={temperature}, weight_decay={prior_prec}, "
            "local_rep={local_rep}"
        )
        if not self.use_bias:
            s += ", bias=False"
        s += ")"
        return s.format(name=self.__class__.__name__, **self.__dict__)


class L0Conv2d(Module):
    """Implementation of L0 regularization for the feature maps of a convolutional layer"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        droprate_init=0.5,
        temperature=2.0 / 3.0,
        weight_decay=1.0,
        lamba=1.0,
        local_rep=False,
        use_reg=True,
        device="cpu",
        **kwargs
    ):
        """
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param kernel_size: Size of the kernel
        :param stride: Stride for the convolution
        :param padding: Padding for the convolution
        :param dilation: Dilation factor for the convolution
        :param groups: How many groups we will assume in the convolution
        :param bias: Whether we will use a bias
        :param droprate_init: Dropout rate that the L0 gates will be initialized to
        :param temperature: Temperature of the concrete distribution
        :param weight_decay: Strength of the L2 penalty
        :param lamba: Strength of the L0 penalty
        :param local_rep: Whether we will use a separate gate sample per element in the minibatch
        """
        super(L0Conv2d, self).__init__()
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.device = device
        self.kernel_size = pair(kernel_size)
        self.stride = pair(stride)
        self.padding = pair(padding)
        self.dilation = pair(dilation)
        self.output_padding = pair(0)
        self.groups = groups
        self.prior_prec = weight_decay
        self.lamba = lamba
        self.use_reg = use_reg
        self.droprate_init = droprate_init if droprate_init != 0.0 else 0.5
        self.temperature = temperature
        self.floatTensor = (
            torch.FloatTensor
            if not torch.cuda.is_available()
            else torch.cuda.FloatTensor
        )
        self.use_bias = False
        self.weights = Parameter(
            torch.Tensor(out_channels, in_channels // groups, *self.kernel_size)
        )
        self.qz_loga = Parameter(torch.Tensor(out_channels))
        self.dim_z = out_channels
        self.input_shape = None
        self.local_rep = local_rep

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
            self.use_bias = True

        self.reset_parameters()
        print(self)

    def reset_parameters(self):
        init.kaiming_normal(self.weights, mode="fan_in")

        self.qz_loga.data.normal_(
            math.log(1 - self.droprate_init) - math.log(self.droprate_init), 1e-2
        )

        if self.use_bias:
            self.bias.data.fill_(0)

    def constrain_parameters(self, **kwargs):
        self.qz_loga.data.clamp_(min=math.log(1e-2), max=math.log(1e2))

    def cdf_qz(self, x):
        """Implements the CDF of the 'stretched' concrete distribution"""
        xn = (x - limit_a) / (limit_b - limit_a)
        logits = math.log(xn) - math.log(1 - xn)
        return torch.sigmoid(logits * self.temperature - self.qz_loga).clamp(
            min=epsilon, max=1 - epsilon
        )

    def quantile_concrete(self, x):
        """Implements the quantile, aka inverse CDF, of the 'stretched' concrete distribution"""
        y = torch.sigmoid(
            (torch.log(x) - torch.log(1 - x) + self.qz_loga) / self.temperature
        )
        return y * (limit_b - limit_a) + limit_a

    def _reg_w(self):
        """Expected L0 norm under the stochastic gates, takes into account and re-weights also a potential L2 penalty"""
        q0 = self.cdf_qz(0)
        logpw_col = (
            torch.sum(-(0.5 * self.prior_prec * self.weights.pow(2)) - self.lamba, 3)
            .sum(2)
            .sum(1)
        )
        logpw = torch.sum((1 - q0) * logpw_col)
        logpb = (
            0
            if not self.use_bias
            else -torch.sum(
                (1 - q0) * (0.5 * self.prior_prec * self.bias.pow(2) - self.lamba)
            )
        )
        return logpw + logpb

    def regularization(self):
        return self._reg_w()

    def count_expected_flops_and_l0(self):
        """Measures the expected floating point operations (FLOPs) and the expected L0 norm"""
        if self.use_reg:
            # ppos = torch.sum(1 - self.cdf_qz(0))
            ppos = torch.sum(self.sample_z(1, sample=False)).item()
        else:
            ppos = self.out_channels
        n = (
            self.kernel_size[0] * self.kernel_size[1] * self.in_channels
        )  # vector_length
        flops_per_instance = n + (n - 1)  # (n: multiplications and n-1: additions)

        num_instances_per_filter = (
            (self.input_shape[1] - self.kernel_size[0] + 2 * self.padding[0])
            / self.stride[0]
        ) + 1  # for rows
        num_instances_per_filter *= (
            (self.input_shape[2] - self.kernel_size[1] + 2 * self.padding[1])
            / self.stride[1]
        ) + 1  # multiplying with cols

        flops_per_filter = num_instances_per_filter * flops_per_instance
        expected_flops = flops_per_filter * ppos  # multiply with number of filters
        expected_l0 = n * ppos

        if self.use_bias:
            # since the gate is applied to the output we also reduce the bias computation
            expected_flops += num_instances_per_filter * ppos
            expected_l0 += ppos

        # return expected_flops.data[0], expected_l0.data[0]
        return expected_flops, expected_l0

    def get_eps(self, size):
        """Uniform random numbers for the concrete distribution"""
        eps = torch.empty(size, device=self.device).uniform_(epsilon, 1 - epsilon)
        eps = Variable(eps)
        return eps

    def sample_z(self, batch_size, sample=True):
        """Sample the hard-concrete gates for training and use a deterministic value for testing"""
        if sample:
            eps = self.get_eps((batch_size, self.dim_z))
            z = self.quantile_concrete(eps).view(batch_size, self.dim_z, 1, 1)
            return F.hardtanh(z, min_val=0, max_val=1)
        else:  # mode
            pi = torch.sigmoid(self.qz_loga).view(1, self.dim_z, 1, 1)
            return F.hardtanh(pi * (limit_b - limit_a) + limit_a, min_val=0, max_val=1)

    def sample_weights(self):
        z = self.quantile_concrete(self.get_eps((self.dim_z))).view(
            self.dim_z, 1, 1, 1
        )
        return F.hardtanh(z, min_val=0, max_val=1) * self.weights

    def forward(self, input_):
        if self.input_shape is None:
            self.input_shape = input_.size()
        b = None if not self.use_bias else self.bias
        if self.use_reg:
            if self.local_rep or not self.training:
                output = F.conv2d(
                    input_,
                    self.weights,
                    b,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.groups,
                )
                z = self.sample_z(output.size(0), sample=self.training)
                return output.mul(z)
            else:
                weights = self.sample_weights()
                output = F.conv2d(
                    input_,
                    weights,
                    None,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.groups,
                )
                return output
        else:
            return F.conv2d(
                input_,
                self.weights,
                b,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )

    def __repr__(self):
        s = (
            "{name}({in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, "
            "droprate_init={droprate_init}, temperature={temperature}, prior_prec={prior_prec}, "
            "lamba={lamba}, local_rep={local_rep}"
        )
        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"
        if self.output_padding != (0,) * len(self.output_padding):
            s += ", output_padding={output_padding}"
        if self.groups != 1:
            s += ", groups={groups}"
        if not self.use_bias:
            s += ", bias=False"
        s += ")"
        return s.format(name=self.__class__.__name__, **self.__dict__)


class L0Conv3d(Module):
    """Implementation of L0 regularization for the feature maps of a convolutional layer"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        droprate_init=0.5,
        temperature=2.0 / 3.0,
        weight_decay=1.0,
        lamba=1.0,
        local_rep=False,
        use_reg=True,
        device="cpu",
        **kwargs
    ):
        """
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param kernel_size: Size of the kernel
        :param stride: Stride for the convolution
        :param padding: Padding for the convolution
        :param dilation: Dilation factor for the convolution
        :param groups: How many groups we will assume in the convolution
        :param bias: Whether we will use a bias
        :param droprate_init: Dropout rate that the L0 gates will be initialized to
        :param temperature: Temperature of the concrete distribution
        :param weight_decay: Strength of the L2 penalty
        :param lamba: Strength of the L0 penalty
        :param local_rep: Whether we will use a separate gate sample per element in the minibatch
        """
        super(L0Conv3d, self).__init__()
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = triple(kernel_size)
        self.stride = triple(stride)
        self.padding = triple(padding)
        self.dilation = triple(dilation)
        self.output_padding = triple(0)
        self.groups = groups
        self.prior_prec = weight_decay
        self.lamba = lamba
        self.droprate_init = droprate_init if droprate_init != 0.0 else 0.5
        self.temperature = temperature
        self.device = device
        self.floatTensor = (
            torch.FloatTensor
            if not torch.cuda.is_available()
            else torch.cuda.FloatTensor
        )
        self.use_bias = False
        self.weights = Parameter(
            torch.Tensor(out_channels, in_channels // groups, *self.kernel_size)
        )
        self.qz_loga = Parameter(torch.Tensor(out_channels))
        self.dim_z = out_channels
        self.input_shape = None
        self.local_rep = local_rep
        self.use_reg = use_reg

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
            self.use_bias = True

        self.reset_parameters()
        print(self)

    def reset_parameters(self):
        init.kaiming_normal(self.weights, mode="fan_in")

        self.qz_loga.data.normal_(
            math.log(1 - self.droprate_init) - math.log(self.droprate_init), 1e-2
        )

        if self.use_bias:
            self.bias.data.fill_(0)

    def constrain_parameters(self, **kwargs):
        self.qz_loga.data.clamp_(min=math.log(1e-2), max=math.log(1e2))

    def cdf_qz(self, x):
        """Implements the CDF of the 'stretched' concrete distribution"""
        xn = (x - limit_a) / (limit_b - limit_a)
        logits = math.log(xn) - math.log(1 - xn)
        return torch.sigmoid(logits * self.temperature - self.qz_loga).clamp(
            min=epsilon, max=1 - epsilon
        )

    def quantile_concrete(self, x):
        """Implements the quantile, aka inverse CDF, of the 'stretched' concrete distribution"""
        y = torch.sigmoid(
            (torch.log(x) - torch.log(1 - x) + self.qz_loga) / self.temperature
        )
        return y * (limit_b - limit_a) + limit_a

    def _reg_w(self):
        """Expected L0 norm under the stochastic gates, takes into account and re-weights also a potential L2 penalty"""
        q0 = self.cdf_qz(0)
        logpw_col = (
            torch.sum(-(0.5 * self.prior_prec * self.weights.pow(2)) - self.lamba, 4)
            .sum(3).sum(2).sum(1))
        logpw = torch.sum((1 - q0) * logpw_col)
        logpb = (
            0
            if not self.use_bias
            else -torch.sum(
                (1 - q0) * (0.5 * self.prior_prec * self.bias.pow(2) - self.lamba)
            )
        )
        return logpw + logpb

    def regularization(self):
        if not self.use_reg:
            return 0
        return self._reg_w()

    def count_expected_flops_and_l0(self):
        if self.use_reg:
            ppos = torch.sum(self.sample_z(1, sample=False))
        else:
            ppos = self.out_channels
        n = (self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2] * self.in_channels)
        flops_per_instance = n + (n - 1)
        num_instances_per_filter = ((self.input_shape[1] - self.kernel_size[0] + 2 * self.padding[0]) / self.stride[0]) + 1
        num_instances_per_filter *= ((self.input_shape[2] - self.kernel_size[1] + 2 * self.padding[1]) / self.stride[1]) + 1
        num_instances_per_filter *= ((self.input_shape[3] - self.kernel_size[2] + 2 * self.padding[2]) / self.stride[2]) + 1
        flops_per_filter = num_instances_per_filter * flops_per_instance
        expected_flops = flops_per_filter * ppos
        expected_l0 = n * ppos
        if self.use_bias:
            expected_flops += num_instances_per_filter * ppos
            expected_l0 += ppos
        return expected_flops, expected_l0

    def get_eps(self, size):
        """Uniform random numbers for the concrete distribution"""
        eps = torch.empty(size, device=self.device).uniform_(epsilon, 1 - epsilon)
        eps = Variable(eps)
        return eps

    def sample_z(self, batch_size, sample=True):
        """Sample the hard-concrete gates for training and use a deterministic value for testing"""
        if sample:
            eps = self.get_eps((batch_size, self.dim_z))
            z = self.quantile_concrete(eps).view(batch_size, self.dim_z, 1, 1, 1)
            return F.hardtanh(z, min_val=0, max_val=1)
        else:  # mode
            pi = F.sigmoid(self.qz_loga).view(1, self.dim_z, 1, 1, 1)
            return F.hardtanh(pi * (limit_b - limit_a) + limit_a, min_val=0, max_val=1)

    def sample_weights(self):
        z = self.quantile_concrete(self.get_eps([self.dim_z])).view(
            self.dim_z, 1, 1, 1, 1
        )
        return F.hardtanh(z, min_val=0, max_val=1) * self.weights

    def forward(self, input_):
        if not self.use_reg:
            return F.conv3d(
                input_,
                self.weights,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )

        if self.input_shape is None:
            self.input_shape = input_.size()
        b = None if not self.use_bias else self.bias
        if self.local_rep or not self.training:
            output = F.conv3d(
                input_,
                self.weights,
                b,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
            z = self.sample_z(output.size(0), sample=self.training)
            return output.mul(z)
        else:
            weights = self.sample_weights()
            output = F.conv3d(
                input_,
                weights,
                None,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
            return output

    def __repr__(self):
        s = (
            "{name}({in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, "
            "droprate_init={droprate_init}, temperature={temperature}, prior_prec={prior_prec}, "
            "lamba={lamba}, local_rep={local_rep}"
        )
        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"
        if self.output_padding != (0,) * len(self.output_padding):
            s += ", output_padding={output_padding}"
        if self.groups != 1:
            s += ", groups={groups}"
        if not self.use_bias:
            s += ", bias=False"
        s += ")"
        return s.format(name=self.__class__.__name__, **self.__dict__)
