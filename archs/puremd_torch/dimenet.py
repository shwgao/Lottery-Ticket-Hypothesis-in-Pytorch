from functools import partial
from typing import Callable, Optional

import torch
import torch.nn as nn
import numpy as np
from scipy import special as sp
from scipy.optimize import brentq
from torch import Tensor
from math import sqrt
from torch.nn import Embedding, Linear

try:
    import sympy as sym
except ImportError:
    sym = None


def Jn(r, n):
    return np.sqrt(np.pi / (2 * r)) * sp.jv(n + 0.5, r)


def Jn_zeros(n, k):
    zerosj = np.zeros((n, k), dtype='float32')
    zerosj[0] = np.arange(1, k + 1) * np.pi
    points = np.arange(1, k + n) * np.pi
    racines = np.zeros(k + n - 1, dtype='float32')
    for i in range(1, n):
        for j in range(k + n - 1 - i):
            foo = brentq(Jn, points[j], points[j + 1], (i, ))
            racines[j] = foo
        points = racines
        zerosj[i][:k] = racines[:k]

    return zerosj


def spherical_bessel_formulas(n):
    x = sym.symbols('x')

    f = [sym.sin(x) / x]
    a = sym.sin(x) / x
    for i in range(1, n):
        b = sym.diff(a, x) / x
        f += [sym.simplify(b * (-x)**i)]
        a = sym.simplify(b)
    return f


def bessel_basis(n, k):
    zeros = Jn_zeros(n, k)
    normalizer = []
    for order in range(n):
        normalizer_tmp = []
        for i in range(k):
            normalizer_tmp += [0.5 * Jn(zeros[order, i], order + 1)**2]
        normalizer_tmp = 1 / np.array(normalizer_tmp)**0.5
        normalizer += [normalizer_tmp]

    f = spherical_bessel_formulas(n)
    x = sym.symbols('x')
    bess_basis = []
    for order in range(n):
        bess_basis_tmp = []
        for i in range(k):
            bess_basis_tmp += [
                sym.simplify(normalizer[order][i] *
                             f[order].subs(x, zeros[order, i] * x))
            ]
        bess_basis += [bess_basis_tmp]
    return bess_basis


def sph_harm_prefactor(k, m):
    return ((2 * k + 1) * np.math.factorial(k - abs(m)) /
            (4 * np.pi * np.math.factorial(k + abs(m))))**0.5


def associated_legendre_polynomials(k, zero_m_only=True):
    r"""Helper function to calculate Y_l^m."""
    z = sym.symbols('z')
    P_l_m = [[0] * (j + 1) for j in range(k)]

    P_l_m[0][0] = 1
    if k > 0:
        P_l_m[1][0] = z

        for j in range(2, k):
            # Use the property of Eq (7) in
            # https://mathworld.wolfram.com/AssociatedLegendrePolynomial.html:
            P_l_m[j][0] = sym.simplify(((2 * j - 1) * z * P_l_m[j - 1][0] -
                                        (j - 1) * P_l_m[j - 2][0]) / j)
        if not zero_m_only:
            for i in range(1, k):
                P_l_m[i][i] = sym.simplify(
                    (1 - 2 * i) * P_l_m[i - 1][i - 1] * (1 - z**2)**0.5)
                if i + 1 < k:
                    # Use the property of Eq (11) in
                    # https://mathworld.wolfram.com/AssociatedLegendrePolynomial.html:
                    P_l_m[i + 1][i] = sym.simplify(
                        (2 * i + 1) * z * P_l_m[i][i])
                for j in range(i + 2, k):
                    # Use the property of Eq (7) in
                    # https://mathworld.wolfram.com/AssociatedLegendrePolynomial.html:
                    P_l_m[j][i] = sym.simplify(
                        ((2 * j - 1) * z * P_l_m[j - 1][i] -
                         (i + j - 1) * P_l_m[j - 2][i]) / (j - i))

    return P_l_m


def real_sph_harm(k, zero_m_only=True, spherical_coordinates=True):
    if not zero_m_only:
        S_m = [0]
        C_m = [1]
        for i in range(1, k):
            x = sym.symbols('x')
            y = sym.symbols('y')
            S_m += [x * S_m[i - 1] + y * C_m[i - 1]]
            C_m += [x * C_m[i - 1] - y * S_m[i - 1]]

    P_l_m = associated_legendre_polynomials(k, zero_m_only)
    if spherical_coordinates:
        theta = sym.symbols('theta')
        z = sym.symbols('z')
        for i in range(len(P_l_m)):
            for j in range(len(P_l_m[i])):
                if not isinstance(P_l_m[i][j], int):
                    P_l_m[i][j] = P_l_m[i][j].subs(z, sym.cos(theta))
        if not zero_m_only:
            phi = sym.symbols('phi')
            for i in range(len(S_m)):
                S_m[i] = S_m[i].subs(x,
                                     sym.sin(theta) * sym.cos(phi)).subs(
                                         y,
                                         sym.sin(theta) * sym.sin(phi))
            for i in range(len(C_m)):
                C_m[i] = C_m[i].subs(x,
                                     sym.sin(theta) * sym.cos(phi)).subs(
                                         y,
                                         sym.sin(theta) * sym.sin(phi))

    Y_func_l_m = [['0'] * (2 * j + 1) for j in range(k)]
    for i in range(k):
        Y_func_l_m[i][0] = sym.simplify(sph_harm_prefactor(i, 0) * P_l_m[i][0])

    if not zero_m_only:
        for i in range(1, k):
            for j in range(1, i + 1):
                Y_func_l_m[i][j] = sym.simplify(
                    2**0.5 * sph_harm_prefactor(i, j) * C_m[j] * P_l_m[i][j])
        for i in range(1, k):
            for j in range(1, i + 1):
                Y_func_l_m[i][-j] = sym.simplify(
                    2**0.5 * sph_harm_prefactor(i, -j) * S_m[j] * P_l_m[i][j])

    return Y_func_l_m


def glorot_orthogonal(tensor, scale):
    if tensor is not None:
        torch.nn.init.orthogonal_(tensor.data)
        scale /= ((tensor.size(-2) + tensor.size(-1)) * tensor.var())
        tensor.data *= scale.sqrt()


def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src.long()


def scatter(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    dim_size: int = None,
) -> torch.Tensor:
    index = broadcast(index, src, dim)
    size = list(src.size())
    if dim_size is not None:
        size[dim] = dim_size
    elif index.numel() == 0:
        size[dim] = 0
    else:
        size[dim] = int(index.max()) + 1
    out = torch.zeros(size, dtype=src.dtype, device=src.device)
    return out.scatter_add_(dim, index, src)


def segment_aggregate(tensor, segment_ids, reduce='sum'):
    """
    Aggregates tensor elements along segments defined in segment_ids.

    Args:
        tensor (torch.Tensor): The tensor to aggregate.
        segment_ids (torch.Tensor): A 1D tensor containing segment indices for aggregation.
        reduce (str): Either 'sum' or 'mean' to define the aggregation method.

    Returns:
        torch.Tensor: Aggregated tensor.
    """
    unique_segments = segment_ids.unique(sorted=True)
    aggregated_tensor = []

    for segment in unique_segments:
        segment_mask = (segment_ids == segment)
        segment_values = tensor[segment_mask]

        if reduce == 'sum':
            aggregated_tensor.append(segment_values.sum(dim=0))
        elif reduce == 'mean':
            aggregated_tensor.append(segment_values.mean(dim=0))
        else:
            raise ValueError("Invalid reduce argument. Use 'sum' or 'mean'.")

    return torch.stack(aggregated_tensor)


def calculate_neighbor_angles(R, id3_i, id3_j, id3_k):
    # Ri = torch.index_select(R, 0, id3_i)
    # Rj = torch.index_select(R, 0, id3_j)
    # Rk = torch.index_select(R, 0, id3_k)
    Ri = R[id3_i]
    Rj = R[id3_j]
    Rk = R[id3_k]
    R1 = Rj - Ri
    R2 = Rk - Rj
    x = torch.sum(R1 * R2, dim=-1)
    y = torch.cross(R1, R2)
    y = torch.norm(y, dim=-1)
    angle = torch.atan2(y, x)
    return angle


def calculate_interatomic_distances(R, idx_i, idx_j):
    # Ri = torch.index_select(R, 0, idx_i)
    # Rj = torch.index_select(R, 0, idx_j)
    Ri = R[idx_i]
    Rj = R[idx_j]
    Dij = torch.sqrt(torch.relu(torch.sum((Ri - Rj)**2, -1)))
    return Dij


def swish(x):
    return x * torch.sigmoid(x)


def bessel_basis(n, k):
    zeros = Jn_zeros(n, k)
    normalizer = []
    for order in range(n):
        normalizer_tmp = []
        for i in range(k):
            normalizer_tmp += [0.5 * Jn(zeros[order, i], order + 1)**2]
        normalizer_tmp = 1 / np.array(normalizer_tmp)**0.5
        normalizer += [normalizer_tmp]

    f = spherical_bessel_formulas(n)
    x = sym.symbols('x')
    bess_basis = []
    for order in range(n):
        bess_basis_tmp = []
        for i in range(k):
            bess_basis_tmp += [
                sym.simplify(normalizer[order][i] *
                             f[order].subs(x, zeros[order, i] * x))
            ]
        bess_basis += [bess_basis_tmp]
    return bess_basis


class Envelope(torch.nn.Module):
    def __init__(self, exponent: int):
        super().__init__()
        self.p = exponent + 1
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, x):
        p, a, b, c = self.p, self.a, self.b, self.c
        x_pow_p0 = x.pow(p - 1)
        x_pow_p1 = x_pow_p0 * x
        x_pow_p2 = x_pow_p1 * x
        return (1.0 / x + a * x_pow_p0 + b * x_pow_p1 +
                c * x_pow_p2) * (x < 1.0).to(x.dtype)


class BesselBasisLayer(nn.Module):
    def __init__(self, num_radial, cutoff, envelope_exponent=5, name='bessel_basis'):
        super().__init__()
        self.num_radial = num_radial
        self.inv_cutoff = 1 / cutoff
        self.envelope = Envelope(envelope_exponent)

        # Initialize frequencies at canonical positions
        self.frequencies = nn.Parameter(torch.from_numpy(np.pi * np.arange(1, num_radial + 1, dtype=np.float32)), requires_grad=True)

    def forward(self, inputs):
        d_scaled = inputs * self.inv_cutoff

        # Necessary for proper broadcasting behaviour
        d_scaled = d_scaled.unsqueeze(-1)

        d_cutoff = self.envelope(d_scaled)
        return d_cutoff * torch.sin(self.frequencies * d_scaled)


class SphericalBasisLayer(torch.nn.Module):
    def __init__(
        self,
        num_spherical: int,
        num_radial: int,
        cutoff: float = 5.0,
        envelope_exponent: int = 5,
    ):
        super().__init__()
        assert num_radial <= 64
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)

        bessel_forms = bessel_basis(num_spherical, num_radial)
        sph_harm_forms = real_sph_harm(num_spherical)
        self.sph_funcs = []
        self.bessel_funcs = []

        x, theta = sym.symbols('x theta')
        modules = {'sin': torch.sin, 'cos': torch.cos}
        for i in range(num_spherical):
            if i == 0:
                sph1 = sym.lambdify([theta], sph_harm_forms[i][0], modules)(0)
                self.sph_funcs.append(partial(self._sph_to_tensor, sph1))
            else:
                sph = sym.lambdify([theta], sph_harm_forms[i][0], modules)
                self.sph_funcs.append(sph)
            for j in range(num_radial):
                bessel = sym.lambdify([x], bessel_forms[i][j], modules)
                self.bessel_funcs.append(bessel)

    @staticmethod
    def _sph_to_tensor(sph, x):
        return torch.zeros_like(x) + sph

    def forward(self, inputs):
        dist, angle, idx_kj = inputs
        dist = dist / self.cutoff
        rbf = torch.stack([f(dist) for f in self.bessel_funcs], dim=1)
        rbf = self.envelope(dist).unsqueeze(-1) * rbf

        cbf = torch.stack([f(angle) for f in self.sph_funcs], dim=1)

        n, k = self.num_spherical, self.num_radial
        out = (rbf[idx_kj].view(-1, n, k) * cbf.view(-1, n, 1)).view(-1, n * k)
        return out


class EmbeddingBlock(torch.nn.Module):
    def __init__(self, hidden_channels: int, act: Callable):
        super().__init__()
        self.act = act

        self.emb = Embedding(95, hidden_channels)
        self.lin_rbf = Linear(6, hidden_channels)
        self.lin = Linear(3 * hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.emb.weight.data.uniform_(-sqrt(3), sqrt(3))
        self.lin_rbf.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, inputs):
        x, rbf, i, j = inputs
        x = self.emb(x)
        rbf = self.act(self.lin_rbf(rbf))
        return self.act(self.lin(torch.cat([x[i], x[j], rbf], dim=-1)))


class ResidualLayer(torch.nn.Module):
    def __init__(self, hidden_channels: int, act: Callable):
        super().__init__()
        self.act = act
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin1.weight, scale=2.0)
        # self.lin1.bias.data.fill_(0)
        glorot_orthogonal(self.lin2.weight, scale=2.0)
        # self.lin2.bias.data.fill_(0)

    def forward(self, x):
        return x + self.act(self.lin2(self.act(self.lin1(x))))


class InteractionPPBlock(torch.nn.Module):
    def __init__(
        self,
        hidden_channels: int = 128,
        int_emb_size: int = 64,
        basis_emb_size: int = 8,
        num_before_skip: int = 1,
        num_after_skip: int = 2,
        act: Callable = swish,
        num_spherical=7,
        num_radial=6,
    ):
        super().__init__()
        self.act = act

        # Transformation of Bessel and spherical basis representations:
        self.lin_rbf1 = Linear(num_radial, basis_emb_size, bias=False)
        self.lin_rbf2 = Linear(basis_emb_size, hidden_channels, bias=False)

        self.lin_sbf1 = Linear(num_spherical * num_radial, basis_emb_size,
                               bias=False)
        self.lin_sbf2 = Linear(basis_emb_size, int_emb_size, bias=False)

        # Hidden transformation of input message:
        self.lin_kj = Linear(hidden_channels, hidden_channels)
        self.lin_ji = Linear(hidden_channels, hidden_channels)

        # Embedding projections for interaction triplets:
        self.lin_down = Linear(hidden_channels, int_emb_size, bias=False)
        self.lin_up = Linear(int_emb_size, hidden_channels, bias=False)

        # Residual layers before and after skip connection:
        self.layers_before_skip = torch.nn.ModuleList([
            ResidualLayer(hidden_channels, act) for _ in range(num_before_skip)
        ])
        self.lin = Linear(hidden_channels, hidden_channels)
        self.layers_after_skip = torch.nn.ModuleList([
            ResidualLayer(hidden_channels, act) for _ in range(num_after_skip)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_rbf2.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf2.weight, scale=2.0)

        glorot_orthogonal(self.lin_kj.weight, scale=2.0)
        # self.lin_kj.bias.data.fill_(0)
        glorot_orthogonal(self.lin_ji.weight, scale=2.0)
        # self.lin_ji.bias.data.fill_(0)

        glorot_orthogonal(self.lin_down.weight, scale=2.0)
        glorot_orthogonal(self.lin_up.weight, scale=2.0)

        for res_layer in self.layers_before_skip:
            res_layer.reset_parameters()
        glorot_orthogonal(self.lin.weight, scale=2.0)
        # self.lin.bias.data.fill_(0)
        for res_layer in self.layers_after_skip:
            res_layer.reset_parameters()

    def forward(self, inputs) -> Tensor:
        x, rbf, sbf, idx_kj, idx_ji = inputs
        # Initial transformation:
        x_ji = self.act(self.lin_ji(x))
        x_kj = self.act(self.lin_kj(x))

        # Transformation via Bessel basis:
        rbf = self.lin_rbf1(rbf)
        rbf = self.lin_rbf2(rbf)
        x_kj = x_kj * rbf

        # Down project embedding and generating triple-interactions:
        x_kj = self.act(self.lin_down(x_kj))

        # Transform via 2D spherical basis:
        sbf = self.lin_sbf1(sbf)
        sbf = self.lin_sbf2(sbf)
        x_kj = x_kj[idx_kj] * sbf

        # Aggregate interactions and up-project embeddings:
        x_kj = scatter(x_kj, idx_ji, dim=0, dim_size=x.size(0))
        x_kj = self.act(self.lin_up(x_kj))

        h = x_ji + x_kj
        for layer in self.layers_before_skip:
            h = layer(h)
        h = self.act(self.lin(h)) + x
        for layer in self.layers_after_skip:
            h = layer(h)

        return h


class OutputPPBlock(torch.nn.Module):
    def __init__(
        self,
        hidden_channels: int = 128,
        out_emb_channels: int = 256,
        out_channels: int = 1,
        num_layers: int = 3,
        act: Callable = swish,
        output_initializer: str = 'glorot_orthogonal',
        num_radial: int = 6,
    ):
        assert output_initializer in {'zeros', 'glorot_orthogonal'}

        super().__init__()

        self.act = act
        self.output_initializer = output_initializer

        self.lin_rbf = Linear(num_radial, hidden_channels, bias=False)

        # The up-projection layer:
        self.lin_up = Linear(hidden_channels, out_emb_channels, bias=False)
        self.lins = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.lins.append(Linear(out_emb_channels, out_emb_channels))
        self.lin = Linear(out_emb_channels, out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf.weight, scale=2.0)
        glorot_orthogonal(self.lin_up.weight, scale=2.0)
        for lin in self.lins:
            glorot_orthogonal(lin.weight, scale=2.0)
            # lin.bias.data.fill_(0)
        if self.output_initializer == 'zeros':
            self.lin.weight.data.fill_(0)
        elif self.output_initializer == 'glorot_orthogonal':
            glorot_orthogonal(self.lin.weight, scale=2.0)

    def forward(self, inputs) -> Tensor:
        x, rbf, i, num_nodes = inputs
        x = self.lin_rbf(rbf) * x
        x = scatter(x, i, dim=0, dim_size=num_nodes)
        x = self.lin_up(x)
        for lin in self.lins:
            x = self.act(lin(x))
        return self.lin(x)


class DimenetPP(torch.nn.Module):
    def __init__(
        self,
        emb_size=128,
        num_blocks=4,
        num_spherical=7,
        num_radial=6,
        cutoff=5.0,
        envelope_exponent=5,
        activation=swish,
        extensive=True,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.num_blocks = num_blocks
        self.extensive = extensive

        # Cosine basis function expansion layer
        self.rbf_layer = BesselBasisLayer(
            num_radial, cutoff=cutoff, envelope_exponent=envelope_exponent
        )
        self.sbf_layer = SphericalBasisLayer(
            num_spherical,
            num_radial,
            cutoff=cutoff,
            envelope_exponent=envelope_exponent,
        )

        # Embedding and first output block
        self.output_blocks = nn.ModuleList()
        self.emb_block = EmbeddingBlock(emb_size, activation)
        self.output_blocks.append(OutputPPBlock())

        # Interaction and remaining output blocks
        self.int_blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.int_blocks.append(InteractionPPBlock())
            self.output_blocks.append(OutputPPBlock())

    def forward(self, inputs):
        # remove the first dimension of the input
        for key in inputs:
            if type(inputs[key]) is torch.Tensor:
                inputs[key] = inputs[key][0]

        Z, R = inputs["Z"], inputs["R"]
        batch_seg = inputs["batch_seg"]
        idnb_i, idnb_j = inputs["idnb_i"], inputs["idnb_j"]
        id_expand_kj, id_reduce_ji = inputs["id_expand_kj"], inputs["id_reduce_ji"]
        id3dnb_i, id3dnb_j, id3dnb_k = (
            inputs["id3dnb_i"],
            inputs["id3dnb_j"],
            inputs["id3dnb_k"],
        )
        n_atoms = Z.shape[0]

        # Calculate distances
        Dij = calculate_interatomic_distances(R, idnb_i, idnb_j)
        rbf = self.rbf_layer(Dij)

        # Calculate angles
        Anglesijk = calculate_neighbor_angles(R, id3dnb_i, id3dnb_j, id3dnb_k)
        sbf = self.sbf_layer([Dij, Anglesijk, id_expand_kj])

        # Embedding block
        x = self.emb_block([Z, rbf, idnb_i, idnb_j])
        P = self.output_blocks[0]([x, rbf, idnb_i, n_atoms])

        # Interaction blocks
        for i in range(self.num_blocks):
            x = self.int_blocks[i]([x, rbf, sbf, id_expand_kj, id_reduce_ji])
            P += self.output_blocks[i + 1]([x, rbf, idnb_i, n_atoms])

        if self.extensive:
            P = segment_aggregate(P, batch_seg, reduce='sum')
        else:
            P = segment_aggregate(P, batch_seg, reduce='mean')

        return P
