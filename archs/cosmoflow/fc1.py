import numpy as np
import torch.nn as nn
import torch


class scale_1p2(nn.Module):
    def forward(self, input):
        return 1.2 * input


def build_model(input_shape, target_size, dropout=0):
    shape = np.array(input_shape[:-1], dtype=np.int)
    conv_args = {
        "in_channels": input_shape[-1],
        "out_channels": 32,
        "kernel_size": 3,
    }
    maxpool_args = dict(kernel_size=2)

    layers = [
        nn.Conv3d(**conv_args),
        nn.LeakyReLU(),
        nn.MaxPool3d(**maxpool_args),
    ]
    shape = (shape - 1) // 2

    conv_args["in_channels"] = conv_args["out_channels"]
    for _ in range(4):
        layers += [
            nn.Conv3d(**conv_args),
            nn.LeakyReLU(),
            nn.MaxPool3d(**maxpool_args),
        ]
        shape = (shape - 1) // 2

    flat_shape = np.prod(shape) * conv_args["out_channels"]
    layers += [
        nn.Flatten(),
        nn.Dropout(dropout),
        #
        nn.Linear(flat_shape, 128),
        nn.LeakyReLU(),
        nn.Dropout(dropout),
        #
        nn.Linear(128, 64),
        nn.LeakyReLU(),
        nn.Dropout(dropout),
        #
        nn.Linear(64, target_size),
        nn.Tanh(),
        scale_1p2(),
    ]

    return nn.Sequential(*layers)


class fc1(nn.Module):
    def __init__(self):
        super().__init__()
        # self.net = build_model((128, 128, 128, 4), 4, 0)
        self.net = get_standard_cosmoflow_model()
        self.net.train()
        self.example_input_array = torch.zeros((1, 4, 128, 128, 128))

    def forward(self, x):
        return self.net(x)


# Copyright (c) 2021-2022 NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import enum

import torch


class Convolution3DLayout(enum.Enum):
    NCDHW = "NCDHW"
    NDHWC = "NDHWC"

    @property
    def channel_last(self) -> bool:
        return self == Convolution3DLayout.NDHWC

    @property
    def pytorch_memory_format(self) -> torch.memory_format:
        return torch.channels_last_3d if self.channel_last else torch.contiguous_format


import torch
import torch.nn as nn
import torch.nn.functional as nnf

from typing import Iterable, Optional


class Conv3DActMP(nn.Module):
    def __init__(self,
                 conv_kernel: int,
                 conv_channel_in: int,
                 conv_channel_out: int):
        super().__init__()

        self.conv = nn.Conv3d(conv_channel_in, conv_channel_out,
                              kernel_size=conv_kernel,
                              stride=1, padding=1, bias=True)
        self.act = nn.LeakyReLU(negative_slope=0.3)
        self.mp = nn.MaxPool3d(kernel_size=2, stride=2)

        torch.nn.init.xavier_uniform_(self.conv.weight)
        torch.nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mp(self.act(self.conv(x)))


class CosmoFlow(nn.Module):
    def __init__(self,
                 n_conv_layers: int,
                 n_conv_filters: int,
                 conv_kernel: int,
                 dropout_rate: Optional[float] = 0.5):
        super().__init__()

        self.conv_seq = nn.ModuleList()
        input_channel_size = 4
        for i in range(n_conv_layers):
            output_channel_size = n_conv_filters * (1 << i)
            self.conv_seq.append(Conv3DActMP(conv_kernel,
                                             input_channel_size,
                                             output_channel_size))
            input_channel_size = output_channel_size

        flatten_inputs = 128 // (2 ** n_conv_layers)
        flatten_inputs = (flatten_inputs ** 3) * input_channel_size
        self.dense1 = nn.Linear(flatten_inputs, 128)
        self.dense2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 4)

        self.dropout_rate = dropout_rate
        if self.dropout_rate is not None:
            self.dr1 = nn.Dropout(p=self.dropout_rate)
            self.dr2 = nn.Dropout(p=self.dropout_rate)

        for layer in [self.dense1, self.dense2, self.output]:
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, conv_layer in enumerate(self.conv_seq):
            x = conv_layer(x)
        #x = x.flatten(1)

        # for tf compatibility
        x = x.permute(0, 2, 3, 4, 1).flatten(1)

        x = nnf.leaky_relu(self.dense1(x.flatten(1)), negative_slope=0.3)
        if self.dropout_rate is not None:
            x = self.dr1(x)

        x = nnf.leaky_relu(self.dense2(x), negative_slope=0.3)
        if self.dropout_rate is not None:
            x = self.dr2(x)

        return torch.tanh(self.output(x)) * 1.2


def get_standard_cosmoflow_model(kernel_size: int = 3,
                                 n_conv_layer: int = 5,
                                 n_conv_filters: int = 32,
                                 dropout_rate: Optional[float] = 0.5,
                                 layout: Convolution3DLayout = Convolution3DLayout.NCDHW,
                                 script: bool = False,
                                 device: str = "cuda") -> nn.Module:
    model = CosmoFlow(n_conv_layers=n_conv_layer,
                      n_conv_filters=n_conv_filters,
                      conv_kernel=kernel_size,
                      dropout_rate=dropout_rate)

    # model.to(memory_format=layout.pytorch_memory_format,
    #          device=device)

    if script:
        model = torch.jit.script(model)
    return model


if __name__ == "__main__":
    model = get_standard_cosmoflow_model()
    print(model)
    print(model.example_input_array.shape)
    model.to("cuda")
    model.eval()
    with torch.no_grad():
        model(torch.randn(1, 4, 128, 128, 128).to("cuda"))