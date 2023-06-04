import math
from typing import List, Tuple

import torch
import torch.nn as nn


# sin activation
class Sine(nn.Module):
    def __init__(self, w0=30.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


def init_sine(dim_in, layer, c, w0, first=False):
    w_std = (1 / dim_in) if first else (math.sqrt(c / dim_in) / w0)
    layer.weight.data.uniform_(-w_std, w_std)
    if layer.bias is not None:
        layer.bias.data.uniform_(-w_std, w_std)


class SharedMLP(nn.Sequential):
    def __init__(
        self,
        args: List[int],
        *,
        bn: bool = False,
        activation="relu",
        preact: bool = False,
        first: bool = False,
        name: str = "",
        instance_norm: bool = False,
    ):
        super().__init__()

        for i in range(len(args) - 1):
            self.add_module(
                name + "layer{}".format(i),
                Conv2d(
                    args[i],
                    args[i + 1],
                    bn=(not first or not preact or (i != 0)) and bn,
                    activation=activation
                    if (not first or not preact or (i != 0))
                    else None,
                    preact=preact,
                    instance_norm=instance_norm,
                    first=first,
                ),
            )


class _ConvBase(nn.Sequential):
    def __init__(
        self,
        in_size,
        out_size,
        kernel_size,
        stride,
        padding,
        activation,
        bn,
        init,
        conv=None,
        batch_norm=None,
        bias=True,
        preact=False,
        name="",
        instance_norm=False,
        instance_norm_func=None,
        first=False,
    ):
        super().__init__()

        bias = bias and (not bn)
        conv_unit = conv(
            in_size,
            out_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        activation = Sine() if activation == "sine" else nn.ReLU(inplace=True)
        if isinstance(activation, Sine):
            # init_sine(conv_unit.in_channels, conv_unit, c=6., w0=activation.w0, first=first)
            init(conv_unit.weight)
            if bias:
                nn.init.constant_(conv_unit.bias, 0)
        else:
            init(conv_unit.weight)
            if bias:
                nn.init.constant_(conv_unit.bias, 0)

        if bn:
            if not preact:
                bn_unit = batch_norm(out_size)
            else:
                bn_unit = batch_norm(in_size)
        if instance_norm:
            if not preact:
                in_unit = instance_norm_func(
                    out_size, affine=False, track_running_stats=False
                )
            else:
                in_unit = instance_norm_func(
                    in_size, affine=False, track_running_stats=False
                )

        if preact:
            if bn:
                self.add_module(name + "bn", bn_unit)

            if activation is not None:
                self.add_module(name + "activation", activation)

            if not bn and instance_norm:
                self.add_module(name + "in", in_unit)

        self.add_module(name + "conv", conv_unit)

        if not preact:
            if bn:
                self.add_module(name + "bn", bn_unit)

            if activation is not None:
                self.add_module(name + "activation", activation)

            if not bn and instance_norm:
                self.add_module(name + "in", in_unit)


class _BNBase(nn.Sequential):
    def __init__(self, in_size, batch_norm=None, name=""):
        super().__init__()
        self.add_module(name + "bn", batch_norm(in_size))

        nn.init.constant_(self[0].weight, 1.0)
        nn.init.constant_(self[0].bias, 0)


class BatchNorm1d(_BNBase):
    def __init__(self, in_size: int, *, name: str = ""):
        super().__init__(in_size, batch_norm=nn.BatchNorm1d, name=name)


class BatchNorm2d(_BNBase):
    def __init__(self, in_size: int, name: str = ""):
        super().__init__(in_size, batch_norm=nn.BatchNorm2d, name=name)


class BatchNorm3d(_BNBase):
    def __init__(self, in_size: int, name: str = ""):
        super().__init__(in_size, batch_norm=nn.BatchNorm3d, name=name)


class Conv1d(_ConvBase):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        *,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        activation="relu",
        bn: bool = False,
        init=nn.init.kaiming_normal_,
        bias: bool = True,
        preact: bool = False,
        name: str = "",
        instance_norm=False,
        first=False,
    ):
        super().__init__(
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            activation,
            bn,
            init,
            conv=nn.Conv1d,
            batch_norm=BatchNorm1d,
            bias=bias,
            preact=preact,
            name=name,
            instance_norm=instance_norm,
            instance_norm_func=nn.InstanceNorm1d,
            first=first,
        )


class Conv2d(_ConvBase):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        *,
        kernel_size: Tuple[int, int] = (1, 1),
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0),
        activation="relu",
        bn: bool = False,
        init=nn.init.kaiming_normal_,
        bias: bool = True,
        preact: bool = False,
        name: str = "",
        instance_norm=False,
        first=False,
    ):
        super().__init__(
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            activation,
            bn,
            init,
            conv=nn.Conv2d,
            batch_norm=BatchNorm2d,
            bias=bias,
            preact=preact,
            name=name,
            instance_norm=instance_norm,
            instance_norm_func=nn.InstanceNorm2d,
            first=first,
        )


class Conv3d(_ConvBase):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        *,
        kernel_size: Tuple[int, int, int] = (1, 1, 1),
        stride: Tuple[int, int, int] = (1, 1, 1),
        padding: Tuple[int, int, int] = (0, 0, 0),
        activation="relu",
        bn: bool = False,
        init=nn.init.kaiming_normal_,
        bias: bool = True,
        preact: bool = False,
        name: str = "",
        instance_norm=False,
        first=False,
    ):
        super().__init__(
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            activation,
            bn,
            init,
            conv=nn.Conv3d,
            batch_norm=BatchNorm3d,
            bias=bias,
            preact=preact,
            name=name,
            instance_norm=instance_norm,
            instance_norm_func=nn.InstanceNorm3d,
            first=first,
        )


class FC(nn.Sequential):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        *,
        activation="relu",
        bn: bool = False,
        init=None,
        preact: bool = False,
        name: str = "",
        first=False,
    ):
        super().__init__()

        fc = nn.Linear(in_size, out_size, bias=not bn)
        if activation == "sine":
            activation = Sine()
            # init_sine(fc.in_features, fc, c=6., w0=activation.w0, first=first)
            if init is not None:
                init(fc.weight)
            if not bn:
                nn.init.constant_(fc.bias, 0)
        else:
            if activation is not None:
                activation = nn.ReLU(inplace=True)
            if init is not None:
                init(fc.weight)
            if not bn:
                nn.init.constant_(fc.bias, 0)

        if preact:
            if bn:
                self.add_module(name + "bn", BatchNorm1d(in_size))

            if activation is not None:
                self.add_module(name + "activation", activation)

        self.add_module(name + "fc", fc)

        if not preact:
            if bn:
                self.add_module(name + "bn", BatchNorm1d(out_size))

            if activation is not None:
                self.add_module(name + "activation", activation)
