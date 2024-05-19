import copy as cp
import torch
import torch.nn as nn
import sys
from .utils import mstcn, unit_gcn, unit_tcn

EPS = 1e-4


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


class STGCNBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 stride=1,
                 residual=True,
                 **kwargs):
        super().__init__()
        gcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'gcn_'}
        tcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'tcn_'}
        kwargs = {k: v for k, v in kwargs.items() if k[:4] not in [
            'gcn_', 'tcn_']}
        assert len(kwargs) == 0, f'Invalid arguments: {kwargs}'

        tcn_type = tcn_kwargs.pop('type', 'unit_tcn')
        assert tcn_type in ['unit_tcn', 'mstcn']
        gcn_type = gcn_kwargs.pop('type', 'unit_gcn')
        assert gcn_type in ['unit_gcn']

        self.gcn = unit_gcn(in_channels, out_channels, A, **gcn_kwargs)

        if tcn_type == 'unit_tcn':
            self.tcn = unit_tcn(out_channels, out_channels,
                                9, stride=stride, **tcn_kwargs)
        elif tcn_type == 'mstcn':
            self.tcn = mstcn(out_channels, out_channels,
                             stride=stride, **tcn_kwargs)
        self.relu = nn.ReLU()

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(
                in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x, A=None):
        """Defines the computation performed at every call."""
        res = self.residual(x)
        x = self.tcn(self.gcn(x, A)) + res
        return self.relu(x)


class Model(nn.Module):

    def __init__(self,
                 graph=None,
                 graph_args=dict(),
                 in_channels=3,
                 num_class=100,
                 base_channels=64,
                 data_bn_type='VC',
                 ch_ratio=2,
                 num_person=2,  # * Only used when data_bn_type == 'MVC'
                 num_stages=10,
                 inflate_stages=[5, 8],
                 down_stages=[5, 8],
                 **kwargs):

        super().__init__()
        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = torch.tensor(self.graph.A, dtype=torch.float32,
                         requires_grad=False)

        self.data_bn_type = data_bn_type
        self.num_class = num_class
        self.kwargs = kwargs

        if data_bn_type == 'MVC':
            self.data_bn = nn.BatchNorm1d(num_person * in_channels * A.size(1))
        elif data_bn_type == 'VC':
            self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        else:
            self.data_bn = nn.Identity()

        lw_kwargs = [cp.deepcopy(kwargs) for i in range(num_stages)]
        for k, v in kwargs.items():
            if isinstance(v, tuple) and len(v) == num_stages:
                for i in range(num_stages):
                    lw_kwargs[i][k] = v[i]
        lw_kwargs[0].pop('tcn_dropout', None)

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.ch_ratio = ch_ratio
        self.inflate_stages = inflate_stages
        self.down_stages = down_stages

        modules = []
        if self.in_channels != self.base_channels:
            modules = [STGCNBlock(in_channels, base_channels,
                                  A.clone(), 1, residual=False, **lw_kwargs[0])]

        inflate_times = 0
        for i in range(2, num_stages + 1):
            stride = 1 + (i in down_stages)
            in_channels = base_channels
            if i in inflate_stages:
                inflate_times += 1
            out_channels = int(self.base_channels *
                               self.ch_ratio ** inflate_times + EPS)
            base_channels = out_channels
            modules.append(STGCNBlock(in_channels, out_channels,
                           A.clone(), stride, **lw_kwargs[i - 1]))

        if self.in_channels == self.base_channels:
            num_stages -= 1

        self.num_stages = num_stages
        self.gcn = nn.ModuleList(modules)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fcn = nn.Linear(256, num_class)

    def forward(self, x):

        N, C, T, V, M = x.size()
        # Convert (N, C, T, V, M) to (N, M, V, C, T)
        x = x.permute(0, 4, 3, 1, 2).contiguous()

        # N, M, T, V, C = x.size()
        # Convert (N, M, T, V, C) to (N, M, V, C, T)
        # x = x.permute(0, 1, 3, 4, 2).contiguous()

        if self.data_bn_type == 'MVC':
            x = self.data_bn(x.view(N, M * V * C, T))
        else:
            x = self.data_bn(x.view(N * M, V * C, T))
        x = x.view(N, M, V, C, T).permute(
            0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        for i in range(self.num_stages):
            x = self.gcn[i](x)

        x = x.reshape((N, M) + x.shape[1:])

        N, M, C, T, V = x.shape
        x = x.reshape(N * M, C, T, V)

        x = self.pool(x)
        x = x.reshape(N, M, C)
        x = x.mean(dim=1)
        x = self.fcn(x)
        return x
