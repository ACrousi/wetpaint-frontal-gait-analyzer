import torch
from torch import nn

from .modules import ResGCN_Module


class ResGCN_Input_Branch(nn.Module):
    def __init__(self, structure, block, num_channel, A, **kwargs):
        super(ResGCN_Input_Branch, self).__init__()

        self.register_buffer('A', A)

        module_list = [ResGCN_Module(num_channel, 64, 'Basic', A, initial=True, **kwargs)]
        module_list += [ResGCN_Module(64, 64, 'Basic', A, initial=True, **kwargs) for _ in range(structure[0] - 1)]
        module_list += [ResGCN_Module(64, 64, block, A, **kwargs) for _ in range(structure[1] - 1)]
        module_list += [ResGCN_Module(64, 32, block, A, **kwargs)]

        self.bn = nn.BatchNorm2d(num_channel)
        self.layers = nn.ModuleList(module_list)

    def forward(self, x):

        N, C, T, V, M = x.size()
        x = self.bn(x.permute(0,4,1,2,3).contiguous().view(N*M, C, T, V))
        for layer in self.layers:
            x = layer(x, self.A)

        return x


class ResGCN(nn.Module):
    def __init__(self, module, structure, block, data_shape, num_class, A, **kwargs):
        super(ResGCN, self).__init__()

        self.gait_dim = kwargs.get('gait_dim', None)
        self.use_ldl = kwargs.get('use_ldl', False)
        num_input, num_channel, _, _, _ = data_shape
        self.register_buffer('A', A)

        # input branches
        self.input_branches = nn.ModuleList([
            ResGCN_Input_Branch(structure, block, num_channel, A, **kwargs)
            for _ in range(num_input)
        ])

        # main stream
        module_list = [module(32*num_input, 128, block, A, stride=2, **kwargs)]
        module_list += [module(128, 128, block, A, **kwargs) for _ in range(structure[2] - 1)]
        module_list += [module(128, 256, block, A, stride=2, **kwargs)]
        module_list += [module(256, 256, block, A, **kwargs) for _ in range(structure[3] - 1)]
        self.main_stream = nn.ModuleList(module_list)

        # output
        self.global_pooling = nn.AdaptiveAvgPool2d(1)

        # gait parameter embedding
        self.gait_dim = self.gait_dim
        if self.gait_dim:
            self.gait_embed = nn.Linear(self.gait_dim, 64)
            fcn_input_dim = 256 + 64
        else:
            fcn_input_dim = 256

        # Dropout layer
        self.dropout = nn.Dropout(p=0.5)

        self.fcn = nn.Linear(fcn_input_dim, num_class)

        # init parameters
        init_param(self.modules())
        zero_init_lastBN(self.modules())

    def forward(self, x, gait_params=None):

        N, I, C, T, V, M = x.size()

        # input branches
        x_cat = []
        for i, branch in enumerate(self.input_branches):
            x_cat.append(branch(x[:,i,:,:,:,:]))
        x = torch.cat(x_cat, dim=1)

        # main stream
        for layer in self.main_stream:
            x = layer(x, self.A)

        # extract feature
        _, C, T, V = x.size()
        feature = x.view(N, M, C, T, V).permute(0, 2, 3, 4, 1)

        # output
        x = self.global_pooling(x)
        x = x.view(N, M, -1).mean(dim=1)

        # late fusion with gait parameters
        if self.gait_dim and gait_params is not None:
            gait_embed = self.gait_embed(gait_params)
            x = torch.cat([x, gait_embed], dim=1)

        feature = x.clone()
        # Apply dropout before classifier
        x = self.dropout(x)

        x = self.fcn(x)

        return x, feature


def init_param(modules):
    for m in modules:
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            #m.bias = None
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def zero_init_lastBN(modules):
    for m in modules:
        if isinstance(m, ResGCN_Module):
            if hasattr(m.scn, 'bn_up'):
                nn.init.constant_(m.scn.bn_up.weight, 0)
            if hasattr(m.tcn, 'bn_up'):
                nn.init.constant_(m.tcn.bn_up.weight, 0)
