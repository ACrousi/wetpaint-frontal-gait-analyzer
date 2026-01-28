import math
import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class DilatedFusionLayer(nn.Module):
    """
    MS-TCN++ inspired bidirectional dilated fusion layer.
    Combines forward and backward dilation paths to capture both
    short-term movements (instantaneous jitter) and long-term gait patterns.
    
    Forward path: increasing dilation (1, 2, 4, ...)
    Backward path: decreasing dilation (4, 2, 1, ...)
    Both paths are fused at each layer for multi-scale temporal understanding.
    """
    def __init__(self, channels, num_layers=3, dropout=0.3):
        super().__init__()
        self.num_layers = num_layers
        
        # Forward path: increasing dilation (1, 2, 4, ...)
        self.conv_dilated_forward = nn.ModuleList([
            nn.Conv2d(channels, channels, kernel_size=(3, 1), 
                     padding=(2**i, 0), dilation=(2**i, 1))
            for i in range(num_layers)
        ])
        
        # Backward path: decreasing dilation (4, 2, 1)
        self.conv_dilated_backward = nn.ModuleList([
            nn.Conv2d(channels, channels, kernel_size=(3, 1),
                     padding=(2**(num_layers-1-i), 0), dilation=(2**(num_layers-1-i), 1))
            for i in range(num_layers)
        ])
        
        # Fusion: combine forward and backward paths
        self.conv_fusion = nn.ModuleList([
            nn.Conv2d(2*channels, channels, kernel_size=1)
            for _ in range(num_layers)
        ])
        
        self.bn = nn.ModuleList([
            nn.BatchNorm2d(channels) for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (N, C, T, V) - input features
        Returns:
            (N, C, T, V) - fused features with multi-scale temporal context
        """
        for i in range(self.num_layers):
            f_forward = self.conv_dilated_forward[i](x)
            f_backward = self.conv_dilated_backward[i](x)
            f_fused = self.conv_fusion[i](torch.cat([f_forward, f_backward], dim=1))
            f_fused = self.bn[i](f_fused)
            f_fused = F.relu(f_fused)
            f_fused = self.dropout(f_fused)
            x = x + f_fused  # Residual connection
        return x


class DropGraph(nn.Module):
    """
    Randomly drops edges in the adjacency matrix during training.
    This prevents over-reliance on specific joint relationships and
    improves generalization across different infant gait patterns.
    
    Similar to DropEdge but applied to the learned adjacency matrix.
    """
    def __init__(self, drop_prob=0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, A):
        """
        Args:
            A: Adjacency matrix (num_subset, V, V) or (V, V)
        Returns:
            Dropped adjacency matrix with rescaling
        """
        if self.training and self.drop_prob > 0:
            # Create mask with same shape as A
            mask = torch.bernoulli(
                torch.ones_like(A) * (1 - self.drop_prob)
            )
            # Apply mask and rescale to maintain expected value
            A = A * mask / (1 - self.drop_prob + 1e-8)
        return A


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size-1) * (dilation-1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class MultiScale_TemporalConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilations=[1,2,3,4],
                 residual=True,
                 residual_kernel_size=1,
                 use_mstcn_fusion=False,
                 mstcn_layers=3,
                 mstcn_dropout=0.3):

        super().__init__()
        
        self.use_mstcn_fusion = use_mstcn_fusion
        self.stride = stride
        
        if self.use_mstcn_fusion:
            # === [簡化模式] Adapter Mode ===
            # 因為後面有強大的 Fusion 層，這裡我們不需要多分支
            # 只需要一個 1x1 卷積把 in_channels 映射到 out_channels 即可
            # 這能節省原本 80% 以上的參數
            self.num_branches = 1
            self.branches = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    # 這裡還可以選配一個 Dropout 來進一步抗過擬合
                    nn.Dropout(0.3) 
                )
            ])
            
            # Handle stride if needed
            if stride > 1:
                self.stride_pool = nn.AvgPool2d(kernel_size=(stride, 1), stride=(stride, 1))
            else:
                self.stride_pool = nn.Identity()
            
            # 初始化 Fusion 層
            self.mstcn_fusion = DilatedFusionLayer(
                out_channels, num_layers=mstcn_layers, dropout=mstcn_dropout
            )
            
        else:
            # === [原始模式] Legacy Mode ===
            # 如果不用 Fusion，才建立原本複雜的並行分支
            assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'
            self.num_branches = len(dilations) + 2
            branch_channels = out_channels // self.num_branches
            
            if type(kernel_size) == list:
                assert len(kernel_size) == len(dilations)
            else:
                kernel_size = [kernel_size]*len(dilations)
            
            self.branches = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
                    nn.BatchNorm2d(branch_channels),
                    nn.ReLU(inplace=True),
                    TemporalConv(branch_channels, branch_channels, kernel_size=ks, stride=stride, dilation=dilation),
                )
                for ks, dilation in zip(kernel_size, dilations)
            ])
            
            # Additional Max & 1x1 branch
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(3,1), stride=(stride,1), padding=(1,0)),
                nn.BatchNorm2d(branch_channels)
            ))

            self.branches.append(nn.Sequential(
                nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride,1)),
                nn.BatchNorm2d(branch_channels)
            ))

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)

        # initialize
        self.apply(weights_init)

    def forward(self, x):
        # Input dim: (N,C,T,V)
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        
        # Apply MS-TCN++ bidirectional dilated fusion if enabled
        if self.use_mstcn_fusion:
            # Apply stride pooling if needed (before fusion for efficiency)
            out = self.stride_pool(out)
            out = self.mstcn_fusion(out)
        
        out += res
        return out


class CTRGC(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=8, mid_reduction=1):
        super(CTRGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels <= 9:
            # For small in_channels (3, 6, 9), use fixed channel sizes
            self.rel_channels = 8
            self.mid_channels = 16
        else:
            self.rel_channels = max(1, in_channels // rel_reduction)
            self.mid_channels = max(1, in_channels // mid_reduction)
        self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
        self.tanh = nn.Tanh()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x, A=None, alpha=1):
        x1, x2, x3 = self.conv1(x).mean(-2), self.conv2(x).mean(-2), self.conv3(x)
        x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
        x1 = self.conv4(x1) * alpha + (A.unsqueeze(0).unsqueeze(0) if A is not None else 0)  # N,C,V,V
        x1 = torch.einsum('ncuv,nctv->nctu', x1, x3)
        return x1

class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, adaptive=True, residual=True, drop_graph=0.0):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.adaptive = adaptive
        self.num_subset = A.shape[0]
        self.convs = nn.ModuleList()
        for i in range(self.num_subset):
            self.convs.append(CTRGC(in_channels, out_channels))

        if residual:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.down = lambda x: x
        else:
            self.down = lambda x: 0
        if self.adaptive:
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        else:
            self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU(inplace=True)
        
        # DropGraph regularization for adjacency matrix
        self.drop_graph = DropGraph(drop_graph) if drop_graph > 0 else nn.Identity()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

    def forward(self, x):
        y = None
        if self.adaptive:
            A = self.PA
        else:
            A = self.A.cuda(x.get_device())
        
        # Apply DropGraph to adjacency matrix during training
        A = self.drop_graph(A)
        
        for i in range(self.num_subset):
            z = self.convs[i](x, A[i], self.alpha)
            y = z + y if y is not None else z
        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)


        return y


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True, 
                 kernel_size=5, dilations=[1,2], drop_channel=0.0, drop_graph=0.0,
                 use_mstcn_fusion=False, mstcn_layers=3, mstcn_dropout=0.3):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A, adaptive=adaptive, drop_graph=drop_graph)
        self.tcn1 = MultiScale_TemporalConv(out_channels, out_channels, kernel_size=kernel_size, stride=stride, dilations=dilations,
                                            residual=False, use_mstcn_fusion=use_mstcn_fusion,
                                            mstcn_layers=mstcn_layers, mstcn_dropout=mstcn_dropout)
        self.relu = nn.ReLU(inplace=True)
        self.drop_out = nn.Dropout(0.5)
        
        # Channel Dropout (Spatial Dropout)
        self.drop_channel = nn.Dropout2d(drop_channel) if drop_channel > 0 else nn.Identity()

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        x_out = self.tcn1(self.gcn1(x))
        x_out = self.drop_channel(x_out)
        y = self.relu(x_out + self.residual(x))
        y = self.drop_out(y) 
        return y

class CTRGCN(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=1, A=None, in_channels=3,
                 drop_out=0.5, adaptive=True, drop_channel=0.4, drop_graph=0.4,
                 use_mstcn_fusion=True, mstcn_layers=2, mstcn_dropout=0.4, **kwargs):
        super(CTRGCN, self).__init__()

        self.gait_dim = kwargs. get('gait_dim', None)
        self.use_ldl = kwargs. get('use_ldl', False)
        
        if torch.is_tensor(A):
            A = A.cpu().numpy()
        if A is None: 
            raise ValueError("Adjacency matrix A must be provided")

        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn. BatchNorm1d(num_person * in_channels * num_point)

        # 減少基礎通道數
        base_channel = 32  # 原本 64 → 32
        
        # Common layer kwargs for enhanced regularization
        layer_kwargs = dict(
            adaptive=adaptive, 
            drop_channel=drop_channel,
            drop_graph=drop_graph,
            use_mstcn_fusion=use_mstcn_fusion,
            mstcn_layers=mstcn_layers,
            mstcn_dropout=mstcn_dropout
        )
        
        # 簡化的層結構 (6層 → 4層)
        self.l1 = TCN_GCN_unit(in_channels, base_channel, A, residual=False, **layer_kwargs)
        self.l2 = TCN_GCN_unit(base_channel, base_channel, A, **layer_kwargs)
        
        # 唯一的下採樣層 (128 → 64, 2倍)
        self.l3 = TCN_GCN_unit(base_channel, base_channel, A, **layer_kwargs)
        self.l4 = TCN_GCN_unit(base_channel, base_channel, A, **layer_kwargs)
        self.l5 = TCN_GCN_unit(base_channel, base_channel, A, **layer_kwargs)
        self.l6 = TCN_GCN_unit(base_channel, base_channel, A, **layer_kwargs)
        self.l7 = TCN_GCN_unit(base_channel*2, base_channel*2, A, **layer_kwargs) #
        self.l8 = TCN_GCN_unit(base_channel*2, base_channel*4, A, stride=2, **layer_kwargs)
        self.l9 = TCN_GCN_unit(base_channel*4, base_channel*4, A, **layer_kwargs)
        self.l10 = TCN_GCN_unit(base_channel*4, base_channel*4, A, **layer_kwargs) #

        final_channels = base_channel # 64

        if self.gait_dim:
            self.gait_embed = nn.Linear(self.gait_dim, 32)
            fcn_input_dim = final_channels + 32
        else: 
            fcn_input_dim = final_channels

        self.drop_out = nn.Dropout(drop_out) if drop_out else nn.Identity()
        self.fcn = nn.Linear(fcn_input_dim, num_class)
        
        if not self.use_ldl:
            nn.init.normal_(self.fcn.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)

    def forward(self, x, gait_params=None):
        if len(x.shape) == 6:
            x = x[:, 0, :3, : , :, :]
            
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        
        x = self.l1(x)  # 128
        x = self.l2(x)  # 128
        # x = self.l3(x)  # 64 (stride=2)
        # x = self.l4(x)  # 64
        # x = self.l5(x)  # 128 (stride=2)
        # x = self.l6(x)  # 128
        # x = self.l7(x)  # 128
        # x = self.l8(x)  # 256 (stride=2)
        # x = self.l9(x)  # 256
        # x = self.l10(x)  # 256

        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.drop_out(x)
        
        if self.gait_dim and gait_params is not None: 
            gait_embed = self.gait_embed(gait_params)
            x = torch.cat([x, gait_embed], dim=1)

        feature = x.clone()
        x = self.fcn(x)

        return x, feature


class CTRGCN_InputBranch(nn.Module):
    """
    Input branch for processing a single stream (joints, bones, or velocities).
    Similar to ResGCN_Input_Branch but using CTR-GCN's TCN_GCN_unit.
    """
    def __init__(self, in_channels, out_channels, A, num_person=1, num_point=17, adaptive=True, 
                 drop_channel=0.3, drop_graph=0.1, use_mstcn_fusion=True, mstcn_layers=3, mstcn_dropout=0.3):
        super(CTRGCN_InputBranch, self).__init__()
        
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        
        # Common layer kwargs for enhanced regularization
        layer_kwargs = dict(
            adaptive=adaptive, 
            drop_channel=drop_channel,
            drop_graph=drop_graph,
            use_mstcn_fusion=use_mstcn_fusion,
            mstcn_layers=mstcn_layers,
            mstcn_dropout=mstcn_dropout
        )
        
        # Input branch layers: in_channels -> out_channels
        self.l1 = TCN_GCN_unit(in_channels, out_channels, A, residual=False, **layer_kwargs)
        self.l2 = TCN_GCN_unit(out_channels, out_channels, A, **layer_kwargs)
        
        bn_init(self.data_bn, 1)
    
    def forward(self, x):
        """
        Args:
            x: (N, C, T, V, M)
        Returns:
            (N*M, out_channels, T, V)
        """
        N, C, T, V, M = x.size()
        
        # Batch normalization
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        
        # Process through input branch layers
        x = self.l1(x)
        x = self.l2(x)
        
        return x


class CTRGCN_MultiBranch(nn.Module):
    """
    CTR-GCN with Multi-Branch Input, similar to ResGCN's architecture.
    Supports multiple input streams (joints, bones, velocities) that are processed
    independently then concatenated before main processing.
    
    Input shape: (N, I, C, T, V, M)
        - N: batch size
        - I: number of input branches (e.g., 3 for joints, bones, velocities)
        - C: channels per branch (e.g., 3 for x, y, score or 6 for motion features)
        - T: temporal frames
        - V: vertices (joints)
        - M: persons
    """
    def __init__(self, num_class=60, num_point=25, num_person=1, A=None, in_channels=3,
                 num_input_branch=3, drop_out=0.5, adaptive=True, drop_channel=0.3, drop_graph=0.1,
                 use_mstcn_fusion=True, mstcn_layers=3, mstcn_dropout=0.3, **kwargs):
        super(CTRGCN_MultiBranch, self).__init__()

        self.gait_dim = kwargs.get('gait_dim', None)
        self.use_ldl = kwargs.get('use_ldl', False)
        self.num_input_branch = num_input_branch
        self.num_point = num_point
        
        if torch.is_tensor(A):
            A = A.cpu().numpy()
        if A is None:
            raise ValueError("Adjacency matrix A must be provided")

        self.num_class = num_class
        
        # Common layer kwargs for enhanced regularization
        layer_kwargs = dict(
            adaptive=adaptive, 
            drop_channel=drop_channel,
            drop_graph=drop_graph,
            use_mstcn_fusion=use_mstcn_fusion,
            mstcn_layers=mstcn_layers,
            mstcn_dropout=mstcn_dropout
        )
        
        # Input branch output channels (before concatenation)
        branch_out_channels = 32  # Each branch outputs 32 channels
        
        # Create input branches (one for each input stream)
        self.input_branches = nn.ModuleList([
            CTRGCN_InputBranch(
                in_channels=in_channels,
                out_channels=branch_out_channels,
                A=A,
                num_person=num_person,
                num_point=num_point,
                **layer_kwargs
            )
            for _ in range(num_input_branch)
        ])
        
        # Main stream: processes concatenated features from all branches
        # Input channels = branch_out_channels * num_input_branch
        main_in_channels = branch_out_channels * num_input_branch  # e.g., 32 * 3 = 96
        base_channel = 64
        
        self.l3 = TCN_GCN_unit(main_in_channels, base_channel, A, **layer_kwargs)
        self.l4 = TCN_GCN_unit(base_channel, base_channel, A, **layer_kwargs)
        
        # Downsample layer
        self.l5 = TCN_GCN_unit(base_channel, base_channel*2, A, stride=2, **layer_kwargs)
        self.l6 = TCN_GCN_unit(base_channel*2, base_channel*2, A, **layer_kwargs)

        final_channels = base_channel * 2  # 128

        # Gait parameters embedding
        if self.gait_dim:
            self.gait_embed = nn.Linear(self.gait_dim, 32)
            fcn_input_dim = final_channels + 32
        else:
            fcn_input_dim = final_channels

        self.drop_out = nn.Dropout(drop_out) if drop_out else nn.Identity()
        self.fcn = nn.Linear(fcn_input_dim, num_class)
        
        if not self.use_ldl:
            nn.init.normal_(self.fcn.weight, 0, math.sqrt(2. / num_class))

    def forward(self, x, gait_params=None):
        """
        Args:
            x: (N, I, C, T, V, M) - multi-branch input
            gait_params: optional gait parameters for late fusion
        Returns:
            output: (N, num_class)
            feature: (N, final_channels)
        """
        if len(x.shape) == 5:
            # Single branch input (N, C, T, V, M) -> expand to (N, 1, C, T, V, M)
            x = x.unsqueeze(1)
        
        N, I, C, T, V, M = x.size()
        
        # Process each input branch
        branch_outputs = []
        for i, branch in enumerate(self.input_branches):
            if i < I:
                branch_out = branch(x[:, i, :, :, :, :])  # (N*M, branch_channels, T, V)
            else:
                # If fewer branches in input than model expects, use zeros
                branch_out = torch.zeros(N * M, 32, T, V, device=x.device, dtype=x.dtype)
            branch_outputs.append(branch_out)
        
        # Concatenate all branch outputs along channel dimension
        x = torch.cat(branch_outputs, dim=1)  # (N*M, branch_channels * num_branches, T, V)
        
        # Main stream processing
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)  # stride=2 downsample
        x = self.l6(x)
        
        # Global pooling
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)  # Average over T*V and M
        x = self.drop_out(x)
        
        # Late fusion with gait parameters
        if self.gait_dim and gait_params is not None:
            gait_embed = self.gait_embed(gait_params)
            x = torch.cat([x, gait_embed], dim=1)

        feature = x.clone()
        x = self.fcn(x)

        return x, feature

