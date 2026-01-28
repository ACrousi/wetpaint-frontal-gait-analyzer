import torch
from torch import nn


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample.
    
    This is the same as Stochastic Depth used in EfficientNet and other modern architectures.
    It randomly drops the entire residual path during training.
    
    Args:
        x: Input tensor
        drop_prob: Probability of dropping the path (0-1)
        training: Whether in training mode
    
    Returns:
        Tensor with dropped paths (scaled appropriately)
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # Shape: (batch_size, 1, 1, 1) - same drop decision for all spatial/temporal dims
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize to 0 or 1
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample.
    
    When applied to the main path of a residual block, this is equivalent to
    stochastic depth as described in "Deep Networks with Stochastic Depth".
    """
    def __init__(self, drop_prob: float = 0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self):
        return f'drop_prob={self.drop_prob}'


def drop_edge(A, drop_prob: float = 0., training: bool = False):
    """Randomly drop edges from adjacency matrix A (DropEdge).
    
    DropEdge is a regularization technique that randomly removes edges from
    the graph during training to reduce over-smoothing in deep GCNs.
    
    Args:
        A: Adjacency matrix tensor of shape (K, V, V)
        drop_prob: Probability of dropping each edge (0-1)
        training: Whether in training mode
    
    Returns:
        Adjacency matrix with randomly dropped edges
    """
    if drop_prob == 0. or not training:
        return A
    # Create a mask where each edge has (1 - drop_prob) chance of being kept
    mask = (torch.rand_like(A.float()) > drop_prob).float()
    # Apply mask and rescale to maintain expected value
    A_dropped = A * mask / (1 - drop_prob)
    return A_dropped


class Spatial_Bottleneck_Block(nn.Module):
    def __init__(self, in_channels, out_channels, max_graph_distance, residual=False, reduction=4, **kwargs):
        super(Spatial_Bottleneck_Block, self).__init__()

        inter_channels = out_channels // reduction

        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
            )

        self.conv_down = nn.Conv2d(in_channels, inter_channels, 1)
        self.bn_down = nn.BatchNorm2d(inter_channels)
        self.conv = SpatialGraphConv(inter_channels, inter_channels, max_graph_distance)
        self.bn = nn.BatchNorm2d(inter_channels)
        self.conv_up = nn.Conv2d(inter_channels, out_channels, 1)
        self.bn_up = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        res_block = self.residual(x)

        x = self.conv_down(x)
        x = self.bn_down(x)
        x = self.relu(x)

        x = self.conv(x, A)
        x = self.bn(x)
        x = self.relu(x)

        x = self.conv_up(x)
        x = self.bn_up(x)
        x = self.relu(x + res_block)

        return x


class Temporal_Bottleneck_Block(nn.Module):
    def __init__(self, channels, temporal_window_size, stride=1, residual=False, reduction=4, **kwargs):
        super(Temporal_Bottleneck_Block, self).__init__()

        padding = ((temporal_window_size - 1) // 2, 0)
        inter_channels = channels // reduction

        if not residual:
            self.residual = lambda x: 0
        elif stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(channels, channels, 1, (stride,1)),
                nn.BatchNorm2d(channels),
            )

        self.conv_down = nn.Conv2d(channels, inter_channels, 1)
        self.bn_down = nn.BatchNorm2d(inter_channels)
        self.conv = nn.Conv2d(inter_channels, inter_channels, (temporal_window_size,1), (stride,1), padding)
        self.bn = nn.BatchNorm2d(inter_channels)
        self.conv_up = nn.Conv2d(inter_channels, channels, 1)
        self.bn_up = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, res_module):

        res_block = self.residual(x)

        x = self.conv_down(x)
        x = self.bn_down(x)
        x = self.relu(x)

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.conv_up(x)
        x = self.bn_up(x)
        x = self.relu(x + res_block + res_module)

        return x


class Spatial_Basic_Block(nn.Module):
    def __init__(self, in_channels, out_channels, max_graph_distance, residual=False, **kwargs):
        super(Spatial_Basic_Block, self).__init__()

        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
            )

        self.conv = SpatialGraphConv(in_channels, out_channels, max_graph_distance)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        res_block = self.residual(x)

        x = self.conv(x, A)
        x = self.bn(x)
        x = self.relu(x + res_block)

        return x


class Temporal_Basic_Block(nn.Module):
    def __init__(self, channels, temporal_window_size, stride=1, residual=False, **kwargs):
        super(Temporal_Basic_Block, self).__init__()

        padding = ((temporal_window_size - 1) // 2, 0)

        if not residual:
            self.residual = lambda x: 0
        elif stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(channels, channels, 1, (stride,1)),
                nn.BatchNorm2d(channels),
            )

        self.conv = nn.Conv2d(channels, channels, (temporal_window_size,1), (stride,1), padding)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, res_module):

        res_block = self.residual(x)

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x + res_block + res_module)

        return x


# Thanks to YAN Sijie for the released code on Github (https://github.com/yysijie/st-gcn)
class SpatialGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, max_graph_distance):
        super(SpatialGraphConv, self).__init__()

        # spatial class number (distance = 0 for class 0, distance = 1 for class 1, ...)
        self.s_kernel_size = max_graph_distance + 1

        # weights of different spatial classes
        self.gcn = nn.Conv2d(in_channels, out_channels*self.s_kernel_size, 1)

    def forward(self, x, A):

        # numbers in same class have same weight
        x = self.gcn(x)

        # divide nodes into different classes
        n, kc, t, v = x.size()
        x = x.view(n, self.s_kernel_size, kc//self.s_kernel_size, t, v)

        # spatial graph convolution
        x = torch.einsum('nkctv,kvw->nctw', (x, A[:self.s_kernel_size])).contiguous()

        return x
