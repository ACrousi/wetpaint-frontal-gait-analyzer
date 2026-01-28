import torch
import torch.nn as nn


class GaitMLP(nn.Module):
    def __init__(self, input_dim, num_class, hidden_dims=[128, 64], dropout=0.5, **kwargs):
        super(GaitMLP, self).__init__()
        self.use_ldl = kwargs.get('use_ldl', False)
        
        # Feature extraction layers
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        self.features = nn.Sequential(*layers)
        
        # Output layers
        # Output layers
        self.fcn = nn.Linear(in_dim, num_class)

    def forward(self, x, gait_params=None):
        # 如果有步態參數，使用步態參數作為輸入
        if gait_params is not None:
            x = gait_params
        else:
            # 如果沒有步態參數，使用骨架數據（需要展平）
            N = x.size(0)
            x = x.view(N, -1)
            
        x = self.features(x)
        
        if self.use_ldl:
            out = self.fcn(x)
        else:
            out = self.fcn(x)
            
        return out, None  # 返回分類輸出和 None（無特徵提取）