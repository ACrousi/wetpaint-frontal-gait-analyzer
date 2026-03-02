import torch
import torch.nn as nn


class GaitLSTM(nn.Module):
    """LSTM baseline for skeleton-based gait age estimation.
    
    Receives skeleton sequences in the same format as ResGCN:
    (N, I, C, T, V, M) where:
        N = batch size
        I = number of input branches (e.g., 3 for joint+velocity+bone)
        C = channels (x, y, score)
        T = temporal frames
        V = number of joints (17 for COCO)
        M = number of persons
    
    Reshapes to (N, T, I*C*V*M) and feeds into LSTM → FC → output.
    """
    
    def __init__(self, data_shape, num_class, A=None, parts=None,
                 hidden_size=128, num_layers=2, dropout=0.5,
                 bidirectional=True, **kwargs):
        super(GaitLSTM, self).__init__()
        
        self.use_ldl = kwargs.get('use_ldl', False)
        
        # Parse data shape: (num_input, num_channel, num_frame, num_joint, num_person)
        num_input, num_channel, num_frame, num_joint, num_person = data_shape
        self.input_dim = num_input * num_channel * num_joint * num_person
        
        # Batch normalization on input
        self.bn_input = nn.BatchNorm1d(self.input_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Output dimension depends on bidirectional
        lstm_output_dim = hidden_size * 2 if bidirectional else hidden_size
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout)
        
        # Fully connected output
        self.fcn = nn.Linear(lstm_output_dim, num_class)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Set forget gate bias to 1
                n = param.size(0)
                param.data[n//4:n//2].fill_(1)
        
        nn.init.kaiming_normal_(self.fcn.weight, mode='fan_out', nonlinearity='relu')
        if self.fcn.bias is not None:
            nn.init.constant_(self.fcn.bias, 0)
    
    def forward(self, x, gait_params=None):
        # x shape: (N, I, C, T, V, M)
        N, I, C, T, V, M = x.size()
        
        # Reshape to (N, T, I*C*V*M) - flatten all spatial dims, keep temporal
        x = x.permute(0, 3, 1, 2, 4, 5).contiguous()  # (N, T, I, C, V, M)
        x = x.view(N, T, -1)  # (N, T, I*C*V*M)
        
        # Batch normalization: (N, T, D) -> (N, D, T) -> BN -> (N, T, D)
        x = x.permute(0, 2, 1)  # (N, D, T)
        x = self.bn_input(x)
        x = x.permute(0, 2, 1)  # (N, T, D)
        
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: (N, T, hidden*2)
        
        # Use last hidden state (average of forward and backward)
        # For bidirectional: concatenate the last forward and first backward hidden states
        feature = lstm_out[:, -1, :]  # (N, hidden*2) - use last timestep
        
        # Dropout + FC
        x = self.dropout(feature)
        out = self.fcn(x)
        
        return out, feature
