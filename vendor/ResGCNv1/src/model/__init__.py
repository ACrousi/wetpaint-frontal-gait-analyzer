import logging

from . import blocks
from .nets import ResGCN
from .mlp import GaitMLP
from .lstm import GaitLSTM
from .modules import ResGCN_Module, AttGCN_Module
from .attentions import *


from .ctrgcn import CTRGCN, CTRGCN_MultiBranch

__model = {
    'resgcn': ResGCN,
    'gaitmlp': GaitMLP,
    'lstm': GaitLSTM,
    'ctrgcn': CTRGCN,
    'ctrgcn-mb': CTRGCN_MultiBranch,
}

__attention = {
    'pa': Part_Att,
    'ca': Channel_Att,
    'fa': Frame_Att,
    'ja': Joint_Att,
    'pca': Part_Conv_Att,
    'psa': Part_Share_Att,
}

__structure = {
    'custom': {'structure': [1,1,1,1], 'block': 'Basic'},
    'b15': {'structure': [1,2,2,2], 'block': 'Basic'},
    'b19': {'structure': [1,2,3,3], 'block': 'Basic'},
    'b23': {'structure': [1,3,4,3], 'block': 'Basic'},
    'b29': {'structure': [1,3,6,4], 'block': 'Basic'},
    'n39': {'structure': [1,2,2,2], 'block': 'Bottleneck'},
    'n51': {'structure': [1,2,3,3], 'block': 'Bottleneck'},
    'n57': {'structure': [1,3,4,3], 'block': 'Bottleneck'},
    'n75': {'structure': [1,3,6,4], 'block': 'Bottleneck'},
}

__reduction = {
    'r1': {'reduction': 1},
    'r2': {'reduction': 2},
    'r4': {'reduction': 4},
    'r8': {'reduction': 8},
}

def create(model_type, **kwargs):
    model_split = model_type.split('-')
    if model_split[0] in __attention.keys():
        kwargs.update({'module': AttGCN_Module, 'attention': __attention[model_split[0]]})
        del(model_split[0])
    else:
        kwargs.update({'module': ResGCN_Module, 'attention': None})

    model = model_split[0]
    
    # Check for multi-branch CTR-GCN variant (ctrgcn-mb)
    if model == 'ctrgcn' and len(model_split) > 1 and model_split[1] == 'mb':
        model_key = 'ctrgcn-mb'
        if 'data_shape' in kwargs:
            # data_shape: (N_branch, C, T, V, M)
            kwargs['num_input_branch'] = kwargs['data_shape'][0]  # Number of input branches
            kwargs['num_point'] = kwargs['data_shape'][3]
            kwargs['num_person'] = kwargs['data_shape'][4]
            kwargs['in_channels'] = kwargs['data_shape'][1]  # Channels per branch
        return __model[model_key](**kwargs)
    elif model == 'gaitmlp':
        # For MLP, expect input_dim and num_class in kwargs
        if 'input_dim' not in kwargs or 'num_class' not in kwargs:
            logging.error('Error: gaitmlp requires input_dim and num_class in kwargs!')
            raise ValueError()
        return __model[model](**kwargs)
    elif model == 'lstm':
        # For LSTM, pass data_shape, num_class, and model_args
        return __model[model](**kwargs)
    elif model == 'ctrgcn':
        if 'data_shape' in kwargs:
             # data_shape: (N_branch, C, T, V, M)
             kwargs['num_point'] = kwargs['data_shape'][3]
             kwargs['num_person'] = kwargs['data_shape'][4]
             # Force in_channels to 3 (x, y, score) to match original CTR-GCN behavior
             kwargs['in_channels'] = 3
        return __model[model](**kwargs)
    else:
        # For ResGCN variants
        try:
            [model, structure, reduction] = model_split
        except:
            [model, structure], reduction = model_split, 'r1'
        if not (model in __model.keys() and structure in __structure.keys() and reduction in __reduction.keys()):
            logging.info('')
            logging.error('Error: Do NOT exist this model_type: {}!'.format(model_type))
            raise ValueError()
        return __model[model](**(__structure[structure]), **(__reduction[reduction]), **kwargs)

