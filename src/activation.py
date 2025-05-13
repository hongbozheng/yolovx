"""
activation function
"""

import torch.nn as nn


def get_activation(activation_function_type):
    if activation_function_type == 'mish':
        activation_function_type = 'mish'
        activation_function = nn.Module()
    elif activation_function_type == 'leaky':
        activation_function_type = 'leakyrelu'
        activation_function = nn.LeakyReLU(negative_slope=0.1,inplace=True)
    elif activation_function_type == 'logistic':
        activation_function_type = 'logistic'
        activation_function = nn.Sigmoid()
    else:
        print('[ERROR]: INVALID ACTIVATION FUNCTION')

    return activation_function_type,activation_function
