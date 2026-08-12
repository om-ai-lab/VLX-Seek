import torch
import torch.nn as nn
import re
from .honeybee import CAbstractor
from functools import partial
import numpy as np
from torch.nn.init import trunc_normal_
from torch.nn import functional as F
import math


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)
    if projector_type == "cabstract":
        n_query = getattr(config, 'mm_projector_n_query', None)
        image_size = getattr(config, 'image_size', None)
        if not n_query:
            n_query = kwargs.get("mm_projector_n_query",144)
        if not image_size: 
            image_size = kwargs.get("image_size",336)
        vokens = int(image_size/14*image_size/14)
        print ("n_query",n_query)
        print ("image_size",image_size)
        print ("vokens",vokens)

        return CAbstractor(vokens, config.mm_hidden_size, config.hidden_size, num_queries=n_query)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)
    
    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')

def build_vision_projector_aux(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_aux_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_object_hidden_size, config.hidden_size)
    if projector_type == "cabstract":
        n_query = getattr(config, 'mm_projector_n_query', None)
        image_size = getattr(config, 'image_size', None)
        if not n_query:
            n_query = kwargs.get("mm_projector_n_query",144)
        if not image_size: 
            image_size = kwargs.get("image_size",336)
        vokens = int(image_size/14*image_size/14)
        print ("n_query",n_query)
        print ("image_size",image_size)
        print ("vokens",vokens)

        return CAbstractor(vokens, config.mm_object_hidden_size, config.hidden_size, num_queries=n_query)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_object_hidden_size, config.text_config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')
