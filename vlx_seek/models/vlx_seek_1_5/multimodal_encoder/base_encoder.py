import torch
import torch.nn as nn


class AbsVisionTower(nn.Module):
    @torch.no_grad()
    def forward(self, images):
        raise NotImplementedError

    @property
    def dummy_feature(self):
        raise NotImplementedError

    @property
    def dtype(self):
        raise NotImplementedError

    @property
    def device(self):
        raise NotImplementedError

    @property
    def config(self):
        raise NotImplementedError


    @property
    def hidden_size(self):
        raise NotImplementedError

    @property
    def num_patches(self):
        raise NotImplementedError
