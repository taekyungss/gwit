from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F
import pdb

class SubjectLayers(nn.Module):
    """Per subject linear layer."""
    def __init__(self, in_channels: int, out_channels: int, n_subjects: int, init_id: bool = False):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(n_subjects, in_channels, out_channels))
        if init_id:
            assert in_channels == out_channels
            self.weights.data[:] = torch.eye(in_channels)[None]
        self.weights.data *= 1 / in_channels**0.5

    def forward(self, x, subjects):
        #Extract Dimensions:
        # print(subjects, subjects.shape)
        _, C, D = self.weights.shape
        #
        # pdb.set_trace()
        weights = self.weights.gather(0, subjects.view(-1, 1, 1).expand(-1, C, D))
        return torch.einsum("bct,bcd->bdt", x, weights)

    def __repr__(self):
        S, C, D = self.weights.shape
        return f"SubjectLayers({C}, {D}, {S})"

class ControlNetEEGConditioningEmbedding(nn.Module):
    """
    Quoting from https://arxiv.org/abs/2302.05543: "Stable Diffusion uses a pre-processing method similar to VQ-GAN
    [11] to convert the entire dataset of 512 × 512 images into smaller 64 × 64 “latent images” for stabilized
    training. This requires ControlNets to convert image-based conditions to 64 × 64 feature space to match the
    convolution size. We use a tiny network E(·) of four convolution layers with 4 × 4 kernels and 2 × 2 strides
    (activated by ReLU, channels are 16, 32, 64, 128, initialized with Gaussian weights, trained jointly with the full
    model) to encode image-space conditions ... into feature maps ..."
    """

    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 128,
        block_out_channels: Tuple[int, ...] = (320, 640, 1280, 2560),
        block_strides: Tuple[int, ...] = (5, 2, 2, 2),
        n_subjects=7,
        x20=False, #TODO: remove this and handle from outside
        is_sd_XL=False,#TODO: remove this and handle from outside
    ):
        super().__init__()
        #TEST
        self.x20 = x20
        self.is_sd_XL = is_sd_XL
        block_strides = (1,1,1,1) if x20 else block_strides
        block_out_channels = (128,128,128,128) if x20 else block_out_channels

        self.subj_layers = SubjectLayers(conditioning_channels, conditioning_channels, n_subjects)
        self.conv_in = nn.Conv1d(conditioning_channels, block_out_channels[0], kernel_size=1, stride=1) if x20 else nn.Conv1d(conditioning_channels, block_out_channels[0], kernel_size=3, stride=1)
        self.blocks = nn.ModuleList([])
        
        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(nn.Conv1d(channel_in, channel_in, kernel_size=1) if x20 else \
                               nn.Conv1d(channel_in, channel_in, kernel_size=3, padding=1))
            self.blocks.append(nn.Conv1d(channel_in, channel_out, kernel_size=block_strides[i + 1], stride=block_strides[i + 1]) if x20 else \
                                nn.Conv1d(channel_in, channel_out, kernel_size=block_strides[i + 1]+1, padding=1, stride=block_strides[i + 1]))
        self.conditioning_embedding_channels = conditioning_embedding_channels
        # self.conv_out = zero_module(
        #     nn.Conv2d(block_out_channels[-1], conditioning_embedding_channels, kernel_size=3, padding=1)
        # )

    def forward(self, conditioning, subjects):
        # #conditioning.shape must be #,128,512 but only in validation there is a 1 dimension that must be removed
        # if conditioning.shape[1] != 128:
        #     conditioning = conditioning.squeeze(1)
        # Calculate the padding needed for the last dimension
        # print(conditioning.shape)
        if (conditioning.shape[1]> conditioning.shape[2]):
            conditioning = conditioning.permute(0,2,1)
        
        padding_needed = 512 - conditioning.shape[2]
        if padding_needed > 0:
            # Apply the padding
            conditioning = F.pad(conditioning, (0, padding_needed))
        # print("egg subj forward", subjects, subjects.shape)
        conditioning = self.subj_layers(conditioning, subjects)
        # print(conditioning.shape)
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)
        #test
        embedding = torch.cat([embedding]*20, dim=0) if self.x20 else embedding
        # print(embedding.shape)
        embedding = embedding.reshape(conditioning.shape[0],
                                      self.conditioning_embedding_channels,
                                      128 if self.is_sd_XL else 64,
                                      128 if self.is_sd_XL else 64) if self.x20 else  embedding.reshape(embedding.shape[0],
                                          self.conditioning_embedding_channels,
                                    embedding.shape[1] // self.conditioning_embedding_channels,
                                      embedding.shape[2])
        # embedding = self.conv_out(embedding)
        embedding = embedding.permute(0, 1, 3, 2)
        # print(embedding.shape)
        if not self.x20:
            # Pad to (4, 320, 64, 64)
            padding = (0, 120, 0, 64) if self.is_sd_XL else (0, 64-embedding.shape[3], 0, 0)# This pads the right side by 120 and the bottom by 64
            # padding = (0, (128 if self.is_sd_XL else 64)-embedding.shape[3], 0, 0)  # Pad the last dimension to 64
            embedding = nn.functional.pad(embedding, padding, mode='constant', value=0)

        return embedding



# x = torch.randn(4, 128, 512)
# weights = torch.randn(6, 3, 3)
# subjects = torch.tensor([4]*x.shape[0])
# print(x.shape, weights.shape)
# #Extract Dimensions:
# _, C, D = weights.shape
# #
# weights = weights.gather(0, subjects.view(-1, 1, 1).expand(-1, C, D)) # shape 4,3,3
# print(torch.einsum("bct,bcd->bdt", x, weights))


