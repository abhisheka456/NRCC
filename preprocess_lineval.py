# -*- coding: UTF-8 -*-

import torch

from models.psclr.psclr_wrapper import PSCLRWrapper
from network import backbone_dict

if __name__ == '__main__':
    

    # backbone = 'bigresnet18'
    # backbone = 'bigresnet18'
    backbone = 'resnet50'
    encoder_type, dim_in = backbone_dict[backbone]
    encoder = encoder_type()
    byol = PSCLRWrapper(encoder,
                       num_cluster=10,
                       in_dim=dim_in,
                       temperature=0.5,
                       hidden_size=4096,
                       fea_dim=256,
                       byol_momentum=0.999,
                       symmetric=True,
                       shuffling_bn=True,
                       latent_std=0.001)

    checkpoint = '/Data2/akumar/ProPos/ckpt/2023_09_28_04_32_57-imagenet_r50_psclr2/save_models/byol-200'
    msg = byol.load_state_dict(torch.load(checkpoint, map_location='cpu'), strict=True)
    encoder = byol.encoder_k
    torch.save(encoder.state_dict(), '/Data2/akumar/ProPos/ckpt/encoder/imagenet/encoder_checkpoint.pth')
