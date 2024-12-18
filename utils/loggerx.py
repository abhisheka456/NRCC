# -*- coding: UTF-8 -*-

import torch
import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
from typing import Union
import os.path as osp
from glob import glob
import os
import time
from torchvision.utils import save_image
import torch.distributed as dist
import math
import inspect
# from torch._six import string_classes
import collections.abc as container_abcs
import warnings
from utils.ops import load_network

string_classes = str


def get_varname(var):
    """
    Gets the name of var. Does it from the out most frame inner-wards.
    :param var: variable to get name from.
    :return: string
    """
    for fi in reversed(inspect.stack()):
        names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
        if len(names) > 0:
            return names[0]


def reduce_tensor(rt):
    rt = rt.clone()
    if dist.is_initialized():
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        world_size = dist.get_world_size()
    else:
        world_size = 1
    rt /= world_size
    return rt


class LoggerX(object):

    def __init__(self, save_root, enable_wandb=False, **kwargs):
        assert dist.is_initialized()
        self.models_save_dir = osp.join(save_root, 'save_models')
        self.images_save_dir = osp.join(save_root, 'save_images')
        os.makedirs(self.models_save_dir, exist_ok=True)
        os.makedirs(self.images_save_dir, exist_ok=True)
        self._modules = []
        self._module_names = []
        self.world_size = dist.get_world_size()
        self.local_rank = dist.get_rank()
        self.enable_wandb = enable_wandb
        if enable_wandb and self.local_rank == 0:
            import wandb
            wandb.init(dir=save_root, settings=wandb.Settings(_disable_stats=True, _disable_meta=True), **kwargs)

    @property
    def modules(self):
        return self._modules

    @property
    def module_names(self):
        return self._module_names

    @modules.setter
    def modules(self, modules):
        for i in range(len(modules)):
            self._modules.append(modules[i])
            self._module_names.append(get_varname(modules[i]))

    def append(self, module, name=None):
        self._modules.append(module)
        if name is None:
            name = get_varname(module)
        self._module_names.append(name)

    def checkpoints(self, epoch):
        if self.local_rank != 0:
            return
        for i in range(len(self.modules)):
            module_name = self.module_names[i]
            module = self.modules[i]
            # delete other saved model
            model_path = osp.join(self.models_save_dir, f'{module_name}-{epoch-1}')
            if osp.exists(model_path):
                os.remove(model_path)
            torch.save(module.state_dict(), osp.join(self.models_save_dir, '{}-{}'.format(module_name, epoch)))

    def best_checkpoints(self, results):
        if self.local_rank != 0:
            return
        output_str = ''
        for i in range(3):
            var_name, var = list(results.items())[i]
            output_str += '-{}-{:2.2f}'.format(var_name[10:], var)
        for i in range(len(self.modules)):
            module_name = self.module_names[i]
            module = self.modules[i]
            # delete other saved model
            model_path = glob(osp.join(self.models_save_dir, f'{module_name}-best*'))
            if model_path == []:
                warnings.warn(f'No best checkpoint found for {module_name}')
                model_path = ''
            else:
                model_path = model_path[0]
            if osp.exists(model_path):
                os.remove(model_path)
            torch.save(module.state_dict(), osp.join(self.models_save_dir, f'{module_name}-best'+output_str))



    def load_checkpoints(self, epoch):
        for i in range(len(self.modules)):
            module_name = self.module_names[i]
            module = self.modules[i]
            module.load_state_dict(load_network(osp.join(self.models_save_dir, '{}-{}'.format(module_name, epoch))))

    def load_best_checkpoints(self):
        for i in range(len(self.modules)):
            module_name = self.module_names[i]
            module = self.modules[i]
            model_path = glob(osp.join(self.models_save_dir, f'{module_name}-best*'))
            if model_path == []:
                warnings.warn(f'No best checkpoint found for {module_name}')
                model_path = ''
            else:
                model_path = model_path[0]
            module.load_state_dict(load_network(model_path))

    def load_restart_checkpoints(self, epoch, restart_dir):
        for i in range(len(self.modules)):
            module_name = self.module_names[i]
            if module_name == 'byol':
                module = self.modules[i]
                module.load_state_dict(load_network(osp.join('ckpt',restart_dir,'save_models',  '{}-{}'.format(module_name, epoch))))

    def msg(self, stats, step):
        output_str = '[{}] {:05d}, '.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), step)
        output_dict = {}
        for i in range(len(stats)):
            if isinstance(stats, (list, tuple)):
                var = stats[i]
                var_name = get_varname(stats[i])
            elif isinstance(stats, dict):
                var_name, var = list(stats.items())[i]
            else:
                raise NotImplementedError
            if isinstance(var, torch.Tensor):
                var = var.detach().mean()
                var = reduce_tensor(var)
                var = var.item()
            output_str += '{} {:2.5f}, '.format(var_name, var)
            output_dict[var_name] = var

        if self.enable_wandb and self.local_rank == 0:
            import wandb
            wandb.log(output_dict, step)

        if self.local_rank == 0:
            print(output_str)

    def msg_str(self, output_str):
        if self.local_rank == 0:
            print(str(output_str))

    def save_image(self, grid_img, n_iter, sample_type):
        save_image(grid_img, osp.join(self.images_save_dir,
                                      '{}_{}_{}.jpg'.format(n_iter, self.local_rank, sample_type)),
                   nrow=1)
