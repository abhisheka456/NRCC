# -*- coding: UTF-8 -*-


import torch
import torch.distributed as dist
import numpy as np
import random
from PIL import ImageFile
import sys
import yaml
import os.path as osp
from datetime import datetime

ImageFile.LOAD_TRUNCATED_IMAGES = True

from models.basic_template import TrainTask
from models import model_dict

"""Parses command line arguments from the model-specific parser, 
using the default options namespace to populate any missing args.

This allows model-specific args to override the default ones.
"""
"""Parses command-line arguments from model-specific parser using 
unknown options and default namespace.

Merges model-specific options with default common options, allowing
model-specific options to override defaults. Returns namespace 
containing merged options."""
if __name__ == '__main__':
    # Parses configuration from YAML file and initializes model, distributed training, 
    # random seeds, and other runtime settings. Main entry point for training script.
    config_path = sys.argv[2]    # Parses the configuration file path from the command line arguments

    
    with open(config_path) as f:
        # Load YAML configuration file
        # Use FullLoader if available, otherwise use default loader
        if hasattr(yaml, 'FullLoader'):
            configs = yaml.load(f.read(), Loader=yaml.FullLoader)
        else:
            configs = yaml.load(f.read())

    MODEL = model_dict[configs['model_name']]    # Loads the model class from the model dictionary based on the 
                                                 # model name specified in the YAML config file.

    default_parser = TrainTask.build_default_options()    # Builds the default command line argument parser with common options 
                                                          # for training tasks.
    default_opt, unknown_opt = default_parser.parse_known_args('')    # Parses known command-line arguments from default parser, ignoring unknown args.
                                                                      # Returns namespace object with known args and list of unknown args.
    private_parser = MODEL.build_options()    # Builds the command line argument parser containing model-specific options.
    opt = private_parser.parse_args(unknown_opt, namespace=default_opt)    # Parses command-line arguments from model-specific parser, merging them into the default options namespace. This allows model-specific options to override the defaults. The returned opt contains the merged options.


    if opt.run_name is None:
        # Sets the run name to be the base name of the config path, removing the file extension, if no run name was specified.
        # This allows the run name to default to the config file name.
        opt.run_name = osp.basename(config_path)[:-4]
    opt.run_name = '{}-{}'.format(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"), opt.run_name)    # Sets the run name to be a formatted string containing the current 
                                                                                                 # date and time, along with the original run name. This ensures each run has a unique name including a timestamp.

    for k in configs:
        # Sets attributes on the opt namespace object to values from the configs dictionary.
        # This populates the opt namespace with configuration values loaded from the YAML file.
        setattr(opt, k, configs[k])

    if opt.dist:
        # Initializes distributed PyTorch using the NCCL backend and environment variable 
        # for communication. Sets the CUDA device to the rank of the process.
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(dist.get_rank())

    if opt.num_devices > 0:
        # Checks that the number of GPU devices specified matches the number actually available.
        # This ensures we don't try to allocate more GPU memory than is available.
        assert opt.num_devices == torch.cuda.device_count()  # total batch size

    seed = opt.seed
    # Sets random seeds for reproducibility. Sets the main seed value 
    # from the opt namespace, and applies it to PyTorch, NumPy and 
    # Python's random module.    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = MODEL(opt)    # Initializes an instance of the model class specified in opt.model_name, 
    # passing the opt namespace containing the parsed command line arguments.
    # This creates the model object that will be trained.

    model.sghmc_distance()
    # model.sghmc_save()    # Saves extracted features from the model for later use.
    # This allows feature extraction without needing to rerun 
    # the model, improving efficiency for downstream tasks.
    
