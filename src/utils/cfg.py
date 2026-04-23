import os
import pathlib
import yaml
from argparse import ArgumentParser

def save_all_hparams(trainer, cfg):
    """
    save ckpt
    TODO
    """
    if not os.path.exists(trainer.logger.log_dir):
        os.makedirs(trainer.logger.log_dir)
    save_dict = cfg.__dict__
    save_dict.pop('checkpoint_callback')
    with open(trainer.logger.log_dir + '/hparams.yaml', 'w') as f:
        yaml.dump(save_dict, f)


def build_cfg():
    """
    build ArgumentParser
    Returns:
        ArgumentParser: return ArgumentParser.parse_cfg()
    """
    parser = ArgumentParser()

    # arguments
    parser.add_argument(
        '--config_file',
        type=pathlib.Path,          
        help='If given, experiment configuration will be loaded from this yaml file.',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=2,          
        help='If given, experiment configuration will be loaded from this yaml file.',
    )
    parser.add_argument(
        '--data_path', 
        default=None,   
        type=pathlib.Path,          
        help='If given, experiment configuration will be loaded from this yaml file.',
    )
    parser.add_argument(
        '--experiment_name', 
        default='cellgptv1',   
        type=str,          
        help='Define the project name.',
    )
    parser.add_argument(
        '--gpus', 
        default=4,   
        type=int,          
        help='num_gpus',
    )
    parser.add_argument(
        '--seed', 
        default=3047,   
        type=int,          
        help='seed',
    )
    parser.add_argument(
        '--ckpt_path',
        default=None,   
        type=pathlib.Path,          
        help='Path to the checkpoint.',
    )
    parser.add_argument(
        '--ele',
        default=0,   
        type=int,          
        help='0: none, 1: absolute position encoding (ele), 2: original transformer sinusoidal encoding.',
    )
    cfg = parser.parse_args()
    
    # Load cfg if config file is given
    if cfg.config_file:
        config_file = cfg.config_file
        if config_file.exists():
            with config_file.open('r') as f:
                d = yaml.safe_load(f)
                for k,v in d.items():
                    setattr(cfg, k, v)
        else:
            print('Config file does not exist.')
        
    return cfg
