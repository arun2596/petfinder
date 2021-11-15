import os

import yaml
import pandas as pd
from utils import set_seed, create_folds
from finetuning import FineTuning
from shutil import copyfile
from datetime import date


def parse_config(config):
    # used to check if config makes sense
    # make later versions of config backwards comaptible

    today = date.today()
    folder_prefix = str(today.strftime("%Y%m%d"))

    folder_list = os.listdir('model_output/finetuning')
    folder_suffixes = [x.split('-')[-1] for x in folder_list]

    if '-' in config['global']['folder_suffix']:
        raise Exception('suffix connot contain "-"')
    elif config['global']['folder_suffix'] in folder_suffixes:
        raise Exception('folder with the suffix already exists')
    else:
        os.makedirs(os.path.join('model_output', 'finetuning', folder_prefix + '-' + config['global']['folder_suffix']))
        config['global']['folder_name'] = folder_prefix + '-' + config['global']['folder_suffix']
        os.makedirs(os.path.join('model_output', 'finetuning', config['global']['folder_name'], 'head_only_model'))
        os.makedirs(os.path.join('model_output', 'finetuning', config['global']['folder_name'], 'full_model'))
    
    copyfile('config.yaml',os.path.join(config['global']['folder_name'],'config.yaml'))
    return config


with open("config.yaml", 'r') as stream:
    ft_config = yaml.safe_load(stream)

ft_config = parse_config(ft_config)

#
set_seed(ft_config['global']['seed'])

train = pd.read_csv('data/raw/train.csv')
train = create_folds(train, ft_config['global']['num_folds'], ft_config['global']['seed'])

ft_exec = FineTuning(train, config=ft_config)
ft_exec.run()
