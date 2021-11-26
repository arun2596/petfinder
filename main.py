import os
import yaml
import pandas as pd
from datetime import date
from shutil import copyfile

from finetuning import PawpularityModel
from pretraining import Pawpularity2018Model

from utils.train_handler import TrainHandler
from utils.utils import set_seed, create_folds


def parse_config(config):
    # used to check if config makes sense
    # make later versions of config backwards comaptible

    today = date.today()
    folder_prefix = str(today.strftime("%Y%m%d"))

    if config['global']['mode'] not in ['pretraining', 'finetuning']:
        raise Exception('Unknown mode in global config settings: ' + str(config['global']['mode']))

    folder_list = os.listdir(os.path.join('model_output', config['global']['mode']))
    folder_suffixes = [x.split('-')[-1] for x in folder_list]

    if '-' in config['global']['folder_suffix']:
        raise Exception('suffix connot contain "-"')
    elif config['global']['folder_suffix'] in folder_suffixes:
        raise Exception('folder with the suffix already exists')
    else:

        os.makedirs(os.path.join('model_output', config['global']['mode'], folder_prefix + '-' + config['global']['folder_suffix']))
        config['global']['folder_name'] = folder_prefix + '-' + config['global']['folder_suffix']
        os.makedirs(os.path.join('model_output', config['global']['mode'], config['global']['folder_name'], 'head_only_model'))
        os.makedirs(os.path.join('model_output', config['global']['mode'], config['global']['folder_name'], 'full_model'))

    if config['global']['mode'] == 'pretraining':
        config['global']['num_folds'] = 1

    if config['global']['load_from_pretrained']:
        if config['global']['mode'] == 'pretraining':
            raise Exception('Cannot load pretrained weights for pretraining step. Change "load_from_pretrained" to False')
        pretrained_folder_list = os.listdir(os.path.join('model_output', 'pretraining'))

        if not config['global']['pretrained_model_location']:
            raise Exception('pretrained_model_location is missing in config')

        if config['global']['pretrained_model_location'] in pretrained_folder_list:
            config['global']['pretrained_folder_name'] = os.path.join('model_output', 'pretraining', config['global']['pretrained_model_location'])
        else:
            raise Exception('pretrained_model_location folder not found')

    copyfile('config.yaml', os.path.join('model_output', config['global']['mode'], config['global']['folder_name'], 'config.yaml'))

    return config


def run_config():
    with open("config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)
    config = parse_config(config)
    if config['global']['mode'] == 'finetuning':
        run_finetuning(config)
    elif config['global']['mode'] == 'pretraining':
        run_pretraining(config)
    else:
        raise Exception('Unknown mode in global config settings: ' + str(config['global']['mode']))


def run_finetuning(config):
    set_seed(config['global']['seed'])

    train = pd.read_csv(os.path.join('data', 'raw', 'train.csv'))
    train = create_folds(train, config['global']['num_folds'], config['global']['seed'])

    train.rename({'Pawpularity':'Target'}, inplace=True, axis=1)

    config['global']['train_image_root_dir'] = os.path.join('data', 'raw', 'train')
    config['global']['test_image_root_dir'] = os.path.join('data', 'raw', 'test')

    ft_handler = TrainHandler(model_class=PawpularityModel, train=train, config=config)
    ft_handler.run()


def run_pretraining(config):

    set_seed(config['global']['seed'])

    train = pd.read_csv(os.path.join('data', '2018_data', 'train', 'train.csv'))
    train.rename({'PetID':'Id', 'AdoptionSpeed': 'Target'}, inplace=True, axis=1)
    train = train[train['PhotoAmt']!=0].reset_index(drop=True)

    train = create_folds(train, config['global']['num_folds'], config['global']['seed'], cross_validation=False)

    train['Id'] += '-1'

    config['global']['train_image_root_dir'] = os.path.join('data', '2018_data', 'train_images')
    config['global']['test_image_root_dir'] = os.path.join('data', '2018_data', 'test_images')

    ft_handler = TrainHandler(model_class=Pawpularity2018Model, train=train, config=config)
    ft_handler.run()


if __name__ == '__main__':
    run_config()
