import yaml
import pandas as pd
from utils import set_seed, create_folds
from finetuning import FineTuning


with open("config.yaml", 'r') as stream:
    ft_config = yaml.safe_load(stream)

set_seed(ft_config['global']['seed'])

train = pd.read_csv('data/raw/train.csv')
train = create_folds(train, ft_config['global']['num_folds'], ft_config['global']['seed'])

ft_exec = FineTuning(train, config=ft_config)
ft_exec.run()
