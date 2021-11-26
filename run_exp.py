from ruamel import yaml
from main import *
from datetime import datetime
import time
import shutil

lr = [0.0005, 0.005, 0.05]

for i, l in enumerate(lr):
    print("Learning Rate set to ", l)

    # Open config file
    with open("config.yaml") as f:
        config = yaml.load(f)

    # Set the path to save models to drive
    config['global']['save_to_drive'] = True
    today = datetime.now()
    folder_prefix = str(today.strftime("%Y_%m_%d__%H:%M:%S"))
    OUT_PATH = os.pat.join(config['global']['drive_save_path'], folder_prefix, "_model_outputs")

    config['global']['folder_suffix'] = 'model_' + folder_prefix + '__' + str(i)

    # Learning rate & Batch size of head only model
    config['head_only_model']['batch_size'] = 32
    config['head_only_model']['learning_rate'] = l

    # Learning rate & Batch size of full model
    config['full_model']['batch_size'] = 32
    config['full_model']['learning_rate'] = l

    # Close config file
    with open("config.yaml", 'w') as f:
        doc = yaml.dump(config, f, default_flow_style=False, allow_unicode=True, encoding=None)

    # Execute the experiment
    run_config()

    # Save model and config to drive
    with open("config.yaml", 'r') as stream:
        ft_config = yaml.safe_load(stream)

    ft_config = parse_config(ft_config)

    if ft_config['global']['save_to_drive']:
        MODEL_PATH = os.path.join("model_output","finetuning" , ft_config['global']['folder_name'])
        shutil.copytree(MODEL_PATH, OUT_PATH)

