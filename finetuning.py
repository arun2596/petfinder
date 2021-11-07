import torch
import torch.nn as nn
import torch.optim as optim
from efficientnet_pytorch import EfficientNet
from torch.optim import lr_scheduler
from torch.utils.data import (
    Dataset, DataLoader,
    SequentialSampler, RandomSampler
)

import torchvision
from torchvision import models, transforms

import efficientnet_pytorch

import cv2

import time
import os
import copy
import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import random

from sklearn.metrics import mean_squared_error
# ------------------------- CUSTOM IMPORTS & UTILS --------------------------------

from utils import set_seed, adjust_learning_rate, SquarePad, Rescale, Normalize, Rerange, FlipLR, AverageMeter, \
    get_efficient_net_size


# -------------------------- CREATE DATASET ----------------------------------------

class DatasetRetriever(Dataset):
    def __init__(self, data, root_dir, mode='train', transform=None):
        self.data = data

        self.img_id = self.data.Id.values.tolist()
        self.targets = self.data.Pawpularity.values.tolist()

        self.mode = mode

        if self.mode == 'train' or self.mode == 'valid':
            self.data_dir = os.path.join(root_dir, 'train')
        elif self.mode == 'test':
            self.data_dir = os.path.join(root_dir, 'test')
        else:
            raise Exception("Invalid mode: " + str(self.mode))

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img_id, target = self.img_id[item], self.targets[item]

        img_name = os.path.join(self.data_dir, img_id + '.jpg')

        image = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2RGB)

        # NOTE: Images is transposed from (H, W, C) to (C, H, W)
        sample = {
            'image': torch.tensor(image.transpose((2, 0, 1)), dtype=torch.float),
            'target': torch.tensor(target, dtype=torch.float),
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


def make_loader(
        data,
        batch_size,
        input_shape=(528, 528),
        fold=0,
        root_dir=os.path.join('data', 'raw')
):
    dataset = {'train': data[data['kfold'] != fold], 'valid': data[data['kfold'] == fold]}

    transform = {'train': transforms.Compose([
        SquarePad(),
        Rescale(input_shape),
        Rerange(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225]),
        FlipLR(0.5)
    ]),
        'valid': transforms.Compose([
            SquarePad(),
            Rescale(input_shape),
            Rerange(),
            Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])
        ])}

    train_dataset, valid_dataset = [
        DatasetRetriever(dataset[mode], transform=transform[mode], root_dir=root_dir, mode=mode) for mode in
        ['train', 'valid']]

    train_sampler = RandomSampler(dataset['train'])
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        pin_memory=True,
        drop_last=False,
        num_workers=4
    )

    valid_sampler = SequentialSampler(dataset['valid'])
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size // 2,
        sampler=valid_sampler,
        pin_memory=True,
        drop_last=False,
        num_workers=4
    )

    return train_loader, valid_loader


# data = pd.read_csv('data/raw/train.csv')
# data['kfold'] = [0] * 10 + [1] * (data.shape[0] - 10)
#
# tl, vl = make_loader(
#     data,
#     20,
#     fold=0
# )

# print(next(iter(tl))['image'][0])

# b, k = 3, 5
#
# for i in range(b):
#     img = next(iter(tl))
# print(img['target'][k])
# img = img['image'][k].cpu().detach().numpy().transpose((1, 2, 0))
#
# plt.imshow(img)
# plt.show()


# -------------------------- CREATE MODEL ----------------------------------------

class PawpularityModel(nn.Module):
    def __init__(
            self,
            model_name,
            efficient_net_version,
            multisample_dropout=False
    ):
        super(PawpularityModel, self).__init__()
        self.model_name = model_name
        self.backbone = EfficientNet.from_pretrained(efficient_net_version)
        self.backbone._fc = nn.Identity()

        self.hidden_size = self.get_hidden_size()

        if multisample_dropout:
            self.dropouts = nn.ModuleList([
                nn.Dropout(0.5) for _ in range(5)
            ])
        else:
            self.dropouts = nn.ModuleList([nn.Dropout(0.3)])
        self.regressor = nn.Linear(self.hidden_size, 1)

        self._init_weights(self.regressor)

    def get_hidden_size(self):
        x = torch.randn(1, 3, 300, 300)
        return self.backbone(x).shape[-1]

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            init_range = 1.0 / math.sqrt(module.weight.shape[1])
            module.weight.data.uniform_(-init_range, init_range)

            # if module.bias is not None:
            #     module.bias.data.zero_()

        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def freeze_backbone(self):
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def unfreeze_backbone(self):
        for parameter in self.backbone.parameters():
            parameter.requires_grad = True

    def forward(
            self,
            image=None,
            target=None
    ):

        backbone_output = self.backbone(image)

        # layer_norm_out = self.layer_norm(backbone_output)

        # multi-sample dropout
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                logits = self.regressor(dropout(backbone_output))
            else:
                logits += self.regressor(dropout(backbone_output))

        logits /= len(self.dropouts)

        logits = torch.clip(logits, min=0, max=100)

        #
        #         # calculate loss
        loss = None
        if target is not None:
            # regression task
            loss_fn = torch.nn.MSELoss()
            logits = logits.view(-1).to(target.dtype)
            # print(logits)
            loss = torch.sqrt(loss_fn(logits, target.view(-1)) + 1e-6)

        return (loss, logits) if loss is not None else logits


class Trainer:
    def __init__(self, model, optimizer, model_output_location, log_interval=1,
                 evaluate_interval=130):
        self.model = model
        self.optimizer = optimizer
        self.log_interval = log_interval
        self.evaluate_interval = evaluate_interval
        self.evaluator = Evaluator(self.model)
        self.model_output_location = model_output_location

    def train(self, train_loader, valid_loader, epoch,
              result_dict, fold):
        count = 0
        losses = AverageMeter()
        weighted_losses = AverageMeter()
        self.model.train()

        for batch_idx, batch_data in enumerate(train_loader):
            image, target = batch_data['image'], batch_data['target']
            image, target = image.cuda(), target.cuda()

            outputs = self.model(
                image=image,
                target=target
            )

            loss, logits = outputs
            count += target.size(0)
            losses.update(loss.item(), target.size(0))  # ------ may need to change this ?

            loss.backward()

            self.optimizer.step()

            self.optimizer.zero_grad()

            if batch_idx % self.log_interval == 0:
                _s = str(len(str(len(train_loader.sampler))))

                ret = [
                    ('epoch: {:0>3} [{: >' + _s + '}/{} ({: >3.0f}%)]').format(epoch, count,
                                                                               len(train_loader.sampler),
                                                                               100 * count / len(
                                                                                   train_loader.sampler)),
                    'train_loss: {: >4.5f}'.format(losses.avg),
                ]

                print(', '.join(ret))

            if batch_idx != 0 and batch_idx % self.evaluate_interval == 0:
                result_dict = self.evaluator.evaluate(
                    valid_loader,
                    epoch,
                    result_dict
                )
                if result_dict['val_loss'][-1] < result_dict['best_val_loss']:
                    print("{} epoch, best epoch was updated! valid_loss: {: >4.5f}".format(epoch,
                                                                                           result_dict['val_loss'][-1]))
                    result_dict["best_val_loss"] = result_dict['val_loss'][-1]
                    torch.save(self.model.state_dict(), self.model_output_location + f"model{fold}.bin")
                self.model.train()
        result_dict['train_loss'].append(losses.avg)
        return result_dict


class Evaluator:
    def __init__(self, model):
        self.model = model

    def evaluate(self, data_loader, epoch, result_dict):
        losses = AverageMeter()

        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(data_loader):
                image, target = batch_data['image'], batch_data['target']
                image, target = image.cuda(), target.cuda()

                outputs = self.model(
                    image=image,
                    target=target
                )

                loss, logits = outputs
                # print(loss)
                losses.update(loss.item(), target.size(0))

        print('----Validation Results Summary----')
        print('Epoch: [{}] valid_loss: {: >4.5f}'.format(epoch, losses.avg))

        result_dict['val_loss'].append(losses.avg)
        return result_dict


# SEND EFFICIENT NET MODEL NAME
# UPDATE WEIGHT DECAY
# MODEL FOLDER NAME MATCHING
# WHAT NAME DOES IT SAVE MODEL WITH ?
# CREATE FOLDERS :( .. UGHH



class FineTuning:

    def __init__(self, train, config):
        self.train = train
        self.config = config
        self.folder_name = os.path.join('model_output', 'finetuning', self.config['global']['folder_prefix'])
        self.input_shape = get_efficient_net_size(self.config['global']['efficient_net_version'])

    def run(self):

        if self.config['global']['train_head_only_model']:
            result_list = []
            for fold in range(self.config['global']['num_folds']):
                print('----')
                print(f'FOLD: {fold}')
                result_dict = self.run_fold(batch_size=self.config['global']['head_only_model']['batch_size'],
                                            learning_rate=self.config['global']['head_only_model']['learning_rate'],
                                            fold=fold,
                                            model_ouput_location='model_output/finetuning/',
                                            epochs=self.config['global']['head_only_model']['epochs'],
                                            freeze_backbone=True,
                                            load_pretrained=False,
                                            pretrained_model_location=None)
                result_list.append(result_dict)
                print('----')

            [print("FOLD::", i, "Loss:: ", fold['best_val_loss']) for i, fold in enumerate(result_list)]

        if self.config['global']['train_fullmodel']:
            result_list = []
            for fold in range(self.config['global']['num_folds']):
                print('----')
                print(f'FOLD: {fold}')
                result_dict = self.run_fold(batch_size=self.config['global']['full_model']['batch_size'],
                                            learning_rate=self.config['global']['full_model']['learning_rate'],
                                            fold=fold,
                                            model_ouput_location='model_output/finetuning/',
                                            epochs=self.config['global']['full_model']['epochs'],
                                            freeze_backbone=True,
                                            load_pretrained=False,
                                            pretrained_model_location=None)
                result_list.append(result_dict)
                print('----')

            [print("FOLD::", i, "Loss:: ", fold['best_val_loss']) for i, fold in enumerate(result_list)]

    def configure_fold(self, batch_size, learning_rate, fold=0):

        model = PawpularityModel('efficientnet-b3-pawpularity', self.config['global']['efficient_net_version'])

        train_loader, valid_loader = make_loader(self.train, batch_size=batch_size, fold=fold, input_shape=self.input_shape)

        num_update_steps_per_epoch = len(train_loader)
        max_train_steps = epochs * num_update_steps_per_epoch

        optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum=0.9, weight_decay=1e-4)

        if torch.cuda.device_count() >= 1:
            print('Model pushed to {} GPU(s), type {}.'.format(
                torch.cuda.device_count(),
                torch.cuda.get_device_name(0))
            )
            model = model.cuda()
        else:
            raise ValueError('CPU training is not supported')

        result_dict = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'best_val_loss': np.inf
        }
        return (
            model,
            optimizer,
            train_loader,
            valid_loader,
            result_dict
        )

    def run_fold(self, batch_size, learning_rate, fold=0, model_ouput_location='model_output/finetuning/', epochs=1,
                 freeze_backbone=False, load_pretrained=False, pretrained_model_location=None):
        model, optimizer, train_loader, valid_loader, result_dict = self.configure_fold(batch_size, learning_rate, fold)

        if freeze_backbone:
            model.freeze_backbone()
        else:
            model.unfreeze_backbone()

        if load_pretrained:
            model.load_state_dict(torch.load(pretrained_model_location))

        trainer = Trainer(model, optimizer, model_ouput_location)
        train_time_list = []

        for epoch in range(epochs):
            # adjust_learning_rate(optimizer, epoch, 0.1)

            result_dict['epoch'] = epoch

            torch.cuda.synchronize()
            tic1 = time.time()

            result_dict = trainer.train(train_loader, valid_loader, epoch, result_dict, fold)

            torch.cuda.synchronize()
            tic2 = time.time()
            train_time_list.append(tic2 - tic1)

        torch.cuda.empty_cache()
        del model, optimizer, train_loader, valid_loader

        return result_dict


# def main():
#     train = pd.read_csv('data/raw/train.csv')
#     test = pd.read_csv('data/raw/test.csv')
#
#     set_seed(0)
#
#     validation_proportion = 0.15
#
#     target_frequency = train[['Id', 'Pawpularity']].groupby('Pawpularity').count().reset_index()
#     target_frequency['Proba'] = sum(target_frequency['Id']) / target_frequency['Id']
#     target_frequency['Proba'] = target_frequency['Proba'] / sum(target_frequency['Proba'])
#
#     row_proba = \
#         train[['Pawpularity']].merge(target_frequency, how='left', left_on='Pawpularity', right_on='Pawpularity')[
#             'Proba']
#     row_proba = row_proba / sum(row_proba)
#
#     indices = np.random.choice(train.index, size=int(train.shape[0] * validation_proportion), replace=False,
#                                p=row_proba)
#
#     train['kfold'] = 1
#     train.loc[indices, 'kfold'] = 0
#
#     model_output_location = 'model_output/finetuning/'
#
#     result_list = []
#     for fold in range(1):
#         print('----')
#         print(f'FOLD: {fold}')
#         result_dict = run(train, fold, model_output_location, freeze_backbone=True)
#         result_list.append(result_dict)
#         print('----')
#
#     [print("FOLD::", i, "Loss:: ", fold['best_val_loss']) for i, fold in enumerate(result_list)]
#
#     oof = np.zeros(len(train))
#     for fold in tqdm.tqdm(range(1), total=1):
#         model = PawpularityModel('efficientnet-b3-pawpularity')
#         model.load_state_dict(
#             torch.load(model_output_location + f'model{fold}.bin')
#         )
#         model.cuda()
#         model.eval()
#         val_index = train[train.kfold == fold].index.tolist()
#         train_loader, val_loader = make_loader(train, batch_size=16, fold=fold)
#
#         preds = []
#         for index, data in enumerate(val_loader):
#             image, target = data['image'], data['target']
#
#             image, target = image.cuda(), target.cuda()
#
#             outputs = model(
#                 image=image,
#                 target=target
#             )
#
#             loss, logits = outputs
#             preds += logits.cpu().detach().numpy().tolist()
#         oof[val_index] = preds
#
#     print("cv", round(np.sqrt(mean_squared_error(train.Pawpularity.values[val_index], oof[val_index])), 4))
#
#
# main()

# backbone = EfficientNet.from_pretrained('efficientnet-b3')
# backbone._fc = nn.Identity()
# add_layer_norm = nn.LayerNorm(1536)(backbone(x))
# add_dropout = nn.Dropout(0.3)(add_layer_norm)
# regressor = nn.Linear(1536, 1)(add_dropout)


# x = torch.randn(10, 3, 300, 300)


# print(backbone._fc)

# a = add_dropout.detach().numpy()
# print(regressor.detach().shape)
# print()

# print(.shape)

#
# print([x for x in backbone.named_modules()])

# print(dir(backbone))


## ADD CODE TO TRAIN FRON LAYERS FIRST AND THEN BACK LAYERS
## SEED
