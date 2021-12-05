import torch
import os
import numpy as np
import time

from utils.utils import adjust_learning_rate, AverageMeter
from utils.utils import get_efficient_net_size, Logger
from utils.data_handler import make_loader


class TrainHandler:

    def __init__(self, model_class, train, config):
        self.model_class = model_class
        self.train = train
        self.config = config
        self.folder_name = os.path.join('model_output', 'finetuning', self.config['global']['folder_name'])
        self.input_shape = get_efficient_net_size(self.config['global']['efficient_net_version'])
        self.logger = Logger(os.path.join(self.folder_name, 'log.txt'))

    def run(self):

        if self.config['global']['train_head_only_model']:
            result_list = []
            self.logger.log('Training head only model')
            for fold in range(self.config['global']['num_folds']):
                self.logger.log('----')
                self.logger.log(f'FOLD: {fold}')
                result_dict = self.run_fold(batch_size=self.config['head_only_model']['batch_size'],
                                            learning_rate=self.config['head_only_model']['learning_rate'],
                                            learning_rate_drop_every=self.config['head_only_model'][
                                                'learning_rate_drop_every'],
                                            learning_rate_drop_factor=self.config['head_only_model'][
                                                'learning_rate_drop_factor'],
                                            fold=fold,
                                            model_ouput_location=os.path.join(self.folder_name, 'head_only_model'),
                                            epochs=self.config['head_only_model']['epochs'],
                                            evaluate_interval_fraction=self.config['head_only_model'][
                                                'evaluate_interval'],
                                            freeze_backbone=True,
                                            load_pretrained=False,
                                            pretrained_model_location=None)
                result_list.append(result_dict)
                self.logger.log('----')

            [self.logger.log("FOLD::" + str(i) + "Loss:: " + str(fold['best_val_loss'])) for i, fold in
             enumerate(result_list)]

        if self.config['global']['train_full_model']:
            result_list = []
            self.logger.log('Training full model')
            for fold in range(self.config['global']['num_folds']):
                self.logger.log('----')
                self.logger.log(f'FOLD: {fold}')
                result_dict = self.run_fold(batch_size=self.config['full_model']['batch_size'],
                                            learning_rate=self.config['full_model']['learning_rate'],
                                            learning_rate_drop_every=self.config['full_model'][
                                                'learning_rate_drop_every'],
                                            learning_rate_drop_factor=self.config['full_model'][
                                                'learning_rate_drop_factor'],
                                            fold=fold,
                                            model_ouput_location=os.path.join(self.folder_name, 'full_model'),
                                            epochs=self.config['full_model']['epochs'],
                                            evaluate_interval_fraction=self.config['full_model']['evaluate_interval'],
                                            freeze_backbone=False,
                                            load_pretrained=True,
                                            pretrained_model_location=os.path.join(self.folder_name, 'head_only_model',
                                                                                   'model' + str(fold) + '.bin')
                                            )
                result_list.append(result_dict)
                self.logger.log('----')

            [self.logger.log("FOLD::" + str(i) + "Loss:: " + str(fold['best_val_loss'])) for i, fold in
             enumerate(result_list)]
        self.logger.save_log()

    def configure_fold(self, batch_size, learning_rate, fold=0):

        model = self.model_class(model_name='efficientnet-b3-pawpularity', config=self.config['global'])

        train_loader, valid_loader = make_loader(self.train, batch_size=batch_size, fold=fold,
                                                 input_shape=self.input_shape)

        # num_update_steps_per_epoch = len(train_loader)
        # max_train_steps = epochs * num_update_steps_per_epoch
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

        #optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum=0.9, weight_decay=1e-4)

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

    def run_fold(self, batch_size, learning_rate, learning_rate_drop_every, learning_rate_drop_factor, fold=0,
                 model_ouput_location='model_output/finetuning/', epochs=1, evaluate_interval_fraction=1,
                 freeze_backbone=False, load_pretrained=False, pretrained_model_location=None):
        model, optimizer, train_loader, valid_loader, result_dict = self.configure_fold(batch_size, learning_rate, fold)

        if freeze_backbone:
            model.freeze_backbone()
            self.logger.log('Backbone freezed')
        else:
            model.unfreeze_backbone()
            self.logger.log('Backbone unfreezed')

        if load_pretrained:
            model.load_state_dict(torch.load(pretrained_model_location))
            self.logger.log('model loaded from: ' + str(model_ouput_location))

        trainer = Trainer(model, optimizer, model_output_location=model_ouput_location, learning_rate=learning_rate,
                          learning_rate_drop_factor=learning_rate_drop_factor,
                          learning_rate_drop_every=learning_rate_drop_every, logger=self.logger,
                          evaluate_interval_fraction=evaluate_interval_fraction)
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


class Trainer:
    def __init__(self, model, optimizer, learning_rate, learning_rate_drop_every, learning_rate_drop_factor,
                 model_output_location, logger, log_interval=1,
                 evaluate_interval_fraction=1):
        self.model = model
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.learning_rate_drop_every = learning_rate_drop_every
        self.learning_rate_drop_factor = learning_rate_drop_factor
        self.log_interval = log_interval
        self.evaluate_interval_fraction = evaluate_interval_fraction
        self.evaluator = Evaluator(self.model)
        self.model_output_location = model_output_location
        self.logger = logger

    def train(self, train_loader, valid_loader, epoch, result_dict, fold):
        count = 0
        losses = AverageMeter()

        new_lr = adjust_learning_rate(self.optimizer, epoch, self.learning_rate, self.learning_rate_drop_every,
                                      self.learning_rate_drop_factor)
        self.logger.log('Learning rate dropped to: ' + str(new_lr))

        self.model.train()

        evaluate_interval = int((len(train_loader) - 1) * self.evaluate_interval_fraction)
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

                self.logger.log(', '.join(ret))

            if batch_idx != 0 and batch_idx % evaluate_interval == 0:
                result_dict = self.evaluator.evaluate(
                    valid_loader,
                    epoch,
                    result_dict
                )
                if result_dict['val_loss'][-1] < result_dict['best_val_loss']:
                    self.logger.log("{} epoch, best epoch was updated! valid_loss: {: >4.5f}".format(epoch,
                                                                                                     result_dict[
                                                                                                         'val_loss'][
                                                                                                         -1]))
                    result_dict["best_val_loss"] = result_dict['val_loss'][-1]
                    torch.save(self.model.state_dict(), os.path.join(self.model_output_location, f"model{fold}.bin"))
                self.model.train()
        result_dict['train_loss'].append(losses.avg)
        return result_dict


class Evaluator:
    def __init__(self, model):
        self.model = model

    def evaluate(self, data_loader, epoch, result_dict):
        losses = AverageMeter()

        self.model.eval()
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
