import torch
import torchvision
import torchvision.transforms.functional as F

import numpy as np
import random

from sklearn import model_selection

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return


def get_efficient_net_size(name):
    input_shapes = {
        'efficientnet-b0': (224,224),
        'efficientnet-b1': (240, 240),
        'efficientnet-b2': (260, 260),
        'efficientnet-b3': (300, 300),
        'efficientnet-b4': (380, 380),
        'efficientnet-b5': (456, 456),
        'efficientnet-b6': (528, 528),
        'efficientnet-b7': (600, 600)
    }

    return input_shapes[name]




def create_folds(data, num_splits, seed):
    data["kfold"] = -1
    kf = model_selection.KFold(n_splits=num_splits, shuffle=True, random_state=seed)
    for f, (t_, v_) in enumerate(kf.split(X=data)):
        data.loc[v_, 'kfold'] = f
    return data


def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return


class Logger():
    def __init__(self, output_location):
        self.body = []
        self.header = []
        self.footer = []
        self.output_location = output_location

    def log(self, text, place='body'):
        if place == 'header':
            self.header.append(text)
        elif place == 'footer':
            self.footer.append(text)
        elif place == 'body':
            self.body.append(text)

        print(text)
        return

    def save_log(self):
        with open(self.output_location, 'a+') as the_file:
            for line in self.header:
                the_file.write(str(line)+'\n')
            the_file.write('-'*50)
            the_file.write('\n')
            for line in self.body:
                the_file.write(str(line) + '\n')
            the_file.write('-' * 50)
            the_file.write('\n')
            for line in self.footer:
                the_file.write(str(line) + '\n')
            the_file.write('-' * 50)
            the_file.write('\n')
        return

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = 0
        self.min = 1e5

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if val > self.max:
            self.max = val
        if val < self.min:
            self.min = val


class SquarePad(object):
    def __call__(self, sample):
        image, target = sample['image'], sample['target']

        img_size = image.size()[1:]
        max_wh = max(img_size)

        p_left, p_top = [(max_wh - s) // 2 for s in img_size]
        p_right, p_bottom = [max_wh - (s + pad) for s, pad in zip(img_size, [p_left, p_top])]
        padding = [p_left, p_top, p_right, p_bottom]
        return {'image': F.pad(image, padding, 0, 'constant'), 'target': target}


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, target = sample['image'], sample['target']

        h, w = image.shape[1:]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = F.resize(image, (new_h, new_w))

        return {'image': img, 'target': target}


class Rerange(object):

    def __call__(self, sample):
        image, target = sample['image'], sample['target']

        return {'image': image / 255.0, 'target': target}


class Normalize(object):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, target = sample['image'], sample['target']

        img = F.normalize(image, mean=self.mean, std=self.std)

        return {'image': img, 'target': target}


class FlipLR(object):

    def __init__(self, proba):
        self.proba = proba

    def __call__(self, sample):
        image, target = sample['image'], sample['target']

        img = torchvision.transforms.RandomHorizontalFlip(p=self.proba)(image)

        return {'image': img, 'target': target}


