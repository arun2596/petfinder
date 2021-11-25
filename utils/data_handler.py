import cv2
import os
from torchvision import transforms
import torch
from torch.utils.data import (
    Dataset, DataLoader,
    SequentialSampler, RandomSampler
)

from utils.utils import SquarePad, Rescale, Normalize, Rerange, FlipLR


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
