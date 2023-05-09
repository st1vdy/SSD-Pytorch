import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import List, Tuple

from data_reader import VOC2007Dataset
train_data_root = '/home/dy/python_projects/SSD/data/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007'

batch_size = 32
num_workers = 0
image_mean = 107.4748
image_std = 61.3333

def collate_fn(batch: List[Tuple[torch.Tensor, np.ndarray]]):
    images = []
    targets = []

    for img, target in batch:
        images.append(img.to(torch.float32))
        targets.append(torch.Tensor(target))

    return torch.stack(images), targets

if __name__ == '__main__':
    train_set = VOC2007Dataset(root=train_data_root)
    train_loader = DataLoader(dataset=train_set, batch_size=1, num_workers=num_workers, drop_last=True, collate_fn=collate_fn, pin_memory=True)
    print(len(train_loader))
    print('Start training!')
    num = len(train_loader)
    mean = 0
    std = 0

    for image, target in train_loader:
        print(target[0].shape)
        break
        # print(image.shape)
        # print(target)