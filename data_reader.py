import os
import torch
import numpy as np
import cv2 as cv
import xml.etree.ElementTree as ET
from typing import Dict

from torch import Tensor
from torch.utils.data import Dataset

voc2007_classes = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]
voc2007_map = { j : i for i, j in enumerate(voc2007_classes) }

def read_xml(
    xml_filepath : str,
    class_map : Dict[str, int],
    keep_difficult : bool=False
):
    tree = ET.parse(xml_filepath)
    root = tree.getroot()
    node_image_size = root.find('size')
    image_size = []
    target = []
    for child in node_image_size:
        image_size.append(int(child.text))
    for object in root.findall('object'):
        difficult = 0
        rect = [] # xmin, ymin, xmax, ymax
        object_name = ''
        for child in object:
            if child.tag == 'difficult':
                difficult = int(child.text)
            elif child.tag == 'bndbox':
                for i, p in enumerate(child):
                    # rect.append(float(p.text))
                    rect.append(float(p.text) / image_size[0] if i % 2 == 0 else float(p.text) / image_size[1]) # 将bbox映射到[0, 1]
            elif child.tag == 'name':
                object_name = child.text
        if (not keep_difficult) and difficult == 1: # 不保留difficult
            continue
        rect.append(class_map[object_name])
        target.append(rect)

        if rect[0] > rect[2] or rect[1] > rect[3]:
            raise ValueError(f"Wrong bbox, object={object_name}, bbox={rect}")

    return Tensor(target)

class VOC2007Dataset(Dataset):
    def __init__(
        self,
        root: str,
        keep_difficult=False
    ):
        super().__init__()
        print('root =', root)
        self.keep_difficult = keep_difficult
        self.annotations_path = os.path.join(root, 'Annotations')
        self.jpeg_image_path = os.path.join(root, 'JPEGImages')
        self.trainval_path = os.path.join(root, 'ImageSets/Main/trainval.txt')
        self.image_names = []
        self.image_annotations = []
        self.image_paths = []

        with open(self.trainval_path) as f:
            for line in f.readlines():
                line = line.strip()
                line_xml = line + '.xml'
                line_jpg = line + '.jpg'
                self.image_names.append(line)
                self.image_annotations.append(os.path.join(self.annotations_path, line_xml))
                self.image_paths.append(os.path.join(self.jpeg_image_path, line_jpg))

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        annotation_path = self.image_annotations[index]
        # print(image_path, annotation_path)

        image = cv.imread(image_path)
        image_modified = cv.resize(src=image, dsize=(300, 300))
        target = read_xml(xml_filepath=annotation_path, class_map=voc2007_map, keep_difficult=self.keep_difficult)

        return torch.from_numpy(image_modified.astype(np.float32)).permute(2, 0, 1), target

