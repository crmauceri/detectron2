import os
import numpy as np
import json
from detectron2.structures import BoxMode
from detectron2.config import get_cfg
from detectron2.data import build_detection_train_loader
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import matplotlib.pyplot as plt
from PIL import Image

from detectron2.data.datasets import register_coco_instances

cfg = get_cfg()
cfg.merge_from_file('configs/COCO-InstanceSegmentation/RGBD_mask_rcnn_R_50_FPN_3x_sunrgbd.yaml')
cfg.DATALOADER.NUM_WORKERS = 1
#cfg.DATASETS.TRAIN = ("sunrgbd_train",)
cfg.INPUT.USE_DEPTH = True

for d in ["train", "val"]:
    register_coco_instances("sunrgbd_{}".format(d), {}, "/Users/Mauceri/Workspace/SUNRGBD/annotations/instances_{}.json".format(d), "/Users/Mauceri/Workspace/", "/Users/Mauceri/Workspace/")

val_data = build_detection_train_loader(cfg)

sample = val_data.dataset.dataset.dataset._dataset[0]
im = Image.open(sample['file_name'])
visualizer = Visualizer(im, MetadataCatalog.get('sunrgbd_val'), scale=0.5)
vis = visualizer.draw_dataset_dict(sample)
plt.figure()
plt.imshow(vis.get_image())
plt.show()

for i, sample in enumerate(val_data):
    im = sample[0]['image'].permute(1,2,0)
    print(im.shape)
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(im[:,:,0:3]/256)
    plt.subplot(1,2,2)
    plt.imshow(im[:,:,3]/256)
    plt.show()