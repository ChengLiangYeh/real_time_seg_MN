# You may need to restart your runtime prior to this, to let your installation take effect
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random
#from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog

#register
from detectron2.data.datasets import register_coco_instances
register_coco_instances("my_dataset_train", {}, "./datasets/0404_new_data/annotations/annotations_train.json", "./datasets/0404_new_data/train/")
register_coco_instances("my_dataset_val", {}, "./datasets/0404_new_data/annotations/annotations_val.json", "./datasets/0404_new_data/val/")
register_coco_instances("my_dataset_test", {}, "./datasets/0404_new_data/annotations/annotations_test.json", "./datasets/0404_new_data/test/")
'''
#visualize training data
my_dataset_train_metadata = MetadataCatalog.get("my_dataset_train")
dataset_dicts = DatasetCatalog.get("my_dataset_train")

import random, os
from detectron2.utils.visualizer import Visualizer

for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    print(d)
    visualizer = Visualizer(img[:, :, ::-1], metadata=my_dataset_train_metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    #cv2_imshow(vis.get_image()[:, :, ::-1])
    cv2.imshow('training data vis',vis.get_image()[:, :, ::-1])
    cv2.waitKey(0)
'''