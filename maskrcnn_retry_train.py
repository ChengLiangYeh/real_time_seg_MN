import torch, torchvision
#print(torch.__version__, torch.cuda.is_available())
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

from detectron2.data.datasets import register_coco_instances
register_coco_instances("my_dataset_train", {}, "../datasets/0630_split_dataset/annotations/0630_train_annotations.json", "../datasets/0630_split_dataset/0630_train/")
register_coco_instances("my_dataset_val", {}, "../datasets/0630_split_dataset/annotations/0630_val_annotations.json", "../datasets/0630_split_dataset/0630_val/")
register_coco_instances("my_dataset_test", {}, "../datasets/0630_split_dataset/annotations/0630_test_annotations.json", "../datasets/0630_split_dataset/0630_test/")

#visualize training data
my_dataset_train_metadata = MetadataCatalog.get("my_dataset_train")
dataset_dicts = DatasetCatalog.get("my_dataset_train")

import random, os
from detectron2.utils.visualizer import Visualizer

for d in random.sample(dataset_dicts, 1):
    #print('d=', d)
    #print('filename=', d['file_name'])
    #filename = d['file_name']
    #print(type(filename))
    #a, b = filename.split('\\')
    #filename = a + '/' + b
    #print(filename)

    img = cv2.imread(d["file_name"])
    #print(d)
    #img = cv2.imread(filename)
    #print(img)
    visualizer = Visualizer(img[:, :, ::-1], metadata=my_dataset_train_metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    #cv2.imshow("random train image",vis.get_image()[:, :, ::-1])
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

#train
from detectron2.engine import DefaultTrainer

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))#之前用,R101FPN
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 3
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from model zoo
#print(cfg.MODEL.WEIGHTS)
cfg.SOLVER.IMS_PER_BATCH = 6
cfg.SOLVER.BASE_LR = 0.001 # pick a good LR
cfg.SOLVER.STEPS = (56112,136272)
cfg.SOLVER.WARMUP_FACTOR = 0.001
cfg.SOLVER.WARMUP_ITERS = 2672
cfg.SOLVER.CHECKPOINT_PERIOD = 13360
cfg.SOLVER.MAX_ITER = 149632   # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
#trainer.train() ###eval時槓掉


# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_0080159.pth")  # path to the model we just trained
#print(cfg.MODEL.WEIGHTS)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8   # set a custom testing threshold #origin = 0.7 #在inference時能夠濾掉一些分類分數較低的
predictor = DefaultPredictor(cfg)



'''
#visualize val data
my_dataset_train_metadata = MetadataCatalog.get("my_dataset_val")
dataset_dicts = DatasetCatalog.get("my_dataset_val")

import random, os
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode

path = "./output/inference_image"
os.mkdir(path)
for d in random.sample(dataset_dicts, 1):
#for d in dataset_dicts:
    img = cv2.imread(d["file_name"])
    outputs = predictor(img)
    #visualizer = Visualizer(img[:, :, ::-1], metadata=my_dataset_train_metadata, scale=0.5, instance_mode=ColorMode.IMAGE_BW)
    visualizer = Visualizer(img[:, :, ::-1], metadata=my_dataset_train_metadata, scale=1)
    #print(outputs)
    out = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))
    #cv2.imshow('inference val image',out.get_image()[:, :, ::-1])
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #print(out.get_image()[:, :, ::-1].shape)

    #print("d=", d)
    #print(d["file_name"])
    total_file_name = d["file_name"]
    file_name = total_file_name.split("/")
    #print(file_name[-1])
    cv2.imwrite(path + "/" + "inference_" + file_name[-1], out.get_image()[:, :, ::-1]) #儲存inference result image
'''



from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
evaluator = COCOEvaluator("my_dataset_val", ("segm",), False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "my_dataset_val")
print(inference_on_dataset(predictor.model, val_loader, evaluator))  #找到bug！ -> 不能用trainer.model啦幹！
# another equivalent way to evaluate the model is to use `trainer.test`

'''
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
evaluator = COCOEvaluator("my_dataset_test", ("bbox", "segm"), False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "my_dataset_test")
print(inference_on_dataset(predictor.model, val_loader, evaluator))
'''

