# Real Time Segmentation Of Median Nerve
![1](real_time_seg_MN/seg_result.png)
- This repo is for "Real-time segmentation of median nerve in dynamic sonography using state-of-the-art deep learning models". (Dec. 2021) 
- We implement some state-of-the-art deep learning instance segmentation frameworks to segment median nerve in dynamic sonography. Our implementation relies on an open source toolbox called [adelaiDet](https://github.com/aim-uofa/AdelaiDet). AdelaiDet is bulit on top of [Detectron2](https://github.com/facebookresearch/detectron2/tree/d4412c7070b28e50037b3797de8a579afd008b2b), which is Facebook AI Research's next generation library that provides state-of-the-art detection and segmentation algorithms.
- Current model weights and codes which are only for inference.
- To date, this repo offers inference codes and following model weights:
  - SOLOv2
  - BlendMask
  - Mask R-CNN
  - Yolact
# Requirements
- Linux or macOS with Python ≥ 3.6
- PyTorch ≥ 1.6 and torchvision that matches the PyTorch version. Please check pytorch.org for more details.
- OpenCV
- pillow
- matplotlib
- pandas
- For detectron2, gcc & g++ ≥ 5.4 are required. ninja is recommended for faster build. If you have them, run
```
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2
```
Note that: Check [detectron2 installation](https://github.com/facebookresearch/detectron2/blob/d4412c7070b28e50037b3797de8a579afd008b2b/INSTALL.md)
- For adelaiDet, build it with:
```
git clone https://github.com/aim-uofa/AdelaiDet.git
cd AdelaiDet
python setup.py build develop
```
Note that: Check [AdelaiDet](https://github.com/aim-uofa/AdelaiDet)

# Model:
- SOLOv2 weight: please check the google drive link for downloading model weight. [R50](https://drive.google.com/file/d/1mX8u2wBSoMSJCZvEChtTVoQvL9Wioi1T/view?usp=share_link), [R101](https://drive.google.com/file/d/1uqVj_jgPrtwRbr46ecl8ThV9AwaChU6w/view?usp=share_link)
- inference: Note that, Setting Detectron2 and AdelaiDet is the first step. Second, git clone this repo for all folders. Finally, copy all files in those folders, and replace the corresponding files in the AdelaiDet folder. 
- command:
```
OMP_NUM_THREADS=3 python tools/train_net.py     --config-file configs/SOLOv2/R50_3x.yaml     --eval-only     --num-gpus 3     OUTPUT_DIR training_dir/SOLOv2_R50     MODEL.WEIGHTS training_dir/SOLOv2_R50/solov2_r50fpn_weight.pth
```
```
OMP_NUM_THREADS=3 python tools/train_net.py     --config-file configs/SOLOv2/R101_3x.yaml     --eval-only     --num-gpus 3     OUTPUT_DIR training_dir/SOLOv2_R101     MODEL.WEIGHTS training_dir/SOLOv2_R101/solov2_r101fpn_weight.pth
```
----------------------------------------
- BlendMask weight: please check the google drive link for downloading model weight. [R50](https://drive.google.com/file/d/12QMHhyuvWfei1K6qDwB9_Cuey6AQKjtB/view?usp=share_link), [R101](https://drive.google.com/file/d/1cDVs-BGCcV1FyzW5rI1G-VAVb7m2cuVy/view?usp=sharing)
- inference: Note that, Check and follow SOLO inference setting. 
- inference command:
```
OMP_NUM_THREADS=3 python tools/train_net.py     --config-file configs/BlendMask/R_50_3x.yaml     --eval-only     --num-gpus 3     OUTPUT_DIR training_dir/bm     MODEL.WEIGHTS training_dir/bm/blendmask_r50fpn_weight.pth
```
```
OMP_NUM_THREADS=3 python tools/train_net.py     --config-file configs/BlendMask/R_101_3x.yaml     --eval-only     --num-gpus 3     OUTPUT_DIR training_dir/bm     MODEL.WEIGHTS training_dir/bm/blendmask_r101fpn_weight.pth
```
----------------------------------------
- Mask R-CNN weight: please check the google drive link for downloading model weight.[R50](https://drive.google.com/file/d/1QvKdyroyDQE7rWSlgGtUxWMtMjUWgonQ/view?usp=share_link), [R101](https://drive.google.com/file/d/1RH3y7w-bj3aiRpM_o95DfwapcEnPwabU/view?usp=share_link)
- inference: Note that, Check and follow detectron2 setting. If you need some basic instruction, please see [TORCHVISION OBJECT DETECTION FINETUNING TUTORIAL](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)
- inference command:
```
python maskrcnn_retry_inference.py #you need to revise the weight path for yourself.
```
----------------------------------------
- Yolact weight: please check the google drive link for downloading model weight.[R50](https://drive.google.com/file/d/1BBpWsUPnHUslAzarXchoTv0C4_me_mKi/view?usp=share_link), [R101](https://drive.google.com/file/d/1yoJREwB-el30tQsOfWbIS39DnN6H0Nb0/view?usp=share_link)
- inference: Note that, Check and follow yolact installation [here](https://github.com/dbolya/yolact). Highly recommand to bulid it in the Anaconda independent environment.
- inference command:
```
#To get mask mAP:
python eval.py --trained_model=weights/yolact_r50fpn_weight.pth
python eval.py --trained_model=weights/yolact_r101fpn_weight.pth

#To run COCOEval on the files:
python run_coco_eval.py
```
- Note that, for more command, check the [yolact](https://github.com/dbolya/yolact)
----------------------------------------

# Dataset: 
- Please contact us for more details.
