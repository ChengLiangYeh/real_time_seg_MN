# Real Time Segmentation Of Median Nerve
- This repo is for "Real-time segmentation of median nerve in dynamic sonography using state-of-the-art deep learning models". (Dec. 2021) 
- We implement some state-of-the-art deep learning instance segmentation frameworks to segment median nerve in dynamic sonography. Our implementation relies on an open source toolbox called adelaiDet. AdelaiDet is bulit on top of Detectron2, which is Facebook AI Research's next generation library that provides state-of-the-art detection and segmentation algorithms.
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
- For detectron2, gcc & g++ ≥ 5.4 are required. ninja is recommended for faster build. If you have them, run
```
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2
```
