# A pipeline approach to context-aware handwritten text recognition

This repository is created for the paper - A pipeline approach to context-aware handwritten text recognition

We will be uploading the codes to this repository soon...


## Requirements
The code is written in Python and requires Tensorflow and Pytorch. You may install the requirements as follows:
```
pip install -r requirements.txt
```

## Dataset
We require the interested researchers to fill in the agreement upon accessing the collected dataset, namely MMUISD HCR Database, Handwritten Medical Receipt, the agreement is accessible at: [MMUISD HCR Database Release Agreement](https://drive.google.com/file/d/1-MaV_aR9jdcLsUNQ-fB9gI25geVeMJAM/view?usp=sharing).

Ten variants of medical receipts with fifty samples each are collected, where the receipt templates are obtained from online sources. We have collected a total of 500 samples, where the empty medical receipts were distributed to the public to fill in. The participants come from various backgrounds, professions and are aged between twelve and fifty from Malaysia.

## Overview
In this project, we deisnged a pipeline for complete recognition of a handwritten document, from Region of Interest (ROI) localization, to text transcription, and lastly, the context recognition. Figure below shows the architecture of the proposed pipeline.

!["alt_text"](https://github.com/yeefantan/ResNet-101T-for-HCR/blob/main/figures/architecture.png "Architecture")
Figure 1. Proposed pipeline architecture

We have uploaded the experimental setup for data pre-processing, YOLO-v5 for text localization, ResNet-101T for text transcription, and NER for context recognition in the directory of "pipeline". However, necessary changes would be necessary to re-use the code, (e.g. the data location).

A side note: To use YOLO-v5, you must first label the data accordingly. The tool that we have used is LabelImg [2]. A sample annotated image is as below.
!["alt_text"](https://github.com/yeefantan/ResNet-101T-for-HCR/blob/main/figures/sample_annotated_image.png "Sample")

## Results


## References
[1]G. Jocher, A. Stoken, A. Chaurasia, J. Borovec, NanoCode012, TaoXie, Y. Kwon, K. Michael, L. Changyu, J. Fang, A. V, Laughing, tkianai, yxNONG, P. Skalski, A. Hogan, J. Nadar, imyhxy, L. Mammana, AlexWang1900, C. Fati, D. Montes, J. Hajek, L. Diaconu, M.T. Minh, Marc, albinxavi, fatih, oleg, wanghaoyang0106, ultralytics/yolov5: v6.0 - YOLOv5n “Nano” models, Roboflow integration, TensorFlow export, OpenCV DNN support, Zenodo, 2021. https://doi.org/10.5281/zenodo.5563715
[2] Tzutalin. (2015). LabelImg. https://github.com/tzutalin/labelImg
