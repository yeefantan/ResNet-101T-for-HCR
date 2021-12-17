# A pipeline approach to context-aware handwritten text recognition

This repository is created for the paper - A pipeline approach to context-aware handwritten text recognition

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
**Figure 1.** Proposed pipeline architecture

We have uploaded the notebooks of the experimental studies setup for data pre-processing (noise processing, line removal, data augmentation, etc), YOLO-v5 for text localization, ResNet-101T for text transcription, and NER for context recognition in the directory of "pipeline". However, some changes would be necessary to re-use the code, (e.g. the data location). The folder "utils" contains the necessary functions, such as loading the data.

A side note: To use YOLO-v5, you must first label the data accordingly. The tool that we have used is LabelImg [2]. A sample annotated image is as below.

!["alt_text"](https://github.com/yeefantan/ResNet-101T-for-HCR/blob/main/figures/sample_annotated_image.png "Sample")

**Figure 2.** Sample annotated image

To fully re-use the code, the researcher must start with data pre-processing, followed by data annotation for YOLO-v5, and data annotation for the segmented handwritten images, different model experiments (LSTM/ViT/ResNet-101T), lastly, data collection and annotation for text data using NER annotation software, and the NER model training. We have also uploaded the full pipeline demonstration in both PDF and notebook version under the directory "pipeline".

## Results

This section presents the experimental result of the paper in terms of Character Error Rate (CER), Word Error Rate (WER), and computational time. Refer to the table below:

|Model    | CER | WER |Computaitonal Time (seconds)|
|---------|-----|-----|----------------------------|
|ResNet-101T|7.77|10.77|350864|
|LSTM|11.55|26.64|87148|
|ViT|12.47|20.18|571428|

Some of the transcribed outputs are as followed:


|!["alt_text"](https://github.com/yeefantan/ResNet-101T-for-HCR/blob/main/figures/demo_1_input.jpg "Demo 1 Input") | !["alt_text"](https://github.com/yeefantan/ResNet-101T-for-HCR/blob/main/figures/demo_1_output.jpg "Demo 1 Output")|
|!["alt_text"](https://github.com/yeefantan/ResNet-101T-for-HCR/blob/main/figures/demo_2_input.jpg "Demo 2 Input") | !["alt_text"](https://github.com/yeefantan/ResNet-101T-for-HCR/blob/main/figures/demo_2_output.jpg "Demo 2 Output")|

## References
[1]G. Jocher, A. Stoken, A. Chaurasia, J. Borovec, NanoCode012, TaoXie, Y. Kwon, K. Michael, L. Changyu, J. Fang, A. V, Laughing, tkianai, yxNONG, P. Skalski, A. Hogan, J. Nadar, imyhxy, L. Mammana, AlexWang1900, C. Fati, D. Montes, J. Hajek, L. Diaconu, M.T. Minh, Marc, albinxavi, fatih, oleg, wanghaoyang0106, ultralytics/yolov5: v6.0 - YOLOv5n “Nano” models, Roboflow integration, TensorFlow export, OpenCV DNN support, Zenodo, 2021. https://doi.org/10.5281/zenodo.5563715

[2] Tzutalin. (2015). LabelImg. https://github.com/tzutalin/labelImg
