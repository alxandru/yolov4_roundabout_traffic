# Roundabout Traffic Dataset

Roundabout Traffic dataset is based on a recorded video with a fixed camera in a roundabout. The video has 2:29 minutes and was annotated and labeled using [VOTT](https://github.com/microsoft/VoTT) tool. Even though the video was shot with 30 FPS, only 15 FPS were labeled due to time constrains.

This is a small dataset. It consists of 1483 images (1920 × 1080 RGB) with their annotations from which 1096 images are used for training and 386 images are used for validation. There are around 9 vehicles per image with various kinds of occlusions (trees, buildings, other vehicles).

In the final YOLO txt files, there are 2 classes of objects. Class 0 is "bus", and class 1 is "car".

The YOLO dataset can be downloaded from [here](https://drive.google.com/file/d/1CswmUyLhKtTSx8NT4RGMML60SWHtJG7M/view?usp=sharing).

The original video can be found [here](https://drive.google.com/file/d/1fB_WHSA1YQJFTdtwfhpDPp4PbMZofKJm/view?usp=sharing).

The VOTT annotation and labeling tool does not have the option to export the datasets in YOLO format. Thus, it was exported in PascalVOC format and then it was converted to YOLO format using [pascalVOC2YOLO](https://github.com/alxandru/pascalVOC2YOLO) script.
