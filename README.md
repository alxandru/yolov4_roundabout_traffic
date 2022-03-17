# YOLOv4 RoundaboutTraffic Tutorial

![Alt Text](https://www.dropbox.com/s/4yj016x3m8bf4vl/intro.gif?raw=1)

This tutorial shows how to train a YOLOv4 vehicle detector using [Darknet](https://github.com/AlexeyAB/darknet) and the [RoundaboutTraffic](data/README.md) dataset on a [NVIDIA Jetson Nano 2GB](https://developer.nvidia.com/embedded/jetson-nano-2gb-developer-kit).

## Table of contents

---

* [Setup](#setup)
* [Installing Darknet](#installing)
* [Downloading training data](#downloading)
* [Configuring YOLO architecture](#configuring)
* [Training the model](#training)
* [Testing the model](#testing)
* [Optimizations](#optimizations)

<a name="setup"></a>

## Setup

---

First you need to setup your Jetson Nano. In order to do that please refer to the [Getting Started Guide](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-2gb-devkit). If you are using the Jetson Nano 4GB version, you can find the Getting Started Guide [here](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit).

Don't forget to check the swap memory. You must have at least 4GB of swap. The following procedure for changing the swap value was taken from [Getting Started with AI on Jetson Nano](https://courses.nvidia.com/courses/course-v1:DLI+S-RX-02+V2/about) :

```bash
# Disable ZRAM:
$ sudo systemctl disable nvzramconfig

# Create 4GB swap file
$ sudo fallocate -l 4G /mnt/4GB.swap
$ sudo chmod 600 /mnt/4GB.swap
$ sudo mkswap /mnt/4GB.swap

# Append the following line to /etc/fstab
$ sudo su
$ echo "/mnt/4GB.swap swap swap defaults 0 0" >> /etc/fstab
$ sudo reboot
```

Now pull the current repository:

```bash
$ mkdir -p ${HOME}/tutorial
$ cd ${HOME}/tutorial
$ git clone https://github.com/alxandru/yolov4_roundabout_traffic.git
```

If you followed the instructions in the Getting Started Guide, you should have a properly installed environment and can proceed to the next steps.

<a name="installing"></a>

## Installing Darknet

---
Next step is to download and build the Darknet framework.

1. Pull the Git repository:

```bash
$ cd ${HOME}/tutorial
$ git clone https://github.com/AlexeyAB/darknet.git
```

2. Modify the `darknet/Makefile`:

We are reconfigure the `Makefile` to run with the following flags activated:

```bash
GPU=1
CUDNN=1
CUDNN_HALF=1
OPENCV=1
```

Add the following architecture for Jetson Nano:

```bash
ARCH= -gencode arch=compute_53,code=sm_53 \
      -gencode arch=compute_53,code=[sm_53,compute_53]
```

You can make the above changes manually or by running the following script that uses `sed`:

```bash
$ bash yolov4_roundabout_traffic/utils/modify_makefile.sh
```

For more information about these settings please refer to [How to compile on Linux (using make)](https://github.com/AlexeyAB/darknet#how-to-compile-on-linux-using-make).

3. (Optional) By default Darknet backups the trained weights of your model every 100 epochs or when it reaches a new best mAP score. Training on a Jetson Nano can be frustrating and slow due to limited resources. I recommend to modify the code in order to change the backup to every 25 epochs or so. This way you can start the training where you left off without waiting to train another 100 epochs if the process is killed by the OS when it consumes all the resources (it may happen more often than you think on a such limited device). I created a patch that does all these modifications:

```bash
$ cd ${HOME}/tutorial
$ cp ${HOME}/tutorial/yolov4_roundabout_traffic/utils/darknet_epochs.patch darknet/
$ cd darknet
$ git apply darknet_epochs.patch
$ git diff # to check the changes
```

4. Compile Darknet:

```bash
$ cd darknet
$ make
```

5. Download yolov4 pre-trained weights to make a sample prediction:

```bash
$ wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
```

6. Now you can check weather the installation is successful: 

```bash
$ ./darknet detector test cfg/coco.data cfg/yolov4.cfg \
            yolov4.weights data/person.jpg
```

<a name="downloading"></a>

## Downloading training data

---
Run the `get_data.sh` script in the `data/` subdirectory. It will download the "RoundaboutTraffic" dataset that's already has the directory structure and the YOLO txt files necessary to train a YOLO network. After the download is completed, it will copy the dataset into the `darknet/data` folder. The images and their annotations are only linked. You could refer to [data/README.md]([data/README.md) for more information about the dataset. You could also check [How to train (to detect your custom objects)](https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects) for an explanation of YOLO txt files.

```bash
$ cd ${HOME}/tutorial/yolov4_roundabout_traffic/data
$ bash get_data.sh
```


<a name="configuring"></a>

## Configuring YOLO architecture

---
YOLO comes with various architectures. Some are large, other are small. For training and testing on a limited embedded device like Jetson Nano, I picked the **yolov4-tiny** architecture, which is the smallest one, and change it for the RoundaboutTraffic dataset.

For a step by step guide on how to configure a YOLO architecture please refer to [How to train (to detect your custom objects)](https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects).

You can just copy the [cfg/yolov4-tiny-bus-car.cfg](cfg/yolov4-tiny-bus-car.cfg) to `darknet/cfg` folder or check the script [utils/configure_arch.sh](utils/configure_arch.sh) and run it. It will make a copy of `darknet/cfg/yolov4-tiny-custom.cfg` file and modify it to suit the training data and the Jetson Nano.

```bash
$ cd ${HOME}/tutorial/yolov4_roundabout_traffic/utils
$ bash configure_arch.sh
# Check weather the cfg file was created
$ less ${HOME}/tutorial/darknet/cfg/yolov4-tiny-bus-car.cfg
```

<a name="training"></a>

## Training the model

---

We download the `yolov4-tiny.conv.29` pre-trained weights:

```bash
$ cd ${HOME}/tutorial/darknet
$ wget https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.conv.29
```

Finally, we train the model:

```bash
$ ./darknet detector train data/obj.data cfg/yolov4-tiny-bus-car.cfg \
            yolov4-tiny.conv.29 -dont_show -mjpeg_port 8090 -map
```

The training will last for a long time and after a couple of hours most likely the process is killed by the OS due to high memory/cpu consumption. If you applied the patch described in [Installing Darknet](#installing) Step 3, the weights are backup every 25 epochs. You can continue the training with the weight of the last backup:

```bash
$ cd backup 
$ cp yolov4-tiny-bus-car_last.weights yolov4-tiny-bus-car_last1.weights
$ cd ..
$ ./darknet detector train data/obj.data cfg/yolov4-tiny-bus-car.cfg \
            backup/yolov4-tiny-bus-car_last1.weights \
            -dont_show -mjpeg_port 8090 -map
```

Normally when the model is being trained, you could monitor its progress on the loss/mAP chart (since the -map option is used). But since the process is restarted various time due to limited resources the chart is fragmented (for every restart there is a new chart). Hence, I don't include the loss/mAP chart here.

I trained the model for 1525 epochs and took about 3 days.

<a name="testing"></a>

## Testing the model

---

After the training is done you can check the mAP of the best model on an [new video](https://drive.google.com/file/d/1GnGOLN_1nlq1-yttD_uk_zJzgfr6vt8Q/view?usp=sharing) like this:

```bash
$ ./darknet detector map data/obj.data cfg/yolov4-tiny-bus-car.cfg \
            backup/yolov4-tiny-bus-car_best.weights \
            ${HOME}/tutorial/test002.mp4
```

I got (mAP@0.50) = 0.953121, or 95.31 % when tested my best custom trained model:

```bash
 calculation mAP (mean average precision)...
 Detection layer: 30 - type = 28
 Detection layer: 37 - type = 28

 detections_count = 5667, unique_truth_count = 2857
class_id = 0, name = bus, ap = 93.85%   	 (TP = 83, FP = 0)
class_id = 1, name = car, ap = 96.77%   	 (TP = 2618, FP = 43)

 for conf_thresh = 0.25, precision = 0.98, recall = 0.95, F1-score = 0.96
 for conf_thresh = 0.25, TP = 2701, FP = 43, FN = 156, average IoU = 78.67 %

 IoU threshold = 50 %, used Area-Under-Curve for each unique Recall
 mean average precision (mAP@0.50) = 0.953121, or 95.31 %
```

And if tested with an IoU threshold = 75%:

```bash
$ ./darknet detector map data/obj.data cfg/yolov4-tiny-bus-car.cfg \
            backup/yolov4-tiny-bus-car_last.weights \
            ${HOME}/tutorial/test002.mp4 -iou_thresh 0.75
```

the mAP decreases to mAP@0.75) = 0.609026, or 60.90 %:

```bash
calculation mAP (mean average precision)...
 Detection layer: 30 - type = 28
 Detection layer: 37 - type = 28

 detections_count = 5667, unique_truth_count = 2857
class_id = 0, name = bus, ap = 48.82%   	 (TP = 55, FP = 28)
class_id = 1, name = car, ap = 72.98%   	 (TP = 2158, FP = 503)

 for conf_thresh = 0.25, precision = 0.81, recall = 0.77, F1-score = 0.79
 for conf_thresh = 0.25, TP = 2213, FP = 531, FN = 644, average IoU = 66.81 %

 IoU threshold = 75 %, used Area-Under-Curve for each unique Recall
 mean average precision (mAP@0.75) = 0.609026, or 60.90 %
```

Here is a video snippet with the darknet detector running only with `yolov4-tiny.conv.29` pre-trained weights:

![Alt Text](https://www.dropbox.com/s/5mpg6q7xiux6n7d/output-pretrained-model.gif?raw=1)

As you can see the occlusions (trees, other cars) in the roundabout and some of the angles the vehicles were caught in different frames clearly affect the detection. The IoU percentage is also not so great.

On the other hand with the best custom trained model the detection and IoU percentage improve significantly with 1525 epochs of training. 

![Alt Text](https://www.dropbox.com/s/pp1tc80712epqax/output-trained-model.gif?raw=1)

In terms of FPS processed by darknet with the trained model and with an input video of 1920x1080@30FPS it gives us around 10 FPS on average:

Finally, let's run a benchmark test on the test video to see the performance in terms of FPS processed by darknet:

```bash
$ ./darknet detector demo data/obj.data cfg/yolov4-tiny-bus-car.cfg \ 
            backup/yolov4-tiny-bus-car_best.weights \ 
            ${HOME}/tutorial/test002.mp4 -benchmark
```

With the trained model for an input video of 1920x1080@30FPS darknet processes around 10 FPS on average:

```bash
FPS:9.9 	 AVG_FPS:10.0
Objects:

FPS:9.9 	 AVG_FPS:10.0
Objects:

FPS:9.9 	 AVG_FPS:10.0
Objects:

FPS:9.9 	 AVG_FPS:10.0
Objects:

FPS:9.9 	 AVG_FPS:10.0
Objects:

FPS:9.9 	 AVG_FPS:10.0
Objects:

...
```

<a name="optimizations"></a>

## Optimizations

---
