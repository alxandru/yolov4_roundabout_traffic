#/bin/bash

CFG_PATH=${HOME}/tutorial/darknet/cfg

cp $CFG_PATH/yolov4-tiny-custom.cfg $CFG_PATH/yolov4-tiny-bus-car.cfg

sed -i'' -e 's/subdivisions=1/subdivisions=32/g' $CFG_PATH/yolov4-tiny-bus-car.cfg
sed -i'' -e 's/width=416/width=608/g' $CFG_PATH/yolov4-tiny-bus-car.cfg
sed -i'' -e 's/height=416/height=608/g' $CFG_PATH/yolov4-tiny-bus-car.cfg
sed -i'' -e 's/max_batches = 500200/max_batches=6000/g' $CFG_PATH/yolov4-tiny-bus-car.cfg
sed -i'' -e 's/steps=400000,450000/steps=4800,5400/g' $CFG_PATH/yolov4-tiny-bus-car.cfg
sed -i'' -e 's/classes=80/classes=2/g' $CFG_PATH/yolov4-tiny-bus-car.cfg
sed -i'' -e 's/random=0/random=1/g' $CFG_PATH/yolov4-tiny-bus-car.cfg
sed -i'' -e 's/filters=255/filters=21/g' $CFG_PATH/yolov4-tiny-bus-car.cfg