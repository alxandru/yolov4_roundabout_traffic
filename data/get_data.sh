#!/bin/bash

DARKNET_PATH=${HOME}/tutorial/darknet

pushd $(dirname $0) > /dev/null

python3 -c "import gdown"

if [[ $? = "1" ]]; then
    echo "*** Installing gdown..."
    pip3 install gdown
fi

echo "*** Donwloading RoundaboutDataset..."
python3 -m gdown.cli https://drive.google.com/uc?id=1CswmUyLhKtTSx8NT4RGMML60SWHtJG7M

echo "*** Extracting data..."
tar -xzvf roundabout-YOLO.tar.gz
rm -f roundabout-YOLO.tar.gz

echo "*** Copying data to darknet/data folder..."
mkdir -p $DARKNET_PATH/data/obj
cd roundabout-YOLO/data
ln obj/* $DARKNET_PATH/data/obj
cp obj.data obj.names train.txt val.txt $DARKNET_PATH/data/

popd > /dev/null


