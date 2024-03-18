#!/bin/bash

DATASET_DIR='./dataset/KunterBunt'
BASE_DIR='KunterbuntLenticular'
START=00001
END=00010
DOLCE_OUTPUT_DIR=$DATASET_DIR/inputIMGs_uncompressed/outputIMGs
mkdir -p $DOLCE_OUTPUT_DIR

#use dolCE to generate debugImages and color Images
# Ask Giorgio how to get correct results here for all cases
./doLCESignalProcessing/doLCE -mode 2 1 3 -profileRelThickness 0.2 -profileRelPosY 0.2 -relaxRaster 1 -rasterSpacing 15.0 -troubleshoot $DATASET_DIR/inputIMGs_uncompressed $BASE_DIR $START $END outputIMGs

#move the generated debugImages to dataset directory 

mv $DOLCE_OUTPUT_DIR $DATASET_DIR

#process input tifs to generate .pngs

python ./utils/preprocessing.py --dataroot $DATASET_DIR --processInputs True

#process debugImages to get truth images
python ./utils/preprocessing.py --dataroot $DATASET_DIR --processOutputs True

mkdir -p $DATASET_DIR/deepdoLCEColorize

#colorize using deep-dolCE
python dolce_test.py --model UResNet --model_2 UResNetColorize --save-dir ./runs/ --resume ./checkpoints/model_lenticule_detection.pth.tar --resume_2 ./checkpoints/model_colorize.pth.tar \
 --mode test_input_images --workers 0 --data lenticular_full_image --datafolder $DATASET_DIR --style new_dolce