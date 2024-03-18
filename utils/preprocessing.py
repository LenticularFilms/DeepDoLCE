import os, glob
import imageio
import numpy as np
from PIL import Image
import argparse
from pathlib import Path
import shutil


parser = argparse.ArgumentParser(description="pre-processing script")

parser.add_argument('--dataroot',required=True, type=str)
parser.add_argument('--processInputs',default=False)
parser.add_argument('--processOutputs',default=False)

args = parser.parse_args()

# future: assign less probability to pink lines?
def process_inputs(input_files_org, input_dir):
    for file in input_files_org:
        img = imageio.imread(file)
        filename = os.path.basename(file)
        name, _ = os.path.splitext(filename)
        img = (img >> 8).astype('uint8') # compression
        new_name = name + '.png'
        img = (Image.fromarray(img)).convert('L')
        img.save(os.path.join(input_dir, new_name))

def process_outputs(debug_files, target_dir):
    for file in debug_files:
        img = imageio.imread(file)
        filename = os.path.basename(file)
        name, _ = os.path.splitext(filename)
        peaks = np.var(img[...,0], axis=0) + np.var(img[...,1], axis=0) + np.var(img[...,2], axis=0) == 0
        result = np.zeros_like(img[...,0]) # 1 channel
        result[:, peaks] = 255
        new_name = name + '_truth' + '.png'
        result = Image.fromarray(result, mode='L')
        result.save(os.path.join(target_dir, new_name))

def process_rgb(rgb_files, target_dir):
    for file in rgb_files:
        shutil.move(file, target_dir)


if args.processInputs:
    input_dir_org = os.path.join(args.dataroot, 'inputIMGs_uncompressed')
    input_files_org = sorted(glob.glob(os.path.join(input_dir_org, '*')))
    input_dir = os.path.join(args.dataroot, 'inputgrayIMGs')
    Path(input_dir).mkdir(parents=True, exist_ok=True)
    process_inputs(input_files_org, input_dir)

if args.processOutputs:
    debug_dir = os.path.join(args.dataroot, 'outputIMGs')
    Path(debug_dir).mkdir(parents=True, exist_ok=True)
    debug_files = sorted(glob.glob(os.path.join(debug_dir, 'debugImage_*tif')))
    rgb_files = sorted(glob.glob(os.path.join(debug_dir, 'rgb_*tif')))
    target_dir = os.path.join(args.dataroot, 'truthIMGs')
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    dolce_dir = os.path.join(args.dataroot, 'doLCEColorize')
    Path(dolce_dir).mkdir(parents=True, exist_ok=True)
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)
    process_outputs(debug_files, target_dir)
    process_rgb(rgb_files, dolce_dir)