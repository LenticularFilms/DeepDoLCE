import os, glob
import imageio
import numpy as np
from PIL import Image

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

dataroot = '/path/to/dolce/dataset/'

input_dir_org = os.path.join(dataroot, 'inputIMGs_uncompressed')
input_files_org = sorted(glob.glob(os.path.join(input_dir_org, '*')))
input_dir = os.path.join(dataroot, 'inputIMGs')
process_inputs(input_files_org, input_dir)

debug_dir = os.path.join(dataroot, 'outputIMGs')
debug_files = sorted(glob.glob(os.path.join(debug_dir, '*')))
target_dir = os.path.join(dataroot, 'truthIMGs')
if not os.path.isdir(target_dir):
    os.mkdir(target_dir)
process_outputs(debug_files, target_dir)