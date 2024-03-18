import numpy as np
import matplotlib.pyplot as plt

from utils.plot_utils import create_overlap
from colorize_suite import *

args = parser.parse_args()

colorizeSuite = ColorizeSuite(args)

iterator_test_dataset = iter(colorizeSuite.dataloaders["test"].dataset)

for sample in iterator_test_dataset:
    x = sample["x"].squeeze().numpy()
    y,z_refined,z_orig,y_mosaic,bottom_list,top_list,y_orig = colorizeSuite.colorize_image(sample["x"])
    plt.imsave(args.datafolder + '/deepdoLCEColorize/' + sample["name"], y.squeeze().permute(1,2,0).numpy().astype(np.float32))
