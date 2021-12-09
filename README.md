# DeepDoLCE

This repo contains all the necessary files to train and run the prediction with the deepDolce algorithm.

This [link](https://polybox.ethz.ch/index.php/s/zj75Ez53zelW7qb) provides a pretrained model and some example images. The ```notebook_colorization``` provides an  example of how it is possible to use the pretrained model to process the example dataset. The flow in the notebook is rather straightforward to follow and it is easy to test on other images. In particular, all is needed to know is that ```colorize_image``` accepts as intput a ```torch.FloatTensor```tensor af dimension ```[H,W]```, where H and W are the height and width of the input image. The input is supposed to take values between 0 and 1.

## Requirements

the ```requirements.txt```provides the list of python packages needed (the list might not be exhaustive).
make sure to install the last version of ```segmentation-models-pytorch``` see [this](https://pypi.org/project/segmentation-models-pytorch/#installation).
