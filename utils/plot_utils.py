import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision
import torch

def create_overlap(x,y, quantile_x=(0.,0.99), cmap='magma',max_alpha=0.5):

    x_plot = x.squeeze()
    y_plot = y.squeeze()

    #x_plot = np.moveaxis(x_plot,0,2)
    x_min = np.quantile(x_plot,quantile_x[0],axis=(0,1),keepdims=True) #,axis=(0,1),keepdims=True)
    x_max = np.quantile(x_plot,quantile_x[1],axis=(0,1),keepdims=True) #,axis=(0,1),keepdims=True)
    x_plot = (x_plot - x_min) / (x_max - x_min)
    x_plot = np.clip(x_plot,0,1)
    
    #y_min = 0#np.quantile(y_plot,quantile_y[0],axis=(0,1),keepdims=True) #,axis=(0,1),keepdims=True)
    #y_max = 1#np.quantile(y_plot,quantile_y[1],axis=(0,1),keepdims=True) #,axis=(0,1),keepdims=True)
    #y_plot = (y_plot - y_min) / (y_max - y_min)
    y_plot = np.clip(y_plot,0,1)

    img_1 = Image.fromarray(np.uint8((x_plot) * 255)).convert("RGBA")
    cm = plt.get_cmap(cmap)
    img_2 = Image.fromarray(cm(y_plot, bytes=True))
    img_a = Image.fromarray((1-y_plot*max_alpha)*255).convert("L")

    img = Image.composite(img_1,img_2,img_a)
        
    return img

def create_RGBimage(x, quantile_x=(0.,0.99)):

    x_plot = x.squeeze()

    x_plot = np.moveaxis(x_plot,0,2)
    x_min = np.quantile(x_plot,quantile_x[0],axis=(0,1),keepdims=True) #,axis=(0,1),keepdims=True)
    x_max = np.quantile(x_plot,quantile_x[1],axis=(0,1),keepdims=True) #,axis=(0,1),keepdims=True)
    x_plot = (x_plot - x_min) / (x_max - x_min)
    x_plot = np.clip(x_plot,0,1)

    img = Image.fromarray(np.uint8((x_plot) * 255)).convert("RGB")
    #converter = ImageEnhance.BRightness(img_1)
    #img_1 = converter.enhance(1.5)
        
    return img


def add_tensorboard_images(writer, sample, output, global_step,args=None):

    max_images = min(8, sample['x'].shape[0])
    
    img_y = []
    img_y_pred = []
    for i in range(0,max_images):

        if "SVDD" in args.model:
            img_y.append(torch.from_numpy(np.asarray(create_RGBimage(sample['x'][i].cpu().detach().clone().numpy(),
                                    quantile_x=(0.,1.)))).permute(2,0,1))
            img_y_pred.append(torch.from_numpy(np.asarray(create_RGBimage(output['x_pred'][i].cpu().detach().clone().numpy(),
                                    quantile_x=(0.,1.)))).permute(2,0,1))
        else:
            img_y.append(torch.from_numpy(np.asarray(create_overlap(sample['x'][i].cpu().detach().clone().numpy(),
                                    sample['y'][i].cpu().detach().clone().numpy(),
                                    quantile_x=(0.,1.),cmap='viridis',max_alpha=0.5))).permute(2,0,1))
            img_y_pred.append(torch.from_numpy(np.asarray(create_overlap(sample['x'][i].cpu().detach().clone().numpy(),
                                    output['y_pred_sigmoid'][i].cpu().detach().clone().numpy(),
                                    quantile_x=(0.,1.),cmap='viridis',max_alpha=0.5))).permute(2,0,1))

    img_y = torchvision.utils.make_grid(img_y,nrow=max_images) #,normalize=True)
    img_y_pred = torchvision.utils.make_grid(img_y_pred,nrow=max_images)
    
    writer.add_image("gt",img_y,global_step=global_step)
    writer.add_image("predicted",img_y_pred,global_step=global_step)
