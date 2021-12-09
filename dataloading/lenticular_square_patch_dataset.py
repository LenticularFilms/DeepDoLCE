import numpy as np
import torch
from torchvision import transforms
import torch.utils.data as torchdata
import json
from PIL import Image
import os
import torchvision.transforms.functional as TF

def load_square_patch_dataset(args):
    
    data_stats = {"input_channels": 1,
                  "output_channels": 1,
                  "input_width":args.input_width}
    
    if "train" in args.mode:
        with open(os.path.join(args.datafolder,'train_val_images_list.json')) as json_file:
            train_val_images_list = json.load(json_file)

        train_images_list = train_val_images_list[0:int(len(train_val_images_list)*args.train_val_ratio)]
        val_images_list = train_val_images_list[int(len(train_val_images_list)*args.train_val_ratio):]

    
    if "test" in args.mode:
        with open(os.path.join(args.datafolder,'test_images_list.json')) as json_file:
            test_images_list = json.load(json_file)   

    dataloaders = {}
    if "train" in args.mode:
        
        val_set = MyDataset(val_images_list,data_stats,args)
        train_set = MyDataset(train_images_list,data_stats,args)

        dataloaders["train"] = torchdata.DataLoader(train_set, batch_size=args.batch_size,num_workers=args.workers,shuffle=True,pin_memory=True,drop_last=True)
        dataloaders["val"] = torchdata.DataLoader(val_set, batch_size=args.batch_size,num_workers=args.workers,shuffle=False,pin_memory=True,drop_last=True)

    if "test" in args.mode:
        test_set = MyDataset(test_images_list,data_stats,args)
        dataloaders["test"] = torchdata.DataLoader(test_set, batch_size=args.batch_size,num_workers=args.workers,shuffle=True)
        
    return dataloaders,data_stats

            
class MyDataset(torch.utils.data.Dataset):
    def __init__(self,images_list, data_stats,args):
        
        self.datafolder = args.datafolder
        self.images_list = images_list
        self.args = args
        
        #fixed_width = 3200
        self.crop_1 = int(1.2 * self.args.input_width)
        self.crop_2 = self.args.input_width
        self.max_roatation_deg = 1

        self.data_stats = data_stats.copy()

    def transforms(self, image, mask):

        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=(self.crop_1, self.crop_1))
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)

        rand_angle = self.max_roatation_deg * (2*torch.rand(size=(1,)).item() - 1)   
        image = TF.affine(image,rand_angle,translate=[0,0],scale=1.,shear=0.,interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.affine(mask,rand_angle,translate=[0,0],scale=1.,shear=0.,interpolation=TF.InterpolationMode.BILINEAR)

        image = TF.center_crop(image,(self.crop_2,self.crop_2))
        mask = TF.center_crop(mask,(self.crop_2,self.crop_2))

        # Transform to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        return image, mask

        
    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        input_image_full = os.path.join(self.datafolder, 'inputgrayIMGs',self.images_list[idx])
        label_image_full = os.path.join(self.datafolder, 'truthIMGs',"debugImage_"+self.images_list[idx][0:-4]+"_truth.png")

        x = Image.open(input_image_full)
        y = Image.open(label_image_full)

        x,y = self.transforms(x,y)
            
        if self.args.normalize_patch:
            x = x - torch.mean(x)
            x = x / torch.std(x)

        return { "x":x, "y":y}
    
    
    def __len__(self):
        return len(self.images_list)
