import numpy as np
import torch
from torchvision import transforms
import torch.utils.data as torchdata
import json
from PIL import Image
import os


def load_destriping_dataset(args,data_stats=None):
    
    data_stats = {"input_channels": 3,
                  "output_channels": 3,
                 }
        
    if "train" in args.mode:
        with open(os.path.join(args.datafolder,'train_val_images_list.json')) as json_file:
            train_val_images_list = json.load(json_file)

        train_images_list = train_val_images_list[0:int(len(train_val_images_list)*args.train_val_ratio)]
        val_images_list = train_val_images_list[int(len(train_val_images_list)*args.train_val_ratio):]

    if "test" in args.mode:
        raise NotImplementedError
        
    dataloaders = {}
    if "train" in args.mode: # == "only_train":
        val_set = DestripingDataset(val_images_list,data_stats,args)
        train_set = DestripingDataset(train_images_list,data_stats,args) 

        dataloaders["train"] = torchdata.DataLoader(train_set, batch_size=args.batch_size,num_workers=args.workers,shuffle=True,pin_memory=True,drop_last=True)
        dataloaders["val"] = torchdata.DataLoader(val_set, batch_size=args.batch_size,num_workers=args.workers,shuffle=False,pin_memory=True,drop_last=True)

    return dataloaders, data_stats


class DestripingDataset(torch.utils.data.Dataset):
    def __init__(self, images_list, data_stats, args):
        
        self.datafolder = args.datafolder
        self.images_list = images_list
        self.args = args

        self.transforms=transforms.Compose([
                               transforms.RandomCrop(256),
                               transforms.ToTensor(),
                               ])

        self.data_stats = data_stats.copy()


    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input_image_full = os.path.join(self.datafolder,self.images_list[idx])

        y = Image.open(input_image_full)
        
        y = self.transforms(y)
        y = y.float()
        
        offset = 0.2 * torch.rand((1,)).item()
        y = offset + (1-offset)*y
        
        x = torch.zeros_like(y)
        x_full = y.clone()
        

        noise_level = 0.1 * torch.rand((1,)).item()
        noise = noise_level * torch.randn_like(x_full)
        x_full = x_full + noise
        x_full = torch.clip(x_full,0.,1.)
        mask = torch.zeros_like(y)
        
        x[0,:,0::3] = x_full[0,:,0::3]
        x[1,:,1::3] = x_full[1,:,1::3]
        x[2,:,2::3] = x_full[2,:,2::3]

        mask[0,:,0::3] = 1
        mask[1,:,1::3] = 1
        mask[2,:,2::3] = 1
        
        interpolation_sets = get_interpolation_sets_extended(x)
        
        out_dict = { "x":x, "y":y, "mask":mask, "name": self.images_list[idx]}
        
        return {**out_dict,**interpolation_sets}
    
    def __len__(self):
        return len(self.images_list)


def get_interpolation_sets_extended(x):
    # gets the 6 color neighbor pixels to be used for interpolating the missing color stripes
    
    x_extended = torch.zeros((3,x.shape[1]+2,x.shape[2]+4))
    x_extended[:,1:-1,2:-2] = x 
    
    u_r = torch.zeros((6,x.shape[1],x.shape[2]))
    u_g = torch.zeros((6,x.shape[1],x.shape[2]))
    u_b = torch.zeros((6,x.shape[1],x.shape[2]))
    u_list = [u_r,u_g,u_b]
    for n in range(3):
        for i in range(0,x.shape[2]):
            u_list[n][0,:,i] = x_extended[n,1:-1,2+((i-n)//3)*3 + n]
            u_list[n][1,:,i] = x_extended[n,1:-1,2+((i+2-n)//3)*3 + n]

            u_list[n][2,:,i] = x_extended[n,0:-2,2+((i-n)//3)*3 + n]
            u_list[n][3,:,i] = x_extended[n,0:-2,2+((i+2-n)//3)*3 + n]

            u_list[n][4,:,i] = x_extended[n,2:,2+((i-n)//3)*3 + n]
            u_list[n][5,:,i] = x_extended[n,2:,2+((i+2-n)//3)*3 + n]

    return { "u_r": u_list[0],
             "u_g": u_list[1],
             "u_b": u_list[2],}