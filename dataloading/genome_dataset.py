import numpy as np
import torch
from torchvision import transforms
import torch.nn.functional as F
import torch.utils.data as torchdata
import scipy.stats as st
import json
from PIL import Image
import os
from skimage import filters



def load_genome_dataset(args,data_stats=None):
    
    #if (args.batch_size != 1):
    #    raise NotImplementedError

    data_stats = {"input_channels": 3,
                  "output_channels": 3,
                 }
        
    if "train" in args.mode:
        with open(os.path.join(args.datafolder,'train_val_images_list.json')) as json_file:
            train_val_images_list = json.load(json_file)

        train_images_list = train_val_images_list[0:int(len(train_val_images_list)*args.train_val_ratio)]
        val_images_list = train_val_images_list[int(len(train_val_images_list)*args.train_val_ratio):]

    if "test" in args.mode:
        with open(os.path.join(args.datafolder,'test_images_list.json')) as json_file:
            test_images_list = json.load(json_file)   
        
        #with open(os.path.join(args.datafolder,"fail_images_list.json")) as json_file:
        #    test_images_list = json.load(json_file)   

    dataloaders = {}
    if "train" in args.mode: # == "only_train":
        val_set = MyGenomeDataset(val_images_list,data_stats,args)
        train_set = MyGenomeDataset(train_images_list,data_stats,args) 

        dataloaders["train"] = torchdata.DataLoader(train_set, batch_size=args.batch_size,num_workers=args.workers,shuffle=True,pin_memory=True,drop_last=True)
        dataloaders["val"] = val_loader = torchdata.DataLoader(val_set, batch_size=args.batch_size,num_workers=args.workers,shuffle=False,pin_memory=True,drop_last=True)

    if "test" in args.mode:
        test_set = MyGenomeDataset(test_images_list,data_stats,args)
        dataloaders["test"] = torchdata.DataLoader(test_set, batch_size=args.batch_size,num_workers=args.workers,shuffle=True,drop_last=True)
        
    return dataloaders, data_stats

class MyGenomeDataset(torch.utils.data.Dataset):
    def __init__(self, images_list, data_stats, args):
        
        self.datafolder = args.datafolder
        self.images_list = images_list
        self.args = args

        image_size = 3*256
                
        self.transforms=transforms.Compose([
                               #transforms.Resize(image_size),
                               transforms.RandomCrop(256),
                               transforms.ToTensor(),
                               ])

        self.data_stats = data_stats.copy()


    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input_image_full = os.path.join(self.datafolder,self.images_list[idx])
        #label_image_full = os.path.join(self.datafolder, 'truthIMGs',"debugImage_"+self.images_list[idx][0:-4]+"_truth.png")

        y = Image.open(input_image_full)
        #y = Image.open(label_image_full)
        #x = np.array(x)
        
        
        y = self.transforms(y)
        y = y.float()
        
        #print(y.shape)
        
        
        offset = 0.2 * torch.rand((1,)).item()
        y = offset + (1-offset)*y
        
        x = torch.zeros_like(y)
        x_full = y.clone()
        
        #print(x.shape)
        
        noise_level = 0.1 * torch.rand((1,)).item()
        noise = noise_level * torch.randn_like(x_full)
        #sigma =  0.2 + 2*torch.rand((1,)).item()
        #noise = filters.gaussian(noise.permute(1,2,0).numpy(), sigma=sigma, multichannel=True,preserve_range=False)
        #noise = torch.from_numpy(noise).float().permute(2,0,1)
        
        x_full = x_full + noise
        x_full = torch.clip(x_full,0.,1.)
        mask = torch.zeros_like(y)
        
        x[0,:,0::3] = x_full[0,:,0::3]
        x[1,:,1::3] = x_full[1,:,1::3]
        x[2,:,2::3] = x_full[2,:,2::3]

        
        #x[0,:,4::19] = x_full[0,:,4::19]
        #x[0,:,5::19] = x_full[0,:,5::19]
        #x[0,:,6::19] = x_full[0,:,6::19]

        #x[1,:,8::19] = x_full[1,:,8::19]
        #x[1,:,9::19] = x_full[1,:,9::19]
        #x[1,:,10::19] = x_full[1,:,10::19]
        
        #x[2,:,13::19] = x_full[2,:,13::19]
        #x[2,:,14::19] = x_full[2,:,14::19]
        #x[2,:,15::19] = x_full[2,:,15::19]

        mask[0,:,0::3] = 1
        mask[1,:,1::3] = 1
        mask[2,:,2::3] = 1
        
        interpolation_sets = get_interpolation_sets_extended(x)
        
        out_dict = { "x":x, "y":y, "mask":mask, "name": self.images_list[idx]}
        
        return {**out_dict,**interpolation_sets}
    
    def __len__(self):
        return len(self.images_list)

def get_interpolation_sets(x):
    
    x_extended = torch.zeros((3,x.shape[1],x.shape[2]+4))
    x_extended[:,:,2:-2] = x 

    u_r = torch.zeros((2,x.shape[1],x.shape[2]))
    u_g = torch.zeros((2,x.shape[1],x.shape[2]))
    u_b = torch.zeros((2,x.shape[1],x.shape[2]))
    for i in range(0,x.shape[2]):
        u_r[0,:,i] = x_extended[0,:,2+((i-0)//3)*3 ]
        u_r[1,:,i] = x_extended[0,:,2+((i+2)//3)*3 ]
        u_g[0,:,i] = x_extended[1,:,2+((i-1)//3)*3 + 1]
        u_g[1,:,i] = x_extended[1,:,2+((i+1)//3)*3 + 1]
        u_b[0,:,i] = x_extended[2,:,2+((i-2)//3)*3 + 2]
        u_b[1,:,i] = x_extended[2,:,2+((i+0)//3)*3 + 2]
        
    return { "u_r": u_r,
             "u_g": u_g,
             "u_b": u_b,}

def get_interpolation_sets_extended(x):
    
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
        
        
        #u_g[0,:,i] = x_extended[1,:,2+((i-1)//3)*3 + 1]
        #u_g[1,:,i] = x_extended[1,:,2+((i+1)//3)*3 + 1]
        #u_b[0,:,i] = x_extended[2,:,2+((i-2)//3)*3 + 2]
        #u_b[1,:,i] = x_extended[2,:,2+((i+0)//3)*3 + 2]
        
    return { "u_r": u_list[0],
             "u_g": u_list[1],
             "u_b": u_list[2],}