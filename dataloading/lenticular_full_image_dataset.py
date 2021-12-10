import numpy as np
import torch
from torchvision import transforms
import torch.utils.data as torchdata
import json
from PIL import Image
import os


def load_full_image_dataset(args,data_stats=None):
    
    if (args.batch_size != 1):
        raise NotImplementedError

    data_stats = {"input_channels": 1,
                  "output_channels": 1,
                 }
        
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
        if len(val_images_list) > 0:
            dataloaders["val"]  = torchdata.DataLoader(val_set, batch_size=args.batch_size,num_workers=args.workers,shuffle=False,pin_memory=True,drop_last=True)

    if "test" in args.mode:
        test_set = MyDataset(test_images_list,data_stats,args)
        dataloaders["test"] = torchdata.DataLoader(test_set, batch_size=args.batch_size,num_workers=args.workers,shuffle=True,drop_last=True)
        
    return dataloaders, data_stats

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, images_list, data_stats, args):
        
        self.datafolder = args.datafolder
        self.images_list = images_list
        self.args = args
                
        self.transforms = transforms.Compose([transforms.ToTensor()])

        self.data_stats = data_stats.copy()


    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        input_image_full = os.path.join(self.datafolder, 'inputgrayIMGs',self.images_list[idx])

        label_image_full = None
        if os.path.isdir(os.path.join(self.datafolder, 'truthIMGs')):
            label_image_full = os.path.join(self.datafolder, 'truthIMGs',"debugImage_"+self.images_list[idx][0:-4]+"_truth.png")

        x = Image.open(input_image_full)
        if label_image_full is not None:
            y = Image.open(label_image_full)
        #x = np.array(x)

        if np.max(x) > 255:
            x = np.array(x)
            x = x / (256*256 - 1)
            x = x.astype(float)

        x = self.transforms(x)
        if label_image_full is not None:
            y = self.transforms(y)
        else:
            y = 0. * x

        if self.args.normalize_patch:
            x = x - torch.mean(x)
            x = x / torch.std(x)

        #print(self.images_list[idx])
        #print(idx)

        x = x.float()
        y = y.float()

        return { "x":x, "y":y, "name": self.images_list[idx]}
    
    def __len__(self):
        return len(self.images_list)
