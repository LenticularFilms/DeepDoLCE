import numpy as np
import torch
from torchvision import transforms
import torch.nn.functional as F
import torch.utils.data as torchdata
import scipy.stats as st
import json
from PIL import Image
import os


def load_dolce_plus_genome_dataset(args,data_stats=None):
    
    if (args.batch_size != 1):
        raise NotImplementedError

    data_stats = {"input_channels": 3,
                  "output_channels": 3,
                 }
        
    dataloaders = {}
    if "train" in args.mode:
        with open(os.path.join(args.datafolder,'train_val_images_list.json')) as json_file:
            train_val_images_list = json.load(json_file)

        train_images_list = train_val_images_list[0:int(len(train_val_images_list)*args.train_val_ratio)]
        val_images_list = train_val_images_list[int(len(train_val_images_list)*args.train_val_ratio):]

        with open(os.path.join(args.datafolder_bis,'train_val_images_list.json')) as json_file:
            train_val_images_list_bis = json.load(json_file)

        train_images_list_bis = train_val_images_list_bis[0:int(len(train_val_images_list_bis)*args.train_val_ratio)]
        val_images_list_bis = train_val_images_list_bis[int(len(train_val_images_list_bis)*args.train_val_ratio):]
        
        train_set = MyDolceGenomeDataset(train_images_list,train_images_list_bis,data_stats,args) 
        val_set = MyDolceGenomeDataset(val_images_list,val_images_list_bis,data_stats,args)

        dataloaders["train"] = torchdata.DataLoader(train_set, batch_size=args.batch_size,num_workers=args.workers,shuffle=True,pin_memory=True,drop_last=True)
        dataloaders["val"]  = torchdata.DataLoader(val_set, batch_size=args.batch_size,num_workers=args.workers,shuffle=False,pin_memory=True,drop_last=True)


    if "test" in args.mode:
        raise NotImplementedError
        
        test_set = MyGenomeDataset(test_images_list,data_stats,args)
        dataloaders["test"] = torchdata.DataLoader(test_set, batch_size=args.batch_size,num_workers=args.workers,shuffle=True,drop_last=True)
        

        #with open(os.path.join(args.datafolder,'test_images_list.json')) as json_file:
        #    test_images_list = json.load(json_file)   
        
        #with open(os.path.join(args.datafolder,"fail_images_list.json")) as json_file:
        #    test_images_list = json.load(json_file)   
        
    return dataloaders, data_stats

class MyDolceGenomeDataset(torch.utils.data.Dataset):
    def __init__(self, images_list_source,images_list_target, data_stats, args):
        
        self.datafolder = args.datafolder
        self.datafolder_bis = args.datafolder_bis
        self.images_list_source = images_list_source
        self.images_list_target = images_list_target
        self.args = args

        self.image_size = 192
                
        self.transforms_source=transforms.Compose([
                               transforms.Resize(self.image_size),
                               transforms.ToTensor(),
                               ])

        self.transforms_target=transforms.Compose([
                                transforms.Resize(self.image_size),
                                transforms.ToTensor(),
                               ])

        self.data_stats = data_stats.copy()


    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if len(self.images_list_source) < len(self.images_list_target):
            idx_source = idx
            idx_target = torch.randint(0,len(self.images_list_target),size=(1,)).item()
        else:
            idx_source = torch.randint(0,len(self.images_list_source),size=(1,)).item()
            idx_target =idx
        #self.idx += 1
        #while "Nazi" not in self.images_list[self.idx]:
        #    self.idx = self.idx + 1
        #idx = self.idx

        input_image_full_source = os.path.join(self.datafolder,self.images_list_source[idx_source])
        input_image_full_target = os.path.join(self.datafolder_bis,self.images_list_target[idx_target])
        #label_image_full = os.path.join(self.datafolder, 'truthIMGs',"debugImage_"+self.images_list[idx][0:-4]+"_truth.png")

        x_source = Image.open(input_image_full_source)
        x_target = Image.open(input_image_full_target)


        #y = Image.open(label_image_full)
        #x = np.array(x)

        x_source = self.transforms_source(x_source)
        x_target = self.transforms_target(x_target)

        x_target = x_target.float()
        x_source = x_source.float()


        return { "x_source":x_source, "x_target":x_target }
    
    def __len__(self):
        return min(len(self.images_list_source),len(self.images_list_target))
