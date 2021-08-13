import sys, argparse, os
import numpy as np

import torch
import torch.nn.functional as F
from skimage import filters
from skimage.morphology import disk

from others import *

from models import get_model
from utils.other import new_log
from dataloading import get_dataloaders
from dataloading.genome_dataset import get_interpolation_sets_extended
from utils.plot_utils import *

if 'ipykernel' in sys.modules:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

from models.color_restoration import ColorRestoration

parser = argparse.ArgumentParser(description="training script")

#### general parameters #####################################################
parser.add_argument('--tag', default="__",type=str)
parser.add_argument("--device",default="cuda",type=str,choices=["cuda", "cpu"])
parser.add_argument("--save-dir",help="Path to directory where models and logs should be saved saved")
parser.add_argument('--mode',default="only_train",type=str,choices=["None","train","test","train_and_test"],help="mode to be run")

#### data parameters ##########################################################
parser.add_argument("--data",default="full_image",type=str,help="dataset selection")
parser.add_argument("--datafolder",default="/scratch/code/data_dolce",type=str,help="root directory of the dataset")
parser.add_argument("--workers", type=int, default=4,metavar="N",help="dataloader threads")
parser.add_argument("--batch-size", type=int, default=1)
parser.add_argument("--normalize-patch",action='store_true',default=False,help="normalize patches")
parser.add_argument("--train-val-ratio",type=float,default=1.,help="ratio of the dataset to use for training")

#### model parameters #####################################################
parser.add_argument("--model",type=str,help="model to run for lenticules detection")
parser.add_argument("--model_2",type=str,default=None,help="model to run for colorization")
parser.add_argument('--resume', type=str, default=None,help='path to resume the model if needed')
parser.add_argument('--resume_2', type=str, default=None,help='path to resume the model if needed')
parser.add_argument("--style",type=str,default="new",choices=["new","old_dolce","interp"],help="type of colorization")
parser.add_argument("--order",type=str,default="nearest",choices=["nearest","linear","cubic"],help="type of interpolation")

class ColorizeSuite(object):

    def __init__(self, args):

        self.args = args
        
        self.dataloaders, self.data_stats =  get_dataloaders(args)

        self.device = torch.device("cuda" if torch.cuda.is_available() and args.device=="cuda" else "cpu")

        self.model = get_model(args,self.data_stats)
        if args.resume is not None:
            self.resume(self.model,path=args.resume)
            self.model.to(self.device)

        if args.model_2 is not None:
            self.model_color = get_model(args,self.data_stats,secondary_model=True)
            if args.resume is not None:
                self.resume(self.model_color,path=args.resume_2)
            self.model_color.to(self.device)
    
        self.experiment_folder = new_log(args.save_dir,args.model + "_" + args.tag,args=args)


    def resume(self,model,path):
        if not os.path.isfile(path):
            raise RuntimeError("=> no checkpoint found at '{}'".format(path))
        
        model.load_state_dict(torch.load(path))
        
        print("model loaded.")

        return

    
    def to_device(self, sample, device=None):
        if device is None:
            device = self.device
        sampleout = {}
        for key, val in sample.items():
            if isinstance(val, torch.Tensor):
                sampleout[key] = val.to(device)
            elif isinstance(val, list):
                new_val = []
                for e in val:
                    if (isinstance(e, torch.Tensor)):
                        new_val.append(e.to(device))
                    else:
                        new_val.append(val)
                sampleout[key] = new_val
            else:
                sampleout[key] = val
        return sampleout


    def colorize_dataset(self,dataset_name="test"):

        if not os.path.exists(os.path.join(self.experiment_folder,"lenticules/")):
            os.mkdir(os.path.join(self.experiment_folder,"lenticules/"))
        if not os.path.exists(os.path.join(self.experiment_folder,"color/")):
            os.mkdir(os.path.join(self.experiment_folder,"color/"))
        if not os.path.exists(os.path.join(self.experiment_folder,"rgb_filter/")):
            os.mkdir(os.path.join(self.experiment_folder,"rgb_filter/"))

        for sample in tqdm(self.dataloaders[dataset_name].dataset,leave=False):
            
            x = sample["x"].unsqueeze(0)

            y,z,_,_,_,_,_ = self.colorize_image(x)

            #rgb_filter_z = rgb_filter.cpu() + z.cpu()
            #img = Image.fromarray((255*rgb_filter_z.squeeze().permute(1,2,0)).numpy().astype('uint8').squeeze(), 'RGB')
            #img.save(os.path.join(self.experiment_folder,"rgb_filter/","rgb_filter"+sample["name"]))

            img = Image.fromarray((255*z).cpu().numpy().astype('uint8').squeeze(), 'L')
            img.save(os.path.join(self.experiment_folder,"lenticules/","lenticules_"+sample["name"]))

            img = Image.fromarray((255*y.squeeze().permute(1,2,0)).cpu().numpy().astype('uint8').squeeze(), 'RGB')
            img.save(os.path.join(self.experiment_folder,"color/","color"+sample["name"]))

    def colorize_image(self,x):

        while len(x.shape) < 4:
            x = x.unsqueeze(0)

        return_device = x.device

        x = x.to(self.device)

        z = self.process_image_lenticules(x)

        #else:
        #    x_np = x.cpu().numpy().squeeze()
        #    #z = filters.sato(x_np,black_ridges=True) 
        #    z = filters.meijering(x_np,black_ridges=True,sigmas=np.linspace(0.1,1,10)) #sigmas=[0.1,0.5,1,2,4]) #
        #    z = torch.from_numpy(z).unsqueeze(0).unsqueeze(0).float().to(self.device)
        
        #style = "new"
        if self.args.style == "old_dolce": #old style
            _, Pxx_den = sg.periodogram(np.sum(z.cpu().numpy().squeeze(),axis=0))
            w = z.squeeze().shape[1] / ( 50 + np.argmax(Pxx_den[50:350]))

            delta_max = 19
            min_lenticule_width = int(np.floor(w)-1)
            max_lenticule_width = int(np.ceil(w)+1)
            u = build_score_matrix(1-z.cpu().numpy().squeeze(),delta_max)

            bottom,top  = optimize_locations(u,delta_max,min_lenticule_width,max_lenticule_width,w,10.)#alpha=0.1)

            z_raster = reconstruct_boundaries(bottom,top,size=x.squeeze().shape,lenticule_min = min_lenticule_width,lenticule_max = max_lenticule_width)

            z_raster = torch.from_numpy(z_raster).unsqueeze(0).unsqueeze(0).float().to(self.device)
            #z_raster = z
            x_mosaic,_,_,_ = self.extract_mosaic(z,x)

            colorize = ColorRestoration(max_lenticule_width).to(self.device)
            out_dict = colorize(x,z_raster)
            
            y = out_dict["y"].squeeze()#.detach().cpu()
        elif self.args.style == "new":
            x_mosaic,z_raster,bottom,top = self.extract_mosaic(z,x)
            y = self.demosaicing(x_mosaic)
        elif self.args.style == "interp":
            x_mosaic,z_raster,bottom,top = self.extract_mosaic(z,x)
            y = torch.from_numpy(nearest_interp(x_mosaic.cpu().numpy().squeeze(),order=self.args.order)).float()

        gain_matrix = torch.tensor([[0.789,0.154,0.057],
                                    [-0.286,1.195,0.060],
                                    [-0.049,0.035,1.035]])
        gain_matrix = gain_matrix.unsqueeze(2).unsqueeze(2).float().to(y.device)
        y_orig = y.clone()
        y = torch.clamp(torch.sum(y.unsqueeze(0)*gain_matrix,1),0,1)
        
        return (y.detach().to(return_device),
                z_raster.detach().to(return_device),
                z.detach().to(return_device),
                x_mosaic.detach().to(return_device),
                bottom,top,y_orig.detach().to(return_device),
               )

    def process_image_lenticules(self,x):

        delta = 8
        network_input_size = 256

        delta_pix = delta * network_input_size

        x_height_reminder = delta_pix - (x.shape[2] % delta_pix)
        x_width_reminder = delta_pix - (x.shape[3] % delta_pix)

        x_padded = F.pad(x,(0,x_width_reminder,0,x_height_reminder))
        y_padded = torch.zeros_like(x_padded)

        iter_height = x_padded.shape[2] // delta_pix
        iter_width = x_padded.shape[3] // delta_pix

        self.model.eval()
        with torch.no_grad():
            for i in range(0,iter_height):
                for j in range(0,iter_width):

                    x_ext = x_padded[:,:,delta_pix*i:min(x_padded.shape[2],delta_pix*(i+1)),delta_pix*j:min(x_padded.shape[3],delta_pix*(j+1))]

                    out = self.model(x_ext.to(self.device))
                    if "y_pred_sigmoid" in out:
                        y_ext = out["y_pred_sigmoid"].to(x.device)
                    else:
                        y_ext = out["y_pred"].to(x.device)

                    y_padded[:,:,delta_pix*i:min(x_padded.shape[2],delta_pix*(i+1)),delta_pix*j:min(x_padded.shape[3],delta_pix*(j+1))] = y_ext

        y = y_padded[:,:,0:x.shape[2],0:x.shape[3]]

        return y.to(x.device).detach()

    def extract_mosaic(self,z,x):

        _, Pxx_den = sg.periodogram(np.sum(z.cpu().numpy().squeeze(),axis=0))
        w = z.squeeze().shape[1] / ( 50 + np.argmax(Pxx_den[50:350]))

        delta_max = 16
        min_lenticule_width = int(np.floor(w)-1)
        max_lenticule_width = int(np.ceil(w)+1)
        u = build_score_matrix(1-z.cpu().numpy().squeeze(),delta_max)

        lenticules_location_bottom,lenticules_location_top  = optimize_locations(u,delta_max,min_lenticule_width,max_lenticule_width,w,10.) #10.)#alpha=0.1)

        z_raster = reconstruct_boundaries(lenticules_location_bottom,lenticules_location_top,size=x.squeeze().shape,lenticule_min = min_lenticule_width,lenticule_max = max_lenticule_width)

        z_raster = torch.from_numpy(z_raster).unsqueeze(0).unsqueeze(0).float().to(self.device)
        x = x.to(self.device)
        #colorize = ColorRestoration(max_lenticule_width).to(self.device)
        #_ = colorize(x,z)
        
        #idx_color = [w/4,w/2,3*w/4]
        
        idx_color = [5.*w/19.,9.*w/19.,14.*w/19.]

        
        #idx_color = [colorize.RGB_extractor.idx[0].item(),colorize.RGB_extractor.idx[1].item(),colorize.RGB_extractor.idx[2].item()]


        x_np = x.cpu().numpy().squeeze()
        H,W = x_np.shape
        
        first = 0
        if lenticules_location_top[0] < 0 or lenticules_location_bottom[0] < 0:
            first = 1
        
        last = len(lenticules_location_top)
        if lenticules_location_top[-1] > W or lenticules_location_bottom[-1] > W:
            last = len(lenticules_location_top) - 1
            
        x_mosaic = np.zeros((3,H,3*(last-first-1)))
    
            
        for c in range(0,3):
            for n,bottom,top in zip(range(len(lenticules_location_top[first:last-1])),lenticules_location_bottom[first:last-1],lenticules_location_top[first:last-1]):

                if bottom < 0 or top < 0 or bottom + idx_color[c] + 1 > W or top + idx_color[c] + 1> W:
                    assert(1==0)
                    #continue

                idx_row = np.arange(0,H)
                idx_col = bottom + idx_color[c] + idx_row * (top-bottom)/H
                
                idx_col_pre = np.floor(idx_col).astype(int)
                idx_col_post = np.ceil(idx_col).astype(int)

                weight_pre = 1 - (idx_col - np.floor(idx_col))
                weight_post = 1 - (np.ceil(idx_col) - idx_col)
                weight_pre = weight_pre / (weight_pre + weight_post)
                weight_post = 1 - weight_pre

                idx_row = idx_row.astype(int)
        
                extr_col = weight_pre * x_np[idx_row,idx_col_pre] + weight_post * x_np[idx_row,idx_col_post]
                extr_col_den = filters.median(extr_col[:,None],disk(6))
                x_mosaic[c,:,3*n+c] = torch.from_numpy(extr_col_den.squeeze()) #x_filter[idx_row,idx_col]

        x_mosaic = torch.from_numpy(x_mosaic).float().unsqueeze(0)
        #print(x_mosaic.shape)

        x_mosaic = F.interpolate(x_mosaic,None,scale_factor=(3/w,1),recompute_scale_factor=True) #,mode=F.bilinear)
        #print(x_mosaic.shape)
        #print("end")

        return x_mosaic,z_raster.squeeze(0),lenticules_location_bottom,lenticules_location_top 
    

    def demosaicing(self,x_mosaic):

        network_input_size = 256

        B,C,H,W = x_mosaic.shape

        H_large,W_large = (H//256 +1)*256, (W//256 +1)*256
        x_padded = F.pad(x_mosaic,(0,W_large-W,0,H_large-H,))
        
        u_dict = get_interpolation_sets_extended(x_padded.squeeze())
        u_dict["u_r"] = u_dict["u_r"].unsqueeze(0).to(self.device)
        u_dict["u_g"] = u_dict["u_g"].unsqueeze(0).to(self.device)
        u_dict["u_b"] = u_dict["u_b"].unsqueeze(0).to(self.device)
        sample = {"x":x_padded.to(self.device),
                 "mask":(x_padded>0.).float().to(self.device)
                 }

        sample = {**sample,**u_dict}

        self.model_color.eval()
        with torch.no_grad():
            out_dict = self.model_color(sample)

        y_padded = out_dict["y_pred"]
        y = y_padded[:,:,0:H,0:W]

        return y.squeeze()


if __name__ == '__main__':

    args = parser.parse_args()
    developingSuite = ColorizeSuite(args)

    developingSuite.colorize_dataset()


# python3 developing_suite.py --model=DeeperUNet --save-dir=./results/ --save-model=best --mode=only_train --workers=12 --data=slice_dolce --dataroot=/scratch2/deepdoLCE_dataset --epochs=20 --logstep-train=20 --batch-size=16 --train-val-ratio=0.9 --optimizer=sgd --lr=0.01 --momentum=0.99 --lr-scheduler=smart --lr-step=3 --lr-gamma=0.1

