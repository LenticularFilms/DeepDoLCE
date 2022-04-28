import sys, argparse, os, time
import numpy as np

import torch
import torch.nn.functional as F
from skimage import filters
from skimage.morphology import disk

if 'ipykernel' in sys.modules:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm


from models import get_model
from utils.other import new_log
from dataloading import get_dataloaders
from dataloading.destriping_dataset import get_interpolation_sets_extended
from utils.plot_utils import *
from models.color_restoration import ColorRestoration
from models.interpolate_stripes import interpolateStripes
from models.lenticules_vectorization import *

parser = argparse.ArgumentParser(description="colorizing script")

#### general parameters #####################################################
parser.add_argument('--tag', default="__",type=str)
parser.add_argument("--device",default="cuda",type=str,choices=["cuda", "cpu"])
parser.add_argument("--save-dir",help="Path to directory where models and logs should be saved saved")
parser.add_argument('--mode',default="only_train",type=str,choices=["None","train","test","train_and_test"],help="mode to be run")
parser.add_argument("--save-only-color",action='store_true',default=False,help="avoids non useful calculation if only interested in obtaining the colored output")


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
parser.add_argument("--style",type=str,default="new_dolce",choices=["new_dolce","old_dolce","interpolate"],help="type of colorization")
parser.add_argument("--order",type=str,default="nope",choices=["nearest","linear","cubic","nope"],help="type of interpolation")
parser.add_argument("--lambda1", type=float, default=1.,help="width variation regularization")
parser.add_argument("--lambda2", type=float, default=10.,help="width absolute value regularization")

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
        if not os.path.exists(os.path.join(self.experiment_folder,"lenticules_combo/")):
            os.mkdir(os.path.join(self.experiment_folder,"lenticules_combo/"))

        for sample in tqdm(self.dataloaders[dataset_name].dataset,leave=False):
            
            x = sample["x"]
            
            y,z,z_rough,_,_,_,_ = self.colorize_image(x)

            if not self.args.save_only_color:
                combo_z = torch.zeros((3,z.shape[-2],z.shape[-1]))
                combo_z[0:1,:,:] = z_rough
                combo_z[1:2,:,:] = z

                img = Image.fromarray((255*z).cpu().numpy().astype('uint8').squeeze(), 'L')
                img.save(os.path.join(self.experiment_folder,"lenticules/","lenticules_"+sample["name"]))

                img = Image.fromarray((255*combo_z).squeeze().permute(1,2,0).cpu().numpy().astype('uint8').squeeze(), 'RGB')
                img.save(os.path.join(self.experiment_folder,"lenticules_combo/","lenticules_combo_"+sample["name"]))

            img = Image.fromarray((255*y).squeeze().permute(1,2,0).cpu().numpy().astype('uint8'), 'RGB')
            img.save(os.path.join(self.experiment_folder,"color/","color"+sample["name"]))


    def colorize_image(self,x):

        while len(x.shape) < 4:
            x = x.unsqueeze(0)

        return_device = x.device

        x = x.to(self.device)
        t_0 = time.time()
        z = self.process_image_lenticules(x)
        #print("raster prediction: {}".format(time.time()-t_0))
        #t_0 = time.time()
        if self.args.style == "old_dolce": #old style -> same color throughout a single lenticule
            _, Pxx_den = sg.periodogram(np.sum(z.cpu().numpy().squeeze(),axis=0))
            w = z.squeeze().shape[1] / ( 50 + np.argmax(Pxx_den[50:350]))

            delta_max = 18
            min_lenticule_width = int(np.floor(w)-1)
            max_lenticule_width = int(np.ceil(w)+1)
            u = build_score_matrix(1-z.cpu().numpy().squeeze(),delta_max)

            bottom,top  = optimize_locations(u,delta_max,min_lenticule_width,max_lenticule_width,w,lambda1=self.args.lambda1,lambda2=self.args.lambda2)

            z_raster = reconstruct_boundaries(bottom,top,size=x.squeeze().shape,lenticule_min = min_lenticule_width,lenticule_max = max_lenticule_width)
            z_raster = torch.from_numpy(z_raster).unsqueeze(0).unsqueeze(0).float().to(self.device)
            x_stripe,_,_,_ = self.extract_stripes(z,x)

            colorize = ColorRestoration(max_lenticule_width).to(self.device)
            out_dict = colorize(x,z_raster)
            
            if self.args.save_only_color:
                z_raster = None

            y = out_dict["y"].squeeze()#.detach().cpu()
        elif self.args.style == "new_dolce": # -> learned destriping
            t_0 = time.time()
            x_stripe,z_raster,bottom,top = self.extract_stripes(z,x)
            #print("vectorization: {}".format(time.time()-t_0))
            #t_0 = time.time()
            y = self.destriping(x_stripe) # -> interpolation destriping
            #print("destriping: {}".format(time.time()-t_0))
            #t_0 = time.time()
        elif self.args.style == "interpolate":
            x_stripe,z_raster,bottom,top = self.extract_stripes(z,x)
            y = torch.from_numpy(interpolateStripes(x_stripe.cpu().numpy().squeeze(),order=self.args.order)).float()
        else:
            raise NotImplementedError
        
        gain_matrix = torch.tensor([[0.789,0.154,0.057],
                                    [-0.286,1.195,0.060],
                                    [-0.049,0.035,1.035]])
        gain_matrix = gain_matrix.unsqueeze(2).unsqueeze(2).float().to(y.device)
        y_orig = y.clone()
        y = torch.clamp(torch.sum(y.unsqueeze(0)*gain_matrix,1),0,1)
        
        return (y.detach().to(return_device),
                None if z_raster is None else z_raster.detach().to(return_device),
                z.detach().to(return_device).squeeze(0),
                x_stripe.detach().to(return_device),
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

    def extract_stripes(self,z,x):
        
        t_0 = time.time()
        _, Pxx_den = sg.periodogram(np.sum(z.cpu().numpy().squeeze(),axis=0))
        w = z.squeeze().shape[1] / ( 50 + np.argmax(Pxx_den[50:350]))
        #print("period ex: {}".format(time.time()-t_0))
        #t_0 = time.time()

        delta_max = 18
        min_lenticule_width = int(np.floor(w)-1)
        max_lenticule_width = int(np.ceil(w)+1)
        u = build_score_matrix(1-z.cpu().numpy().squeeze(),delta_max)
        #print("build score matrix: {}".format(time.time()-t_0))
        #t_0 = time.time()

        lenticules_location_bottom,lenticules_location_top  = optimize_locations(u,delta_max,min_lenticule_width,max_lenticule_width,w,lambda1=self.args.lambda1,lambda2=self.args.lambda2) 

        
        if not self.args.save_only_color:
            t_0 = time.time()
            z_raster = reconstruct_boundaries(lenticules_location_bottom,lenticules_location_top,size=x.squeeze().shape,lenticule_min = min_lenticule_width,lenticule_max = max_lenticule_width)
            #print("raster rec: {}".format(time.time()-t_0))
            #t_0 = time.time()

            z_raster = torch.from_numpy(z_raster).unsqueeze(0).unsqueeze(0).float().to(self.device).squeeze(0)
        else:
            z_raster =None

        x = x.to(self.device)
        
        idx_color = [5.*w/19.,9.*w/19.,14.*w/19.]

        x_np = x.cpu().numpy().squeeze()
        H,W = x_np.shape
        
        first = 0
        if lenticules_location_top[0] < 0 or lenticules_location_bottom[0] < 0:
            first = 1
        
        last = len(lenticules_location_top)
        if lenticules_location_top[-1] > W or lenticules_location_bottom[-1] > W:
            last = len(lenticules_location_top) - 1
            
        x_stripe = np.zeros((3,H,3*(last-first-1)))

        t_0 = time.time()
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
                x_stripe[c,:,3*n+c] = torch.from_numpy(extr_col_den.squeeze()) #x_filter[idx_row,idx_col]

        #print("loop time: {}".format(time.time()-t_0))

        x_stripe = torch.from_numpy(x_stripe).float().unsqueeze(0)

        x_stripe = F.interpolate(x_stripe,None,scale_factor=(3/w,1),recompute_scale_factor=True) #,mode=F.bilinear)

        return x_stripe,z_raster,lenticules_location_bottom,lenticules_location_top 
    

    def destriping(self,x_stripe):

        network_input_size = 256

        B,C,H,W = x_stripe.shape

        H_large,W_large = (H//network_input_size +1)*network_input_size, (W//network_input_size +1)*network_input_size
        x_padded = F.pad(x_stripe,(0,W_large-W,0,H_large-H,))
        
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

