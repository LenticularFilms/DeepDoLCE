import sys, argparse, os
import numpy as np

import torch
import torch.nn.functional as F

from models import get_model
from torch.utils.tensorboard import SummaryWriter
from utils.other import new_log
from dataloading import get_dataloaders
from utils.plot_utils import *

if 'ipykernel' in sys.modules:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

from models.lenticule_refinement import lenticuleRefinment
from models.color_restoration import ColorRestoration

parser = argparse.ArgumentParser(description="training script")

#### general parameters #####################################################
parser.add_argument('--tag', default="__",type=str)
parser.add_argument("--device",default="cuda",type=str,choices=["cuda", "cpu"])
parser.add_argument("--save-dir",help="Path to directory where models and logs should be saved saved")
parser.add_argument("--logstep-train", default=10,type=int,help="iterations step for training log")
parser.add_argument("--save-model", default="No",choices=['last','best','No'],help="which model to save")
parser.add_argument("--val-every-n-epochs", type=int, default=1,help="interval of training epochs to run the validation")
parser.add_argument('--resume', type=str, default=None,help='path to resume the model if needed')
parser.add_argument('--mode',default="only_train",type=str,choices=["None","train","test","train_and_test"],help="mode to be run")

#### data parameters ##########################################################
parser.add_argument("--data",default="full_image",type=str,choices=["square_patch","full_image"],help="dataset selection")
parser.add_argument("--datafolder",default="/scratch/code/data_dolce",type=str,help="root directory of the dataset")
parser.add_argument("--workers", type=int, default=4,metavar="N",help="dataloader threads")
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--input-width", type=int, default=256, help="width of the input patch")
parser.add_argument("--train-val-ratio",type=float,default=0.9,help="ratio of the dataset to use for training")
parser.add_argument("--normalize-patch",action='store_true',default=False,help="normalize patches")

#### optimizer parameters #####################################################
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--optimizer',default='sgd', choices=['sgd','adam'])
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--w-decay', type=float, default=1e-5)
parser.add_argument('--lr-scheduler', type=str, default='no',choices=['no','step', 'smart'],help='lr scheduler mode')
parser.add_argument('--lr-step', type=int, default=4,help=' number of epochs between decreasing steps applies to lr-scheduler in [step, exp]')
parser.add_argument('--lr-gamma', type=float, default=0.9,help='decrease rate')

#### model parameters #####################################################
parser.add_argument("--model",type=str,help="model to run")
parser.add_argument("--loss", default="CE", type=str, choices=["CE","wCE","focal"], help=" loss ")


class DevelopingSuite(object):

    def __init__(self, args):

        self.args = args
        
        self.dataloaders, self.data_stats =  get_dataloaders(args)

        self.device = torch.device("cuda" if torch.cuda.is_available() and args.device=="cuda" else "cpu")
        self.model = get_model(args,self.data_stats)
        self.model.to(self.device)
        if args.resume is not None:
            self.resume(path=args.resume)

        if "train" in args.mode:
            self.experiment_folder = new_log(args.save_dir,args.model + "_" + args.tag,args=args)
            self.writer = SummaryWriter(log_dir=self.experiment_folder)

            if args.optimizer == 'adam':
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr,weight_decay=0.)
            elif args.optimizer == 'sgd':
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, momentum=self.args.momentum,weight_decay=0.)

            if args.lr_scheduler == 'step':
                self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
            elif args.lr_scheduler == 'smart':
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=args.lr_step, factor=args.lr_gamma)
            else:
                self.scheduler = None
            
            self.epoch = 0
        

    def train_and_eval(self):

        with tqdm(range(0,self.args.epochs),leave=True) as tnr:
            tnr.set_postfix(training_loss= np.nan, validation_loss= np.nan,best_validation_loss = np.nan)
            for n in tnr:
                
                self.training(tnr)
                
                if self.epoch % self.args.val_every_n_epochs == 0:
                    self.validate()

                if self.args.lr_scheduler == "step":
                    self.scheduler.step()
                    self.writer.add_scalar('log_lr', np.log10(self.scheduler.get_lr()), self.epoch )
                
                self.epoch += 1

                if self.args.save_model == "last":
                    self.save_model()

    def training(self,tnr=None):
        
        if "train_stats" not in dir(self):
            self.train_stats = {}
            self.val_stats = {}
            self.val_stats["validation_loss"] = np.nan
            self.val_stats["ref_validation_loss"] = np.nan
            self.val_stats["best_validation_loss"] = np.nan
            self.iter = 0

        self.model.train()
        accumulated_loss = 0
        accumulated_std_loss = 0
        with tqdm(self.dataloaders["train"],leave=False) as inner_tnr:
            inner_tnr.set_postfix(training_loss= np.nan)
            for en,sample in enumerate(inner_tnr):
                sample = self.to_device(sample)

                self.optimizer.zero_grad()
                output = self.model(sample)

                loss,std_loss = self.model.get_loss(sample,output)
                accumulated_loss += loss.item()
                accumulated_std_loss += std_loss.item()
                
                loss.backward() #retain_graph=True)
                self.optimizer.step()

                self.iter += 1

                if (en+1) % self.args.logstep_train == 0:
                    self.train_stats['train_loss'] = accumulated_loss / self.args.logstep_train
                    self.train_stats['train_std_loss'] = accumulated_std_loss / self.args.logstep_train
                    
                    accumulated_loss = 0
                    accumulated_std_loss = 0
                    inner_tnr.set_postfix(training_loss=self.train_stats['train_loss'])
                    inner_tnr.set_postfix(training_std_loss=self.train_stats['train_std_loss'])
                    if tnr is not None:
                        tnr.set_postfix(training_loss=self.train_stats['train_loss'],validation_loss= self.val_stats["validation_loss"],best_validation_loss = self.val_stats["best_validation_loss"])
                    self.writer.add_scalar('training/training loss', self.train_stats['train_loss'], self.iter )
                    self.writer.add_scalar('training/training std loss', self.train_stats['train_std_loss'], self.iter )


    def validate(self,tnr=None,save=True):

        with torch.no_grad():
            self.model.eval()
            validation_loss = 0
            validation_std_loss = 0
            for sample in tqdm(self.dataloaders["val"], leave=False):
                sample = self.to_device(sample)

                output = self.model(sample)
                
                loss,std_loss = self.model.get_loss(sample,output)
                validation_loss += loss.item()
                validation_std_loss += std_loss.item()
            
            add_tensorboard_images(self.writer, sample, output, global_step=self.epoch,args=self.args)

            self.val_stats["validation_loss"] = validation_loss / len(self.dataloaders["val"])
            self.val_stats["validation_std_loss"] = validation_std_loss / len(self.dataloaders["val"])
            if not self.val_stats["best_validation_loss"] < self.val_stats["validation_loss"]:
                self.val_stats["best_validation_loss"] = self.val_stats["validation_loss"]
                if save and self.args.save_model == "best":
                    self.save_model()

            self.writer.add_scalar('validation/validation loss', self.val_stats["validation_loss"], self.epoch)
            self.writer.add_scalar('validation/validation std loss', self.val_stats["validation_std_loss"], self.epoch)
            self.writer.add_scalar('validation/best validation loss', self.val_stats["best_validation_loss"], self.epoch)

    def test(self):
        
        with torch.no_grad():
            self.model.eval()
            test_loss = 0.
            for sample in tqdm(self.dataloaders["test"], leave=False):
                sample = self.to_device(sample)
                
                output = self.model(sample)
                loss = self.model.get_loss(sample, output,training=False)

                test_loss += loss.item()
                        
        test_loss = test_loss / len(self.dataloader["test"])
        print("test loss: {}".format(test_loss))
        
        return test_loss

    def save_model(self):
        torch.save(self.model.state_dict(), os.path.join(self.experiment_folder,"model.pth.tar"))
        
    def resume(self,path):
        if not os.path.isfile(path):
            raise RuntimeError("=> no checkpoint found at '{}'".format(path))
        checkpoint = torch.load(path)
        
        self.model.load_state_dict(torch.load(path))
        
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


    def colorize_dataset(self,dataset_name="train"):

        if not os.path.exists(os.path.join(self.experiment_folder,"lenticules/")):
            os.mkdir(os.path.join(self.experiment_folder,"lenticules/"))
        if not os.path.exists(os.path.join(self.experiment_folder,"color/")):
            os.mkdir(os.path.join(self.experiment_folder,"color/"))
        if not os.path.exists(os.path.join(self.experiment_folder,"rgb_filter/")):
            os.mkdir(os.path.join(self.experiment_folder,"rgb_filter/"))

        for sample in tqdm(self.dataloaders[dataset_name].dataset,leave=False):
            
            x = sample["x"].unsqueeze(0)

            y,z_rough,z,rgb_filter,_ = self.colorize_image(x)

            rgb_filter_z = rgb_filter.cpu() + z.cpu()
            img = Image.fromarray((255*rgb_filter_z.squeeze().permute(1,2,0)).numpy().astype('uint8').squeeze(), 'RGB')
            img.save(os.path.join(self.experiment_folder,"rgb_filter/","rgb_filter"+sample["name"]))

            img = Image.fromarray((255*z).cpu().numpy().astype('uint8').squeeze(), 'L')
            img.save(os.path.join(self.experiment_folder,"lenticules/","lenticules_"+sample["name"]))

            img = Image.fromarray((255*y.squeeze().permute(1,2,0)).cpu().numpy().astype('uint8').squeeze(), 'RGB')
            img.save(os.path.join(self.experiment_folder,"color/","color"+sample["name"]))

    def colorize_image(self,x):

        while len(x.shape) < 4:
            x = x.unsqueeze(0)

        return_device = x.device

        x = x.to(self.device)

        z_rough = self.process_image(x)

        lenticules_dict,objective_refinement = lenticuleRefinment(z_rough,device=self.device,)
            
        colorize = ColorRestoration(lenticules_dict["w"]).to(self.device)

        out_dict = colorize(x,lenticules_dict["z"])
        y = out_dict["y"]
        rgb_filter = out_dict["rgb_filter_image"]

        return (y.detach().to(return_device),
               z_rough.detach().to(return_device),
               lenticules_dict["z"].detach().to(return_device),
               rgb_filter.detach().to(return_device),
               objective_refinement)


    def process_image(self,x):

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

                    y_ext = self.model(x_ext.to(self.device))["y_pred_sigmoid"].to(x.device)

                    y_padded[:,:,delta_pix*i:min(x_padded.shape[2],delta_pix*(i+1)),delta_pix*j:min(x_padded.shape[3],delta_pix*(j+1))] = y_ext

        y = y_padded[:,:,0:x.shape[2],0:x.shape[3]]

        return y.to(x.device).detach()


if __name__ == '__main__':

    args = parser.parse_args()
    developingSuite = DevelopingSuite(args)

    developingSuite.process_images()

    developingSuite.writer.close()

#python3 developing_suite.py --model=DeeperUNet --save-dir=./results/ --save-model=best --mode=only_train --workers=12 --data=slice_dolce --dataroot=/scratch2/deepdoLCE_dataset --epochs=20 --logstep-train=20 --batch-size=16 --train-val-ratio=0.9 --optimizer=sgd --lr=0.01 --momentum=0.99 --lr-scheduler=smart --lr-step=3 --lr-gamma=0.1

