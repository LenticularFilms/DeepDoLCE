import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from torchvision import models


class BinaryResnet(nn.Module):
    def __init__(self):
        super(BinaryResnet, self).__init__()
        
        self.net = models.resnet50(pretrained=True,)
        self.net.fc = nn.Linear(self.net.fc.in_features,1)

    def forward(self, input):
        return torch.sigmoid(self.net(input))


class UResNet(nn.Module):

    def __init__(self, args, data_stats):
        super(UResNet, self).__init__()
        
        self.args = args
        input_channels = 3 #data_stats["input_channels"]
        output_channels = 3 * 6 #data_stats["output_channels"]
        
        self.model = smp.Unet('resnet50', classes=output_channels,in_channels=input_channels, activation=None,encoder_weights="imagenet")
    

    def forward(self, input):

        if type(input) is dict:
            x = input['x']
        else:
            x = input
        
        y = self.model(x)
        
        
        y_r = y[:,0:6]
        y_g = y[:,6:12]
        y_b = y[:,12:]

        r = torch.sum(F.softmax(y_r,dim=1) * input["u_r"],dim=1,keepdim=True)
        g = torch.sum(F.softmax(y_g,dim=1) * input["u_g"],dim=1,keepdim=True)
        b = torch.sum(F.softmax(y_b,dim=1) * input["u_b"],dim=1,keepdim=True)

        #r =  y[:,0:1,:,:] * input["u_r"][:,0:1,:,:] + (1 - y[:,0:1,:,:]) * input["u_r"][:,1:,:,:] 
        #g =  y[:,1:2,:,:] * input["u_g"][:,0:1,:,:] + (1 - y[:,1:2,:,:]) * input["u_g"][:,1:,:,:] 
        #b =  y[:,2:,:,:] * input["u_b"][:,0:1,:,:] + (1 - y[:,2:,:,:]) * input["u_b"][:,1:,:,:] 

        y = torch.cat([r,g,b],dim=1)

        #y = input["mask"] * x + (1 - input["mask"]) * y
        
        
        #y = torch.sigmoid(y)
        return {"y_pred": y}


    def get_loss(self, sample, output):

        criterion_l1 = nn.L1Loss()
        criterion_mse = nn.MSELoss()
        l1_loss = criterion_l1(output["y_pred"], sample["y"])
        mse_loss = criterion_mse(output["y_pred"], sample["y"])


        if self.args.loss == "l1":
            loss = l1_loss
        elif self.args.loss == "mse":
            loss = mse_loss
        else:
            raise NotImplementedError

        return  loss, { "optimization_loss": loss.detach().item(),
                        "mse_loss": mse_loss.detach().item(),
                        "l1_loss": l1_loss.detach().item()}


class Adversarialmodel(nn.Module):
    def __init__(self, args, data_stats):
        super(Adversarialmodel, self).__init__()

        self.args = args
        input_channels = data_stats["input_channels"]
        output_channels = data_stats["output_channels"]

        self.discriminator = BinaryResnet() 
        self.generator = UResNet(args, data_stats)

        self.adversarial_loss = torch.nn.BCELoss()


    def forward(self, input, optimizer=None):
        
        x = input["x"]

        valid = torch.ones((x.shape[0], 1)).to(x.device)
        fake = torch.zeros((x.shape[0], 1)).to(x.device)

        y_demosaic = self.generator(input)["y_pred"]

        # Loss measures generator's ability to fool the discriminator
        g_loss =  self.adversarial_loss(self.discriminator(y_demosaic),0.95 * valid)
        
        if "y" in input:
            y = input["y"]
            l1_loss = F.l1_loss(y_demosaic,input["y"])
        else:
            l1_loss = 0
            
        g_loss += 10 * l1_loss

        if self.training:

            g_loss.backward()

            for param in self.discriminator.parameters():
                if param.grad is not None:
                    param.grad.fill_(0.)

        if "y" in input:
            noise = 0.001 * torch.randn_like(y)
            disc_target = self.discriminator(y+noise)
            real_loss = self.adversarial_loss(disc_target,0.95 *  valid)
        
            noise = 0.001 * torch.randn_like(y_demosaic)
            disc_demosaic = self.discriminator(y_demosaic.detach()+noise) #(noise + x_mod).detach())
            fake_loss = self.adversarial_loss(disc_demosaic,0.05 + fake)
            d_loss = 0.5 * (real_loss + fake_loss)
        else:
            disc_target = None
            real_loss = 0
            d_loss = 0
            disc_demosaic = 0

        return {"y_pred":y_demosaic,
                "g_loss": g_loss,
                "d_loss": d_loss,
                "l1_loss": l1_loss,
                "disc_target": disc_target,
                "disc_demosaic": disc_demosaic}


    def get_loss(self, sample, output):

        d_loss = output["d_loss"]
        g_loss = output["g_loss"]
        l1_loss = output["l1_loss"]

        return  d_loss, { "optimization_loss": d_loss.detach().item(),
                          "discriminator_loss": d_loss.detach().item(),
                          "l1_loss":l1_loss.detach().item(),
                          "generator_loss": g_loss.detach().item()}