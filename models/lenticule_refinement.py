import torch 
import numpy as np
import torch.nn.functional as F
from PIL import Image
import sys
import scipy.signal as signal

if 'ipykernel' in sys.modules:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

def lenticuleRefinment(z_rough,device="cpu",S=100):
    
    device_output = z_rough.device

    # estimated lenticul width
    _, Pxx_den = signal.periodogram(np.sum(z_rough.detach().cpu().squeeze().numpy(),axis=0))
    w = z_rough.shape[3] / ( 150 + np.argmax(Pxx_den[150:350]))

    z_rough = z_rough.to(device)
    smooth_H = 51
    z_rough =  F.avg_pool2d(z_rough,kernel_size=(smooth_H,1),stride=(1,1),padding=(smooth_H//2,0),count_include_pad=False)

    # build filter for consistency 
    w_large = int(np.ceil(w+0.))
    w_small = int(np.floor(w-0.))
    w_huge = int(np.ceil(1.1*w))

    kernel_large = torch.ones(1,1,1,w_large).to(device)
    kernel_small = torch.ones(1,1,1,w_small).to(device)
    kernel_huge = torch.ones(1,1,1,w_huge).to(device)
    
    tv_kernel_h = torch.ones(1,1,2,1).to(device)
    tv_kernel_h[0,0,1,0] = -1

    z = z_rough.clone()
    z.requires_grad = True

    optimizer = torch.optim.Adam([z], lr=0.01,weight_decay=0.)  #,momentum=0.9)
    loss_v = torch.zeros(S)

    lam = 10.
    with tqdm(range(0,S),leave=False) as tnr:
        tnr.set_postfix(objective= np.nan)
        for iter in tnr:
            
            global count_eval
            count_eval = 0 

            def objective():
                loss = 0

                #z = torch.sigmoid(logit_z)
                with torch.no_grad():
                    z.data = torch.clamp(z.data,0,1)

                
                if torch.is_grad_enabled():
                    optimizer.zero_grad()

                # likelihood
                loss += - torch.sum(z  * (z_rough) ) # + (1-z) * (1-z_rough)) 

                # horizontal prior
                z_large = torch.conv2d(z,kernel_large,padding = 0, bias=None, stride=(1,1))
                z_huge = torch.conv2d(z,kernel_huge,padding = 0, bias=None, stride=(1,1))
                z_small = torch.conv2d(z,kernel_small,padding = 0, bias=None, stride=(1,1))

                constr_small = z_small - 1 #torch.clamp(z_small - 1,0)
                constr_large =  1 - z_large # torch.clamp(1 - z_large,0)
                constr_huge = z_huge - 2 # torch.clamp(z_huge - 2,0)

                loss += 100 * (torch.sum(torch.clamp(1 - z_large,0)**2) \
                        + torch.sum(torch.clamp(z_small - 1,0)**2) \
                        + torch.sum(torch.clamp(z_huge - 2,0)**2) )
                
                # vertical prior
                z_tv_h = torch.conv2d(z,tv_kernel_h,padding = 0, bias=None, stride=(1,1))

                loss += 10 * torch.sum(torch.abs(z_tv_h)**2)
                loss += 10 * torch.sum(torch.abs(z_tv_h))

                if loss.requires_grad:
                    loss.backward()

                return loss.item()


            loss_v[iter] = objective()
            tnr.set_postfix(objective=loss_v[iter])
            optimizer.step() 
        
                       
    with torch.no_grad():

        z.data = torch.clamp(z.data,0,1)

        z_large = torch.conv2d(z,kernel_large,padding = 0, bias=None, stride=(1,1))
        z_huge = torch.conv2d(z,kernel_huge,padding = 0, bias=None, stride=(1,1))
        z_small = torch.conv2d(z,kernel_small,padding = 0, bias=None, stride=(1,1))


        constr_small = torch.clamp(z_small - 1,0)
        constr_large = torch.clamp(1 - z_large,0)
        constr_huge = torch.clamp(z_huge - 2,0)


    return {"z": z.detach().to(device_output), "w": w}, loss_v