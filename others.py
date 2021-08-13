import numpy as np
import torch
from models.color_restoration import ColorRestoration
import scipy.signal as sg
import scipy.sparse as sp
import scipy
import scipy.optimize as opt
import matplotlib.pyplot as plt

def build_score_matrix(x,delta_max):

    H,W = x.shape
    y = np.zeros((2*delta_max + 1,W))

    for c in range(0,W):
        for delta in range(-delta_max,delta_max+1):

            idx_row = np.arange(0,H)
            idx_col = np.round(c + idx_row * (delta/H)).astype(int)
            idx_row = idx_row.astype(int)
            
            mask = np.logical_and(idx_col >= 0,idx_col < W)
            idx_col = idx_col[mask]
            idx_row = idx_row[mask]

            #print(idx_row)
            #print("#################")
            #print(idx_col)

            val = x[idx_row,idx_col]
            val = np.mean(val)
            y[delta+delta_max,c] = val

    return y

def build_score_matrix_square(x,delta_max):

    H,W = x.shape
    y = sp.dia_matrix((W,W)) # np.zeros((2*delta_max + 1,W))

    for bottom in range(0,W):
        for top in range(max(0,bottom-delta_max),min(W,bottom+delta_max)):

            idx_row = np.arange(0,H)
            idx_col = np.round(bottom + idx_row * (top-bottom)/H).astype(int)
            idx_row = idx_row.astype(int)
            
            mask = np.logical_and(idx_col >= 0,idx_col < W)
            idx_col = idx_col[mask]
            idx_row = idx_row[mask]

            #print(idx_row)
            #print("#################")
            #print(idx_col)

            val = x[idx_row,idx_col]
            val = np.mean(val)
            y[delta+delta_max,c] = val

    return y

def my_find_min(y,delta_max,min_lenticule_width,max_lenticule_width,w): #,alpha):

    _,W = y.shape
    minimum_list = []
    valleys_depth_list = []

    y_full = y.copy()
    #y_full[:,6:] += alpha * y_diff[:,6:]
    #y_full[:,9:] += alpha * y_diff[:,9:]
    #y_full[:,11:] += alpha * y_diff[:,11:]

    for delta in range(-delta_max,delta_max+1):
        z = y_full[delta+delta_max]

        valleys = []
        valley_one = W//2 + np.argmin(z[W//2:W//2+max_lenticule_width])
        #valley_one = np.argmin(z[0:max_lenticule_width])

        valleys.append(valley_one)
        current_valley = valley_one
        while current_valley + min_lenticule_width < W :
            lb = int(current_valley)+min_lenticule_width
            ub = min(int(current_valley)+max_lenticule_width,W)
            j = np.argmin(z[lb:ub])
            current_valley = j + lb
            valleys.append(current_valley)
        
        current_valley = valley_one
        while current_valley - min_lenticule_width >= 0 :
            lb = max(int(current_valley)-max_lenticule_width,0)
            ub = int(current_valley)-min_lenticule_width
            if lb != ub:
                j = np.argmin(z[lb:ub])
            else:
                j = 0
            current_valley = j + lb
            valleys.append(current_valley)

        valleys = np.sort(valleys)

        valleys_depth = np.sum(z[np.round(valleys).astype(int)])

        minimum_list.append(np.array(valleys))
        valleys_depth_list.append(valleys_depth)
    
    idx = np.argmin(np.array(valleys_depth_list))

    lenticules_location_bottom = minimum_list[idx]
    lenticules_location_top = minimum_list[idx] + (idx - delta_max)

    #print(np.min(valleys_depth_list))

    return lenticules_location_bottom,lenticules_location_top,(idx - delta_max)


def reconstruct_boundaries(lenticules_location_bottom,lenticules_location_top,size,lenticule_min,lenticule_max):
    H,W = size
    z = np.zeros((H,W))
    
    for bottom,top in zip(lenticules_location_bottom,lenticules_location_top):

        idx_row = np.arange(0,H)
        idx_col = bottom + idx_row * (top - bottom)/H
        idx_col_floor = np.floor(idx_col).astype(int)
        idx_col_ceil = np.ceil(idx_col).astype(int)
        idx_row = idx_row.astype(int)
            
        mask = np.logical_and(idx_col_floor >= 0,idx_col_ceil < W)
        idx_col = idx_col[mask]
        idx_col_floor = idx_col_floor[mask]
        idx_col_ceil = idx_col_ceil[mask]
        idx_row = idx_row[mask]

        z[idx_row,idx_col_floor] = idx_col_ceil - idx_col
        z[idx_row,idx_col_ceil] = idx_col - idx_col_floor

        mask2 = idx_col_floor==idx_col_ceil
        z[idx_row[mask2],idx_col_ceil[mask2]] = 1.

    return z


def optimize_locations(y,delta_max,min_lenticule_width,max_lenticule_width,w,reg):

    initial_bottom,initial_top,init_delta  = my_find_min(y,delta_max,min_lenticule_width,max_lenticule_width,w)
    
    M = len(initial_bottom) - 1
    
    #rgb_loc = np.array([w/3,w/2,2*w/3])
    x0 = np.concatenate([initial_bottom[1:],initial_top[1:]-initial_bottom[1:]],axis=0)
    #x0 = np.concatenate([initial_bottom[1:],initial_top[1:]],axis=0)
    
    f = scipy.interpolate.RectBivariateSpline(np.arange(0,y.shape[0])-delta_max, np.arange(0,y.shape[1]),y)
    #f_diff = scipy.interpolate.RectBivariateSpline(np.arange(0,y_diff.shape[0])-delta_max, np.arange(0,y_diff.shape[1]),y_diff)

    def obj(x,f): #,init_delta):
        bottom = x[0:M]
        delta = x[M:2*M] #- x[0:M]
        #rgb_loc = x[-3:]
        #reg1 = 0.1*np.sum((delta - init_delta)**2)
        val = f(delta,bottom,grid=False) #,dy=1) #,dy=1)

        val_tot = np.sum(val) # + alpha * (dval_r+dval_g+dval_b) - 0.1 *(val_r+val_g+val_b))

        D1 = - np.eye(M-1,M=M,k=0) + np.eye(M-1,M=M,k=1)
        #Z = np.zeros((M-1,M))
        #Atop = np.concatenate([D,Z],axis=1)
        #Abot = np.concatenate([Z,D],axis=1)
        #A = np.concatenate([Atop,Abot],axis=0)
        e = np.ones((1, M))
        D = scipy.sparse.spdiags(np.vstack((e, -2*e, e)), range(3), M-2, M)

        penalty11 = ( D1 @ bottom - w ).T @ (D1 @ bottom - w)
        #penalty2 = ( D @ delta - init_delta ).T @ (D @ delta - init_delta)

        penalty1 = ( D @ bottom ).T @ (D @ bottom )
        penalty2 = ( D @ delta  ).T @ (D @ delta)

        #print(penalty1.shape)

        return val_tot + reg * (penalty1 + penalty2) + penalty11


    def jac(x,f): #,init_delta):
        bottom = x[0:M]
        delta = x[M:2*M] #- x[0:M]
        #rgb_loc = x[-3:]

        #df_rgb_log = np.zeros_like(rgb_loc)
        df_ddelta = f(delta,bottom,grid=False,dx=1,dy=0)
        df_bottom = f(delta,bottom,grid=False,dx=0,dy=1)

        #df_ddelta += 0.1*2*(delta - init_delta)
        e = np.ones((1, M))
        D = scipy.sparse.spdiags(np.vstack((e, -2*e, e)), range(3), M-2, M)
        D1 = - np.eye(M-1,M=M,k=0) + np.eye(M-1,M=M,k=1)

        gradbottom = reg * ( D @ bottom  ).T @ (D )
        graddelta = reg * ( D @ delta ).T @ (D )
        gradbottom += ( D1 @ bottom - w ).T @ (D1 )
        #graddelta = ( D @ delta - init_delta ).T @ (D )

        grad = np.concatenate([df_bottom+gradbottom,df_ddelta+graddelta],axis=0)

        return grad
    

    #bounds_bottom = [(1,y.shape[1]) for i in range(0,M)]
    #bounds_delta = [(init_delta-2,init_delta+2) for i in range(0,M)]
    #bounds_top = [(1,y.shape[1]) for i in range(0,M)]

    #null_bounds = [(None,None) for i in range(0,2*M)]

    #rgb_bounds = [ (1,w/3),(w/3,2*w/3),(2*w/3,w-1)]

    #bounds = bounds_bottom + bounds_delta #+ bounds_top #
    #bounds = null_bounds + rgb_bounds
    #print("init delta {}".format(init_delta))

    #D = - np.eye(M-1,M=M,k=0) + np.eye(M-1,M=M,k=1)
    #Z = np.zeros((M-1,M))
    #print(D.shape)
    #print(Z.shape)
    #Atop = np.concatenate([D,Z],axis=1)
    #Abot = np.concatenate([Z,D],axis=1)
    #A = Atop # np.concatenate([Atop,Abot],axis=0)

    #lb = np.ones(M-1) * min_lenticule_width
    #ub = np.ones(M-1) * max_lenticule_width

    #lb = np.ones(M-1) * min_lenticule_width
    #ub = np.ones(M-1) * max_lenticule_width
    #print("lb {}".format(lb))
    #print("ub {}".format(ub))

    #Con = opt.LinearConstraint(A,lb,ub,keep_feasible=True)
    #print(A @ x0)
    res = opt.minimize(obj,x0,args=(f),method="L-BFGS-B",jac=jac) #,bounds=bounds) #,constraints = Con)
    
    assert(res.success)
    
    #print(res.success)
    #print(res.message)

    x = res.x

    bottom = initial_bottom.astype(float)
    bottom[1:] = x[0:M]
    top = initial_top.astype(float)
    top[1:]  = x[M:2*M] + x[0:M]
    #rgb = x[-3:]
    return bottom,top #,rgb


def colorize_image(x,x_p):
    
    device = x.device
    
    x_input = x
    
    x = x.squeeze().cpu().numpy()
    x_p = x_p.squeeze().cpu().numpy()
    # estimated lenticule width
    _, Pxx_den = sg.periodogram(np.sum(x_p,axis=0))
    w = x.shape[1] / ( 150 + np.argmax(Pxx_den[150:350]))

    delta_max = 20
    min_lenticule_width = int(np.floor(w) - 1)
    max_lenticule_width = int(np.ceil(w) + 1)

    u = build_score_matrix(x_p,delta_max)

    #lenticules_location_bottom,lenticules_location_top = initial_bottom,initial_top
    #lenticules_location_bottom,lenticules_location_top,_  = my_find_min(u,delta_max,min_lenticule_width,max_lenticule_width)
    lenticules_location_bottom,lenticules_location_top  = optimize_locations(u,delta_max,min_lenticule_width,max_lenticule_width,w,10.)

    z = reconstruct_boundaries(lenticules_location_bottom,lenticules_location_top,size=x.shape,lenticule_min = min_lenticule_width,lenticule_max = max_lenticule_width)

    max_distance = max(np.max(np.diff(lenticules_location_top)),np.max(np.diff(lenticules_location_bottom)))

    z = torch.from_numpy(z).unsqueeze(0).unsqueeze(0).float().to(device)
    x = x_input.unsqueeze(0)
    colorize = ColorRestoration(max_distance).to(device)

    out_dict = colorize(x,z)
    
    #y = out_dict["y"].detach().cpu().squeeze()
    #x = x.detach().cpu().squeeze()
    #z = z.detach().cpu().squeeze()

    return out_dict,z


def nearest_interp(mosaic,order):
    
    _,H,W = mosaic.shape
    #print(mosaic.shape)
    y = np.zeros_like(mosaic)
    col = np.array(range(0,W))
    
    for color in range(0,3):
        ds_color = mosaic[color,:,color::3]
    
        #f_color = scipy.interpolate.RectBivariateSpline(np.arange(0,ds_color.shape[0]),color +  3*np.arange(0,ds_color.shape[1]),ds_color,kx=order,ky=order)
    
        for i in range(0,H):
            
            f_color = scipy.interpolate.interp1d(color +  3*np.arange(0,ds_color.shape[1]),ds_color[i,:],bounds_error=False,kind=order) #"nearest")
            
            val = f_color(col)
            
            #print(col.shape)
        
            y[color,i,:] = val #f_color(row,col,grid=False)
            
    return y