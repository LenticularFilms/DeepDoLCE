import numpy as np
import scipy
import scipy.optimize as opt
import scipy.signal as sg
import time
from joblib import Parallel, delayed

#from torch import float32
#from numba import jit


def build_score_matrix(x,delta_max):

    H,W = x.shape
    y = np.zeros((2*delta_max + 1,W))

    def p(delta):
        r = np.zeros((W,))
        for c in range(0,W):

            idx_row = np.arange(0,H)
            idx_col = np.round(c + idx_row * (delta/H)).astype(int)
            idx_row = idx_row.astype(int)
            
            mask = np.logical_and(idx_col >= 0,idx_col < W)
            idx_col = idx_col[mask]
            idx_row = idx_row[mask]

            val = x[idx_row,idx_col]
            val = np.mean(val)
            #y[delta+delta_max,c] = val
            r[c] = val
        return r

    #for delta in range(-delta_max,delta_max+1):
    
    y = np.array(Parallel(n_jobs=8)(delayed(p)(delta) for delta in range(-delta_max,delta_max+1)))

    return y


def my_find_min(y,delta_max,min_lenticule_width,max_lenticule_width,w):

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


def optimize_locations(y,delta_max,min_lenticule_width,max_lenticule_width,w,lambda1,lambda2):

    t_0 = time.time()
    initial_bottom,initial_top,init_delta  = my_find_min(y,delta_max,min_lenticule_width,max_lenticule_width,w)
    M = len(initial_bottom) - 1
    #print("initialization top/bottom: {}".format(time.time()-t_0))    

    
    x0 = np.concatenate([initial_bottom[1:],initial_top[1:]],axis=0)
    f = scipy.interpolate.RectBivariateSpline(np.arange(0,y.shape[0])-delta_max, np.arange(0,y.shape[1]),y)

    def obj(x,f): #,init_delta):
        bottom = x[0:M]
        top = x[M:2*M]
        delta = top - bottom

        val = f(delta,bottom,grid=False) #,dy=1) #,dy=1)
        val_tot = np.sum(val)

        D = - np.eye(M-1,M=M,k=0) + np.eye(M-1,M=M,k=1)
        e = np.ones((1, M))
        H = scipy.sparse.spdiags(np.vstack((e, -2*e, e)), range(3), M-2, M)

        penalty1_bottom= ( D @ bottom - w ).T @ (D @ bottom - w)
        penalty1_top = ( D @ top - w ).T @ (D @ top - w)

        penalty2_bottom = ( H @ bottom ).T @ (H @ bottom )
        penalty2_top = ( H @ top  ).T @ (H @ top)


        return val_tot + lambda2 * (penalty2_bottom + penalty2_top) + lambda1 * (penalty1_top + penalty1_bottom)


    def jac(x,f): #,init_delta):
        bottom = x[0:M]
        top = x[M:2*M] #- x[0:M]
        delta = top - bottom

        #df_rgb_log = np.zeros_like(rgb_loc)
        df_delta = f(delta,bottom,grid=False,dx=1,dy=0)
        df_bottom = f(delta,bottom,grid=False,dx=0,dy=1)

        df_top = df_delta
        df_bottom = df_bottom - df_delta

        D = - np.eye(M-1,M=M,k=0) + np.eye(M-1,M=M,k=1)
        e = np.ones((1, M))
        H = scipy.sparse.spdiags(np.vstack((e, -2*e, e)), range(3), M-2, M)

        gradbottom = lambda2 * ( H @ bottom  ).T @ (H )
        gradtop = lambda2 * ( H @ top ).T @ (H )
        
        gradbottom += lambda1 * ( ( D @ bottom - w ).T @ (D ) )
        gradtop += lambda1 * ( D @ top - w ).T @ (D )

        grad = np.concatenate([df_bottom+gradbottom,df_top+gradtop],axis=0)

        return grad
    

    res = opt.minimize(obj,x0,args=(f),method="L-BFGS-B",jac=jac) #,bounds=bounds) #,constraints = Con)
    
    #assert(res.success)
    #print(res.success)
    #print(res.message)

    x = res.x
    
    bottom = initial_bottom.astype(float)
    bottom[1:] = x[0:M]
    top = initial_top.astype(float)
    top[1:]  = x[M:2*M]

    return bottom,top


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