import numpy as np
import scipy

def interpolateStripes(x_stripe,order):
    
    _,H,W = x_stripe.shape
    y = np.zeros_like(x_stripe)
    col = np.array(range(0,W))
    
    for color in range(0,3):
        ds_color = x_stripe[color,:,color::3]

        for i in range(0,H):
            f_color = scipy.interpolate.interp1d(color +  3*np.arange(0,ds_color.shape[1]),ds_color[i,:],bounds_error=False,kind=order) 
            val = f_color(col)
            y[color,i,:] = val
            
    return y