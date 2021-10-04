import numpy as np
import pathlib

# http://vision.psych.umn.edu/users/kersten/kersten-lab/courses/Psy5036W2017/Lectures/17_PythonForVision/Demos/html/2b.Gabor.html
def genGabor(sz, omega, theta, func=np.cos, K=np.pi):
    radius = (int(sz[0]/2.0), int(sz[1]/2.0))
    [x, y] = np.meshgrid(range(-radius[0], radius[0]+1), range(-radius[1], radius[1]+1))

    x1 = x * np.cos(theta) + y * np.sin(theta)
    y1 = -x * np.sin(theta) + y * np.cos(theta)
    
    gauss = omega**2 / (4*np.pi * K**2) * np.exp(- omega**2 / (8*K**2) * ( 4 * x1**2 + y1**2))
#     myimshow(gauss)
    sinusoid = func(omega * x1) * np.exp(K**2 / 2)
#     myimshow(sinusoid)
    gabor = gauss * sinusoid
    return gabor

def make_set(n_samples, gabor_type, omegas=[.1,3], thetas=[0,np.pi], colors=[-1,1], size=32):
    # Makes a test or train set of Gabor filters, 32x32. 
    # The labels are theta, color if type==0. 
    # The labels are theta, omega if type==1.
    # Returns both of the above labels in a list if gabor_type=2.

    ims = []
    lbls = []
    if gabor_type==2:
        lbls = [[] for _ in range(2)]
    for n in range(n_samples):
        om = np.random.uniform(*omegas)
        th = np.random.uniform(*thetas)
        gfil = genGabor((size,size),om,th)[1:,1:] + np.random.uniform(*colors)
        
        ims.append(gfil)

        if gabor_type==0:
            lbls.append([th,np.mean(gfil)])
        elif gabor_type==1:
            lbls.append([th,om])
        else:
            lbls[0].append([th,np.mean(gfil)])
            lbls[1].append([th,om])

    if gabor_type!=2:
        ims,lbls = np.array(ims).reshape(-1,1,size,size), np.array(lbls)
    else:
        ims, lbls = np.array(ims).reshape(-1,1,size,size), [np.array(lbls[0]),np.array(lbls[1])]
    return ims,lbls


if __name__=='__main__':
    # Make gabor data folder
    path = pathlib.Path('./gabors/')
    path.mkdir(parents=True, exist_ok=True)

    # Make train sets
    print('Making train sets')
    ims,lbls = make_set(20000, 2)
    np.save('./gabors/gabor_tr_im.npy', ims)
    np.save('./gabors/gabor_tr_lbl.npy', lbls[0])
    np.save('./gabors/gabor2_tr_im.npy', ims)
    np.save('./gabors/gabor2_tr_lbl.npy', lbls[1])

    # Make test sets
    print('Making test sets')
    ims,lbls = make_set(10000, 2)
    np.save('./gabors/gabor_te_im.npy', ims)
    np.save('./gabors/gabor_te_lbl.npy', lbls[0])
    np.save('./gabors/gabor2_te_im.npy', ims)
    np.save('./gabors/gabor2_te_lbl.npy', lbls[1])

    print('Done.')
