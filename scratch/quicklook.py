import sys; sys.path.insert(0,'../analysis')
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import common as c
m=c.mesh(); js,is_=c.box_slices(-70,-30,-5,30)
fig,ax=plt.subplots(2,3,figsize=(16,9),constrained_layout=True)
for a,cfg in zip(ax.flat,['ORIG','W1.5','W2.0','W2.5','R2Ld','R3Ld']):
    u=c.read('U',cfg,t=10,k=0,j=js,i=is_); v=c.read('V',cfg,t=10,k=0,j=js,i=is_)
    ut,vt=c.uv_to_t(u,v); sp=np.hypot(ut,vt); sp[m.tmask[0][js,is_]==0]=np.nan
    pc=a.pcolormesh(m.glamt[js,is_],m.gphit[js,is_],sp,vmin=0,vmax=1.2,cmap='viridis'); a.set_title(cfg+' 1993-01-11 surface speed')
fig.colorbar(pc,ax=ax,shrink=0.6,label='m/s'); fig.savefig('quicklook.png',dpi=70)
