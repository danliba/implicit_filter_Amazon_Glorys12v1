import xarray as xr, numpy as np
D='/work/bk1450/b383184/Amazon/Mercator/'
h=xr.open_dataset(D+'data/Hgr_cmesh.nc').squeeze(); z=xr.open_dataset(D+'data/Zgr_cmesh2.nc').squeeze()
mb=z.mbathy.values.astype(int); nz=50
k=np.arange(nz)[:,None,None]
tmask=(k<mb[None]).astype(bool)
e3t=np.where(k<mb[None]-1, z.e3t_0.values[:,None,None], z.e3t_ps.values[None]); e3t=np.where(tmask,e3t,0.)
u=xr.open_dataset(D+'implicit_data/U_1993-01c.nc').vozocrtx.isel(time_counter=0).values.astype(float)
v=xr.open_dataset(D+'implicit_data/V_1993-01c.nc').vomecrty.isel(time_counter=0).values.astype(float)
w=xr.open_dataset(D+'data/variables/W_1993-01.nc').vovecrtz.isel(time_counter=0).values.astype(float)
umask=np.zeros_like(tmask); umask[:,:,:-1]=tmask[:,:,:-1]&tmask[:,:,1:]
vmask=np.zeros_like(tmask); vmask[:,:-1,:]=tmask[:,:-1,:]&tmask[:,1:,:]
for lev in [0,10,30,45]:
    un=~np.isnan(u[lev]); vn=~np.isnan(v[lev]); wn=~np.isnan(w[lev])
    print(lev,'U wet vs umask mismatch', (un!=umask[lev]).sum(), 'U==0 on wet', (u[lev][umask[lev]]==0).mean().round(3),
          'V mismatch',(vn!=vmask[lev]).sum(), 'W nan vs tmask mismatch',(wn!=tmask[lev]).sum(), 'lastcol U wet', un[:,-1].sum(), 'lastrow V wet', vn[-1].sum())
# e3u, e3v NEMO partial steps
e3u=np.zeros_like(e3t); e3u[:,:,:-1]=np.minimum(e3t[:,:,:-1],e3t[:,:,1:])
e3v=np.zeros_like(e3t); e3v[:,:-1,:]=np.minimum(e3t[:,:-1,:],e3t[:,1:,:])
e1t=h.e1t.values;e2t=h.e2t.values;e2u=h.e2u.values;e1v=h.e1v.values
uu=np.nan_to_num(u)*umask; vv=np.nan_to_num(v)*vmask
fu=e2u*e3u*uu; fv=e1v*e3v*vv
div=np.zeros_like(e3t)
div[:,1:,1:]=(fu[:,1:,1:]-fu[:,1:,:-1]+fv[:,1:,1:]-fv[:,:-1,1:])
div/= (e1t*e2t)[None]   # = e3t*hdiv
wr=np.zeros((nz+1,)+mb.shape)
for kk in range(nz-1,-1,-1):
    wr[kk]=wr[kk+1]-div[kk]
wr=wr[:nz]
m=tmask.copy(); m[:,:3,:]=False; m[:,-3:,:]=False; m[:,:,:3]=False; m[:,:,-3:]=False
for lev in [0,1,5,15,30]:
    a=wr[lev][m[lev]]; b=np.nan_to_num(w[lev])[m[lev]]
    print('lev',lev,'corr',np.corrcoef(a,b)[0,1].round(4),'rms orig',np.sqrt((b**2).mean()),'rms diff',np.sqrt(((a-b)**2).mean()), 'regress', (a@b)/(b@b))
