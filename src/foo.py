import numpy as np
from scipy import interpolate

# Create rectilinear grid in 4D
d1=[1000,3000,10000]
d2=np.arange(2,5)
d3=[-1,-0.5,0,0.3,0.5]
d4=[0,0.2,0.4]
a1,a2,a3,a4=np.meshgrid(d1,d2,d3,d4)

# Fill with values from arbitrary function
values=np.ma.array(a1.shape)
values=a1/1000.+a2**2+np.exp(a3)+a4
n1,n2,n3,n4=values.shape

# Mask off some points, mostly near grid edges
mask=(np.random.uniform(0,1,values.shape)*np.abs(10-values))>3
values[mask]=np.nan

print (values)

# Interpolate
#modelfn=interpolate.RegularGridInterpolator((d1,d2,d3,d4),values,method="linear",bounds_error=False,fill_value=None)
#params=a1,a2,a3,a4
