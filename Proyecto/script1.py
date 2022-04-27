import numpy as np
from scipy import sparse
from Active_set import active_set

c=np.array([[-2],[-5]])
A_I=np.array([[-1,2],[1,2],[1,-2],[-1,0],[0,-1]])
W0=np.array([0,0,1,0,0])
b_I=np.array([[2],[6],[2],[0],[0]])
G=np.array([[2,0],[0,2]])
x0=np.array([[2],[0]])
print(active_set(G,c,np.array([]),np.array([]),A_I,b_I,W0,10,x0))