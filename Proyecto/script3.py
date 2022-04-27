import numpy as np
from AW_loadProblem import loadProblem
from Active_set import active_set
H = loadProblem('lp_afiro.mat')
print('type(A) : ', type(H['AE']))
print(H['bE'].shape)
print(H['c'].shape)
A_E=H['AE']
m,n=A_E.shape
b_E=H['bE'][:,np.newaxis]
c=H['c'][:,np.newaxis]
G=np.eye(n)
A_I=-1*np.eye(n)
b_I=np.zeros((n,1))
W0=np.zeros((1,n))
#print(W0[1])
#np.random.shuffle(W0)
#print(np.shape(W0))
#W0=np.hstack((np.zeros((1,30)),W0[:,np.newaxis].T))
print(active_set(G,c,A_E,b_E,A_I,b_I,W0,100,np.array([])))