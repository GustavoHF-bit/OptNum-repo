import numpy as np
from scipy import sparse
from Active_set import active_set
def getKleeMinty(n):
    G=1e-4*(sparse.eye(n).toarray())
    c=-np.ones((n,1))

    A=np.tril(2*np.ones((n,n))-sparse.eye(n).toarray())

    Ls=[(2**i)-1 for i in range(1,n+1)]
    b=np.array(Ls,ndmin=2).T
    A=np.vstack((A,-sparse.eye(n).toarray()))
    b=np.vstack((b,np.zeros((n,1))))
    return G,c,A,b
#print(np.shape(getKleeMinty(20)[2]))
k=8
W0=np.array([0]*(10-k)+[1]*k)
np.random.shuffle(W0)
#print(np.shape(W0))
W0=np.hstack((np.zeros((1,30)),W0[:,np.newaxis].T))
print(W0)
W0=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1]) # el tomado
G, c, A_I, b_I = getKleeMinty(20)
print(active_set(G,c,np.array([]),np.array([]),A_I,b_I,W0,1000,np.array([])))