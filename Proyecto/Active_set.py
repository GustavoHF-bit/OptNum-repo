import numpy as np
import scipy.optimize as op
import scipy.linalg as LA

def null_space_method(G,c,A,b):
    S_A=np.shape(A)
    #print("Dimension",S_A)
    #print("A\n",A)
    if not A.size:
        q,r=np.eye(np.shape(A)[1]),[]
    else:
        q,r=LA.qr(A.T) #if np.linalg.qr(A.T)[0].size else np.eye(np.shape(A)[1]),[]
    #else:
        #q,r=np.eye(np.shape(A)[1]),[]
    #print("Q\n",q)
    #print("R\n",r)
    Y=q[:,0:S_A[0]]
    Z=q[:,S_A[0]:S_A[1]]
    #print("Y\n",Y)
    cp=LA.solve((A@Y),b) if (A@Y).size else np.array([])
    if (Y@cp).ndim>1:
        xp=Y@cp
    else:
        xp=Y@cp[:,np.newaxis]
     #if (Y@cp).size else np.zeros((np.shape(G)[0],1))
     #if (Y@cp).size else np.zeros((np.shape(G)[0],1))
    ch=LA.solve((Z.T@G@Z),(-Z.T@c)-(Z.T@(G@xp)))
    d=Z@ch
    #print("d\n",np.shape(xp))
    x=xp+d

    lamb=np.linalg.solve((Y.T@A.T),(-Y.T@(c+G@x)))
    return x,lamb


def active_set(G,c,A_Eq,b_Eq,A_Iq,b_Iq,W0,itmax,x0):
    S_Eq=np.shape(A_Eq)
    S_Iq=np.shape(A_Iq)

    W_k_Iq=np.zeros(S_Iq[0])

    for i in range(1,S_Eq[0]+1):
        if W0[0][i-1]==1:
           W_k_Iq[i-1]==i
    #print("W_k_Iq\n",W_k_Iq)
    #print("J\n",A_Iq[W0.reshape(A_Iq.shape[0])==0,:])
    if x0.size>0:
        x_k=x0.copy()
    else:
        x_k=(op.linprog(np.zeros((S_Iq[1],1)),A_Iq[W0.reshape(A_Iq.shape[0])==0,:],b_Iq[W0.reshape(b_Iq.shape[0])==0,:],np.vstack((A_Eq,A_Iq[W0.reshape(A_Iq.shape[0])==1,:])) if A_Eq.size else A_Iq[W0.reshape(A_Iq.shape[0])==1,:]
        ,np.vstack((b_Eq,b_Iq[W0.reshape(A_Iq.shape[0])==1,:])) if b_Eq.size else b_Iq[W0.reshape(A_Iq.shape[0])==1,:]).x)[:,np.newaxis]
    #print('xk\n',x_k)
    for i in range(itmax):
        #print('xk\n',x_k)
        W_k_Iq_C=np.zeros(S_Iq[0])
        for j in range(1,S_Iq[0]+1):
            if W_k_Iq[j-1]==0:
                W_k_Iq_C[j-1]=j
            else:
                W_k_Iq_C[j-1]=0
        #print("W_k_Iq_C\n",W_k_Iq_C)
        n_wk=np.shape(np.nonzero(W_k_Iq!=0)[0])[0]
        #print("n_wk",n_wk)
        g_k=G@x_k+c
        #print('gk\n',g_k)
        #print("A_eq",A_Eq.size)
        if A_Eq.size>0: 
            A_k=np.vstack((A_Eq,A_Iq[np.nonzero(W_k_Iq!=0)[0],:]))
        else: 
            A_k=A_Iq[np.nonzero(W_k_Iq!=0)[0],:] #if (A_Eq.size)>0 else np.zeros((S_Eq[0]+n_wk, S_Iq[1]))
        #print("A_k",A_k)
        #b_k=np.vstack(b_Eq,b_Iq[np.nonzero(W_k_Iq!=0)[0],:]) )
        p_k,lam_k=null_space_method(G,g_k,A_k,np.zeros((n_wk+S_Eq[0],1)))
        #print("pk\n",np.shape(p_k))
        #print("lamk\n",lam_k)

        if LA.norm(p_k,ord=np.inf) <= 10e-9:
            print("lam_k",lam_k)
            if np.any(0<=lam_k) and not (np.shape(lam_k)[0]==0 or np.shape(lam_k)[1]==0) :
                break
            else:
                for j in range(S_Iq[0]):
                    rest_out=1
                    if lam_k[j+S_Eq[0]]<lam_k[rest_out+S_Eq[0]]:
                        rest_out=j
                W_k_Iq[rest_out]=0
                print("Rama 2\n")
                print("Restricción que está fuera: ",(rest_out+S_Eq[0]),"\n")
                print("Valor del multiplicador correspondiente: ",lam_k(rest_out+S_Eq[0]) ,"\n\n")
        else:
            index=np.nonzero(W_k_Iq_C!=0)[0]
            #print("index\n",index)
            S_In=np.shape(index)
            alphas=np.zeros((1,S_In[0]))
            for j in range(S_In[0]):
                if (A_Iq[index[j],:]@p_k)<0:
                    alphas[0][j]=2
                else:
                    alphas[0][j]=((b_Iq[index[j]])-(A_Iq[index[j],:]@x_k))/(A_Iq[index[j],:]@p_k)
            alpha_k=np.append(alphas[0],1).min()
            #print("alphak\n",alpha_k)
            rest_in=np.argmin(np.append(alphas[0],1),axis=None)
            #print('rest_in\n',rest_in)
            #print(alphas)
            if rest_in<S_In[0]:
                 W_k_Iq[index[rest_in]]=index[rest_in]+1
            #print("W_k_Iq_2\n",W_k_Iq)
            x_k=x_k+alpha_k*p_k
            print("Rama 1 \n")
            print("Longitud de pk: ",np.linalg.norm(p_k,ord=np.inf,axis=0),"\n")
            print("q(x_{k+1}):",(0.5*x_k.T@G@x_k+c.T@x_k)[0][0])
            print("Valor de alpha: ",alpha_k,"\n")
            if rest_in < S_In[0]:
                print("Restricción que entra: ",(index[rest_in]+S_Eq[0])+1,"\n")
            print("\n")
    Wk_I_Iq=np.nonzero(W_k_Iq!=0)[0]
    M=0.5*x_k.T@G@x_k+c.T@x_k
    return x_k,i,W_k_Iq,M
            



