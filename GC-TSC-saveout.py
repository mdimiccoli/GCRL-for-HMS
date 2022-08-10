#!/usr/bin/env python3
# coding: utf-8

import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "4"

import sys
import time
import torch
import torch.optim as optim
import torch.nn.functional as FTR
from torch.autograd import Variable
import pickle
import scipy as sp
import scipy.io as spio
from scipy import spatial
import numpy as np
from sklearn.cluster import KMeans, spectral_clustering
from skimage.future import graph
import matplotlib.pyplot as plt
import os
from scipy import signal
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score
#get_ipython().run_line_magic('matplotlib', 'inline')
from spectral_fc import spectral_clustering_fully_connected
import accuracy
from accuracy import *

device = torch.device("cpu")

def compacc(idx1,idx0):
    uids = np.unique(idx1)
    idx = np.copy(idx1)
    for i in range(uids.size):
        uid = uids[i]
        inds = np.absolute(idx1-uid)<0.1
        vids = idx0[inds]
        uvids = np.unique(vids)
        vf = 0
        for j in range(uvids.size):
            vfj = np.sum(np.absolute(vids-uvids[j])<0.1)
            if vfj > vf:
                vid = uvids[j]
                vf = vfj
        idx[inds] = vid
    acc = np.sum(np.absolute(idx-idx0)<0.1)/idx0.size
    return acc

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

def compacc_sk(idx1,idx0):
    acc=adjusted_rand_score(idx0,idx1)
    nmi=normalized_mutual_info_score(idx0,idx1)
    return acc, nmi

def norm11(X):
    for i in range(X.shape[1]):
        X[:,i]=X[:,i]-X[:,i].min()
        mm=X[:,i].max()
        if mm != 0:
            X[:,i]=X[:,i]/mm
    # find frame of all zeros and replace it with average frame values
    sx=(torch.sum(X,1) == 0).nonzero()
    if sx.size(0):
        X[sx[0],:]=torch.mean(X,0)
    return X

def norm11_select(X,thr):
    X=X/torch.std(X)
    for i in range(X.shape[1]):
        if torch.std(X[:,i])>thr:
            X[:,i]=X[:,i]-X[:,i].min()
            mm=X[:,i].max()
            if mm != 0:
                X[:,i]=X[:,i]/mm
        else:
            X[:,i]=0
    # remove all zero features (=thresholded)
    sx=(torch.sum(X,0) != 0).nonzero()
    X=X[:,sx]
    # find frame of all zeros and replace it with average frame values
    sx=(torch.sum(X,1) == 0).nonzero()
    if sx.size(0):
        X[sx[0],:]=torch.mean(X,0)
    X=X[:,:,0]
    return X

def similaridad(X,param):
    #param = 0.01  #AÑADIDO
    dim=X.size()[1]
    Z = torch.mm(X,X.t())
    djj = torch.sqrt(torch.diag(Z))*torch.ones(1,N).double().t().to(device)+1e-16
    Z = torch.div(1 - torch.div(Z,torch.mul(djj,djj.t())) , dim)
    G = torch.exp(torch.mul(Z,-1/param))
    return G

def sim_norm(X,param,cenorm=0):
    N = X.size()[0]
    dim=X.size()[1]
    Z = torch.mm(X,X.t())
    djj = torch.sqrt(torch.diag(Z))*torch.ones(1,N).double().t().to(device)+1e-16
    Z = torch.div(1 - torch.div(Z,torch.mul(djj,djj.t())) , dim)
    G = torch.exp(torch.mul(Z,-1/param))
    if cenorm==1: # normalize full matrix
        G=G / torch.sum(G)
    else: # row-wise cross-entropy
        z = torch.sum(G,0)
        G = torch.div(G,z.repeat(N,1))
    return G

def loss_fun(G,W):
    N=G.size()[0]
    # cross-entropy loss 
    w=W.contiguous().view(N*N,-1)
    g=G.contiguous().view(N*N,-1)
    L=-torch.sum(torch.mul(w,torch.log(g)))
    L=L+torch.sum(torch.mul(w,torch.log(w)))
    return L

def loss_fun_full(X,X0,LAM,W,param,CENORM,lg,rho):
    L0=loss_fun(sim_norm(X,param,CENORM),W)
    L1= torch.sum(torch.mul(LAM,X0-X)) +rho*torch.norm(X-X0,p='fro').cpu().pow(2)
    L = lg*L0+L1
#    print(L0,L1)
    return L

def sim_grad_descent_autograd(X,W,param,niter,X0,LAM,lg,rho,CENORM=0):
    # performs gradient descent with adaptive stepsize
    eta0=1e-6
    N=X.size()[0]
    dim=X.size()[1]
    
    # iteration 1: no adaptive step
    #print(sim_norm(X,param,CENORM)-W)
    #loss=loss_fun(sim_norm(X,param,CENORM),W)
    #loss.backward()
    #dL=lg*X.grad.detach().clone() #+ LAM + rho*(X-X0)
    loss=loss_fun_full(X,X0,LAM,W,param,CENORM,lg,rho)
    loss.backward()
    dL=X.grad.detach().clone()
    X.grad.data.zero_()
    x1=X.permute(1,0)
    g1=dL.permute(1,0)
    x1=x1.contiguous().view(-1,N*dim)
    g1=g1.contiguous().view(-1,N*dim)
    eta=eta0
    X=X-eta*dL
    X=X.detach().clone()
    X.requires_grad=True
    
    #print(eta,loss.item(),torch.norm(dL,p='fro'))
    
    # step 2-niter:
    for it in range(niter-1):
        #loss=loss_fun(simz_fun(distx_fun(X),param),W)
        #loss=loss_fun(sim_norm(X,param,CENORM),W)
        #loss.backward()
        #dL=lg*X.grad.detach().clone() #+ LAM + rho*(X-X0)
        loss=loss_fun_full(X,X0,LAM,W,param,CENORM,lg,rho)
        loss.backward()
        dL=X.grad.detach().clone()
        X.grad.data.zero_()
        x2=x1
        g2=g1
        x2.detach().clone()
        g2.detach().clone()
        x1=X.permute(1,0)
        g1=dL.permute(1,0)
        x1=x1.contiguous().view(-1,N*dim)
        g1=g1.contiguous().view(-1,N*dim)
        dx=x1-x2
        dg=g1-g2
        eta=torch.mm(dg,dx.transpose(1,0)) / torch.mm(dg,dg.transpose(1,0))
        
        if eta != eta: # eta is nan. this actually means that the gradient is zero: STOP
            #print(g1)
            eta=torch.tensor(1e-16)
    
        X=X - eta*dL
        X=X.detach().clone()
        X.requires_grad=True

#print(eta,loss.item(),torch.norm(dL,p='fro'))

    # return new X and loss
    loss=loss_fun(sim_norm(X,param,CENORM),W)
    return X, lg*loss.item()



##################################################
##################################################


# parameters for similarities
param1=par1val
#param2=par2val
param2=param1

thr=0
print(thr)

# parameters for dynamic graph embedding
niter_g=25  # gradient descent steps for update of \tilde X

GCTSC=1 # do graph regularization (switch for ablation study)

N_iter_G=25 # number of iterations

SAVE_OUT=0 # save estimated representations at each iteration

# NORMALIZE FULL MATRIX FOR CE LOSS
CENORM=0

start_time = time.time()

torch.cuda.empty_cache()

####VARIABLES
TSC_rho = 0.025 #0.1
DGE_rho = 0.025 #0.1
lambdaG0 = LGVAL #0.5

lambdaG=lambdaG0

k = 7
tol = 1e-4
dsize = 80
lambda0 = lambda0value
lambda1 = lambda1value
lambda2 = lambda2value
alpha = TSC_rho
beta = TSC_rho
niter_tsc=N_TSC_VAL


#####EXPERIMENT#######

DATADIR = './Weiz_dataset/'

SAVEDIR = "./GC-TSC-Results/"
print(DATADIR)
print(param1,param2)

import random
SEED=SEED_VAL
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED']=str(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic=True

for i in range(9):
    item = i + 1
    name   = "wei_person_"+str(item)
    #fname  = DATADIR+name+"_cnn.npy"
    fname  = DATADIR+name+".npy"
    ftruth = DATADIR+name+"_label.npy"
    snamebase  = SAVEDIR+name
    sname  = snamebase+".npy"

    print(fname)

    # Load data 
    X_data  = np.load(fname)
    GT_data = np.load(ftruth)

    # Transpose X that we have it in the format number_of_elements x num_features
    X1 = X_data.T
    N = X1.shape[0]
    M = X1.shape[1]
    
    if CENORM==1: # compensate smaller loss value when matrix-wise CrossEntropy normalization is used
        lambdaG=N*lambdaG0

    # This is the ground truth
    GT = GT_data[0,:]

    IMG_PATH = SAVEDIR+name+"/"
    try:
        os.makedirs(IMG_PATH,exist_ok=True)
    except OSError:
        print("Error creating directory")
        
    start_time1 = time.time()

    PCAstr = ''
    GST='_auto'
 
    # - initial similarity of cleaned data
    dataiter = torch.from_numpy(X1).to(device)
    #if GCTSC==1:
    #    dataiter = norm11(dataiter)
    #W1 = sim_norm(dataiter,param1,CENORM)


    #######################################################
    #--- PART 2: EMBEDDING: NONE.

    param2=param1

    #######################################################
    #--- PART 3: DGE LOOP
    
    # - normalize and initialize
    if GCTSC==1:
        if thr==0:
            dataiter = norm11(dataiter).double().to(device)
        else:
            dataiter = norm11_select(dataiter,thr).double().to(device)
            M = dataiter.size()[1]
        print(M)
    W1 = sim_norm(dataiter,param1,CENORM)

    if SAVE_OUT==1:
        sname  = snamebase+"_Xembed.npy"
        np.save(sname,dataiter.cpu().numpy())
    
    #--- initial data similarity
    W2 = sim_norm(dataiter,param2,CENORM)

    # initialize D,Z,V, Lagrangians
    D = torch.rand(M, dsize).double().to(device)  #AÑADIDO
    Z = torch.rand(dsize, N).double().to(device) #AÑADIDO
    V = Z.clone().detach()
    Y1 = torch.zeros(M, dsize).double().to(device)
    Y2 = torch.zeros(dsize, N).double().to(device)
    Y3 = torch.zeros_like(dataiter).double().to(device)
    ##--Define the weight matrix--##
    k2 = (k-1)/2
    W = torch.zeros(N,N).double().to(device)
    for ii in range(N):
        W[ii,int(max(0,ii-k2)):int(min(N,ii+k2+1))] = 1
        W[ii,ii] = 0
    ##--Construct Laplacian matrix--##
    DD = torch.diag(torch.sum(W,0)).double().to(device)
    L = DD - W
    I_nx = torch.eye(N).to(device)
    I_dsize = torch.eye(dsize).to(device)

    # initialize loss buffers
    LTSC = torch.zeros(N_iter_G)
    LL = torch.zeros(N_iter_G)
    LL2 = torch.zeros(N_iter_G)
    LG = torch.zeros(N_iter_G)
    L0 = torch.zeros(N_iter_G)
    L1 = torch.zeros(N_iter_G)
    L2 = torch.zeros(N_iter_G)
    LX = torch.zeros(N_iter_G)
    LNX = torch.zeros(N_iter_G)
    # INITIALIZE accuracy buffers
    personal_ariZ = np.zeros(N_iter_G) #
    personal_accZ = np.zeros(N_iter_G) #
    personal_nmiZ = np.zeros(N_iter_G) #
    personal_hrZ = np.zeros(N_iter_G) #
    personal_arikmZ = np.zeros(N_iter_G) #
    personal_acckmZ = np.zeros(N_iter_G) #
    personal_nmikmZ = np.zeros(N_iter_G) #
    personal_hrkmZ = np.zeros(N_iter_G) #

    personal_ariX = np.zeros(N_iter_G) #
    personal_accX = np.zeros(N_iter_G) #
    personal_nmiX = np.zeros(N_iter_G) #
    personal_hrX = np.zeros(N_iter_G) #
    personal_arikmX = np.zeros(N_iter_G) #
    personal_acckmX = np.zeros(N_iter_G) #
    personal_nmikmX = np.zeros(N_iter_G) #
    personal_hrkmX = np.zeros(N_iter_G) #

    dataiterinit=dataiter.clone().detach() # for monitoring purpose only (X - \tilde X)

    # --- MAIN LOOP
    for nDGE in range(N_iter_G):
#        print(nDGE)

        #--- ADMM - Minimizing V,U,Z,D
        dataiter = dataiter.t()
        #D,Z,U,V, _ = TSC_ADMM_torch(dataiter,para,D,Z) #AÑADIDO D,Z
        
        ########################################
        ##### TSC ##############################
        ########################################
        X = dataiter.clone().detach()
        D = D.clone().detach() #torch.rand(d, dsize).double().to(device)
        Z = Z.clone().detach() #torch.rand(dsize, n_x).double().to(device)
        f_old = torch.norm(X- torch.mm(D,Z),p='fro')
        for nTSC in range(niter_tsc):
            ##--Update U--##
            b = 2*lambda0*torch.mm(X,V.t()) - Y1 + alpha*D
            a = 2*lambda0*torch.mm(V,V.t()) + alpha*I_dsize
            U, _ = torch.solve(b.t(),a.t())
            U = U.t()
            ##--Update V--##
            V = torch.tensor(sp.linalg.solve_sylvester((2*lambda0*torch.mm(U.t(),U)+(lambda1+beta)*I_dsize).cpu().numpy(),lambda2*L.cpu().numpy(), (2*lambda0*torch.mm(U.t(),X) - Y2 + beta*Z).cpu().numpy())).to(device)
            ##--Update D--##
            D = U + Y1/alpha
            D[D<0] = 0
            for kk in range(D.shape[1]):
                D[:,kk] = torch.div(D[:,kk],torch.norm(D[:,kk])+1e-16)
            ##--Update Z--##
            Z = V + Y2/beta
            Z[Z<0] = 0
            f_new = torch.norm(X-torch.mm(D,Z),p='fro')
            err = abs(f_new - f_old)/max(1,abs(f_old))
            #Y1 = Y1 + rho*alpha*(U-D)
            #Y2 = Y2 + rho*beta*(V-Z)
            Y1 = Y1 + alpha*(U-D)
            Y2 = Y2 + beta*(V-Z)

        ##--Update X--##
        if GCTSC==1:
            #--- Update for X (= dual for \tilde X)
            Y3 = Y3.t()
            dataiter0 = (2*lambda0*torch.mm(U,V)-Y3+DGE_rho*dataiter)/(2*lambda0+DGE_rho)
            dataiter0 = dataiter0.t()
            dataiter = dataiter.t()
            dataiter0 = norm11(dataiter0)
            Y3 = Y3.t()
            #--- Graph embedding update (= \tilde X)
            dataiter.requires_grad=True # required for AUTOGRAD
#            grad, eta = sim_grad_descent_autograd(dataiter,W1,param2,niter_g,CENORM)
#            dataiter = lambdaG*grad.clone().detach()+Y3+DGE_rho*(dataiter.detach()-dataiter0)
            # CORRECTED UPDATE:
#            print(torch.norm(dataiter,p='fro').cpu().pow(2))
#            print(torch.norm(dataiter0,p='fro').cpu().pow(2))
            dataiter, eta = sim_grad_descent_autograd(dataiter,W1,param2,niter_g,dataiter0,Y3,lambdaG,DGE_rho,CENORM)
            dataiter.requires_grad=False
            dataiter = norm11(dataiter)
            
            #LG
            LG[nDGE] = eta
            #--- Update Lagrange multiplier
            Y3 = Y3 + DGE_rho*(dataiter0-dataiter)

            dataiter = norm11(dataiter)
            
            if SAVE_OUT==1:
                sname  = snamebase+"_X_it"+str(nDGE)+".npy"
                np.save(sname,dataiter.cpu().numpy())
                sname  = snamebase+"_Z_it"+str(nDGE)+".npy"
                np.save(sname,Z.cpu().numpy())
        

        ##--Calculate LOSS--##
        L0[nDGE] = lambda0value*torch.norm(dataiter.t() - torch.mm(D,Z),p='fro').cpu().pow(2)
        L1[nDGE] = lambda1value*torch.norm(Z,p='fro').cpu().pow(2)
        L2[nDGE] = lambda2value*torch.trace(torch.mm(torch.mm(Z,L),Z.t())).cpu()
        if GCTSC==1:
            LL[nDGE] = L0[nDGE] + L1[nDGE] + L2[nDGE] + LG[nDGE]
            LL2[nDGE] = LTSC[nDGE] + LG[nDGE]
        else:
            LL[nDGE] = L0[nDGE] + L1[nDGE] + L2[nDGE]
            LL2[nDGE] = LTSC[nDGE]
        LX[nDGE] = torch.norm(dataiter - dataiterinit,p='fro').cpu().pow(2) #X - \tilde X
        LNX[nDGE] = torch.norm(dataiter,p='fro').cpu().pow(2) #norm \tilde X

        #print(LL[nDGE])

        #--- EVALUATE PERFORMANCE
        #Calculate accuracy in each iteration
        nClusters = np.unique(GT).size;
    
        #Similarity matrix from X
        W2_X = similaridad(dataiter,param2)
        vecNorm = torch.sum(torch.pow(dataiter.t(),2),dim=0)
        vecNorm = vecNorm.unsqueeze(0)
        W2_X2 = torch.div(torch.mm(dataiter,dataiter.t()),torch.mm(vecNorm.t(),vecNorm))
        W2_X = (W2_X+W2_X.T)/2.
        W2_X2 = (W2_X2+W2_X2.T)/2.
        #Similarity matrix from Z
        W2_Z2 = similaridad(Z.t(),param2)
        vecNorm = torch.sum(torch.pow(Z,2),dim=0)
        vecNorm = vecNorm.unsqueeze(0)
        W2_Z = torch.div(torch.mm(Z.t(),Z),torch.mm(vecNorm.t(),vecNorm))
        W2_Z = (W2_Z+W2_Z.T)/2.
        W2_Z2 = (W2_Z2+W2_Z2.T)/2.

        # NCUTS Z
        ncuts_labels_Z = spectral_clustering_fully_connected(W2_Z.cpu().numpy(), n_clusters=nClusters, n_init=50);
        ari_Z, nmi_Z, acc_Z, hr_Z = compacc_sk_h(ncuts_labels_Z,GT)
        personal_ariZ[nDGE] = ari_Z
        personal_nmiZ[nDGE] = nmi_Z
        personal_accZ[nDGE] = acc_Z
        personal_hrZ[nDGE] = hr_Z

        # NCUTS X
        ncuts_labels_X = spectral_clustering_fully_connected(W2_X.cpu().numpy(), n_clusters=nClusters, n_init=50);
        ari_X, nmi_X, acc_X, hr_X = compacc_sk_h(ncuts_labels_X,GT)
        personal_ariX[nDGE] = ari_X
        personal_nmiX[nDGE] = nmi_X
        personal_accX[nDGE] = acc_X
        personal_hrX[nDGE] = hr_X


        # KMEANS Z
        kmeans_labels_Z = KMeans(n_clusters=nClusters, random_state=0, n_init=50, max_iter=500).fit_predict(Z.t().cpu().numpy())
        ari_kmZ, nmi_kmZ, acc_kmZ, hr_kmZ = compacc_sk_h(kmeans_labels_Z,GT)
        personal_arikmZ[nDGE] = ari_kmZ
        personal_nmikmZ[nDGE] = nmi_kmZ
        personal_acckmZ[nDGE] = acc_kmZ
        personal_hrkmZ[nDGE] = hr_kmZ
        # KMEANS X
        kmeans_labels_X = KMeans(n_clusters=nClusters, random_state=0, n_init=50, max_iter=500).fit_predict(dataiter.cpu().numpy())
        ari_kmX, nmi_kmX, acc_kmX, hr_kmX = compacc_sk_h(kmeans_labels_X,GT)
        personal_arikmX[nDGE] = ari_kmX
        personal_nmikmX[nDGE] = nmi_kmX
        personal_acckmX[nDGE] = acc_kmX
        personal_hrkmX[nDGE] = hr_kmX
    
    F = open(IMG_PATH+"results.txt","w")
    F.write("GT\n")
    for x in GT:
        F.write(str(x)+" ")
    F.write("\n\n")
    
    F.write("Z\n")
    F.write("Labels\n")
    for x in ncuts_labels_Z:
        F.write(str(x)+" ")
    F.write("\n")
    F.write("Accuracy\n")
    F.write(str(acc_Z)+"\n\n")
    
    F.write("X\n")
    F.write("Labels\n")
    for x in ncuts_labels_X:
        F.write(str(x)+" ")
    F.write("\n")
    F.write("Accuracy\n")
    F.write(str(acc_X)+"\n\n")

    F.write("Kmeans dataiter\n")
    F.write("Labels\n")
    for x in kmeans_labels_X:
        F.write(str(x)+" ")
    F.write("\n")
    F.write("Accuracy\n")
    F.write(str(acc_kmX)+"\n")
    
    F.write("Kmeans Z\n")
    F.write("Labels\n")
    for x in kmeans_labels_Z:
        F.write(str(x)+" ")
    F.write("\n")
    F.write("Accuracy\n")
    F.write(str(acc_kmZ)+"\n")
    F.close()

    ##############################################################
    #Write the loss/iteration
    F = open(IMG_PATH+"LX.txt","w")
    for x in LX.cpu().numpy():
        F.write(str(x)+"\n")
    F.write("\n")
    F.close()

    F = open(IMG_PATH+"LnormX.txt","w")
    for x in LNX.cpu().numpy():
        F.write(str(x)+"\n")
    F.write("\n")
    F.close()

    F = open(IMG_PATH+"LOSS.txt","w")
    for x in LL.cpu().numpy():
        F.write(str(x)+"\n")
    F.write("\n")
    F.close()

    F = open(IMG_PATH+"LOSS_L0.txt","w")
    for x in L0.cpu().numpy():
        F.write(str(x)+"\n")
    F.write("\n")
    F.close()

    F = open(IMG_PATH+"LOSS_L1.txt","w")
    for x in L1.cpu().numpy():
        F.write(str(x)+"\n")
    F.write("\n")
    F.close()

    F = open(IMG_PATH+"LOSS_L2.txt","w")
    for x in L2.cpu().numpy():
        F.write(str(x)+"\n")
    F.write("\n")
    F.close()

    F = open(IMG_PATH+"LOSS_LG.txt","w")
    for x in LG.cpu().numpy():
        F.write(str(x)+"\n")
    F.write("\n")
    F.close()

    ##############################################################
    #Write the accuracy/iteration

    F = open(IMG_PATH+"results_ari_Z.txt","w")
    for x in personal_ariZ:
        F.write(str(x)+"\n")
    F.write("\n")
    F.close()
    F = open(IMG_PATH+"results_acc_Z.txt","w")
    for x in personal_accZ:
        F.write(str(x)+"\n")
    F.write("\n")
    F.close()
    F = open(IMG_PATH+"results_nmi_Z.txt","w")
    for x in personal_nmiZ:
        F.write(str(x)+"\n")
    F.write("\n")
    F.close()
    F = open(IMG_PATH+"results_hr_Z.txt","w")
    for x in personal_hrZ:
        F.write(str(x)+"\n")
    F.write("\n")
    F.close()

    F = open(IMG_PATH+"results_ari_X.txt","w")
    for x in personal_ariX:
        F.write(str(x)+"\n")
    F.write("\n")
    F.close()
    F = open(IMG_PATH+"results_acc_X.txt","w")
    for x in personal_accX:
        F.write(str(x)+"\n")
    F.write("\n")
    F.close()
    F = open(IMG_PATH+"results_nmi_X.txt","w")
    for x in personal_nmiX:
        F.write(str(x)+"\n")
    F.write("\n")
    F.close()
    F = open(IMG_PATH+"results_hr_X.txt","w")
    for x in personal_hrX:
        F.write(str(x)+"\n")
    F.write("\n")
    F.close()
    
    F = open(IMG_PATH+"results_ari_kmZ.txt","w")
    for x in personal_arikmZ:
        F.write(str(x)+"\n")
    F.write("\n")
    F.close()
    F = open(IMG_PATH+"results_acc_kmZ.txt","w")
    for x in personal_acckmZ:
        F.write(str(x)+"\n")
    F.write("\n")
    F.close()
    F = open(IMG_PATH+"results_nmi_kmZ.txt","w")
    for x in personal_nmikmZ:
        F.write(str(x)+"\n")
    F.write("\n")
    F.close()
    F = open(IMG_PATH+"results_hr_kmZ.txt","w")
    for x in personal_hrkmZ:
        F.write(str(x)+"\n")
    F.write("\n")
    F.close()

    F = open(IMG_PATH+"results_ari_kmX.txt","w")
    for x in personal_arikmX:
        F.write(str(x)+"\n")
    F.write("\n")
    F.close()
    F = open(IMG_PATH+"results_acc_kmX.txt","w")
    for x in personal_acckmX:
        F.write(str(x)+"\n")
    F.write("\n")
    F.close()
    F = open(IMG_PATH+"results_nmi_kmX.txt","w")
    for x in personal_nmikmX:
        F.write(str(x)+"\n")
    F.write("\n")
    F.close()
    F = open(IMG_PATH+"results_hr_kmX.txt","w")
    for x in personal_hrkmX:
        F.write(str(x)+"\n")
    F.write("\n")
    F.close()



    print("--- %s seconds" % (time.time() - start_time1))
    #######################################################

print("--- %s seconds --------" % (time.time() - start_time))

