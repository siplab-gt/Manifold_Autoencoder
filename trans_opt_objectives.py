from __future__ import division
import numpy as np
import scipy as sp
from scipy.optimize import minimize 

from utils import *


def transOptObj_c(c,Psi,x0,x1,zeta):
    """
    Define forward pass for transport operator objective with regularizer on coefficients
    
    Inputs:
        - c:    Vector of transpor toperator coefficients [M]
        - Psi:  Transport operator dictionarys [N^2 x M]
        - x0:   Starting point for transport operator path [N]
        - x1:   Ending point for transport operator path [N]
        - zeta: Weight on the l1 coefficient regularizer
        
    Outputs:
        - objFun: Computed transport operator objective
    """
    N = np.int(np.sqrt(Psi.shape[0]))
    coeff_use = np.expand_dims(c,axis=1)
    x0_use = np.expand_dims(x0,axis=1)
    A = np.reshape(np.dot(Psi,coeff_use),(N,N),order='F')
    try:
        T = np.real(sp.linalg.expm(A))
    except:
        T = np.eye(N)
    x1_est= np.dot(T,x0_use)[:,0]
    objFun = 0.5*np.linalg.norm(x1-x1_est)**2 + zeta*np.sum(np.abs(c))
    
    return objFun

def transOptDerv_c(c,Psi,x0,x1,zeta):
    """
    Compute the gradient for the transport operator objective with regularizer on coefficients
    
    Inputs:
        - c:    Vector of transpor toperator coefficients [M]
        - Psi:  Transport operator dictionarys [N^2 x M]
        - x0:   Starting point for transport operator path [N]
        - x1:   Ending point for transport operator path [N]
        - zeta: Weight on the l1 coefficient regularizer
        
    Outputs:
        - c_grad: Gradient of the transport operator objective with repsect to the coefficients
    """
    N = np.int(np.sqrt(Psi.shape[0]))
    coeff_use = np.expand_dims(c,axis=1)
    M = coeff_use.shape[0]
    x0_use = np.expand_dims(x0,axis=1)
    x1_use = np.expand_dims(x1,axis=1)
    A = np.reshape(np.dot(Psi,coeff_use),(N,N),order='F')
    try:
        T = np.real(sp.linalg.expm(A))
        
        eig_out = np.linalg.eig(A)
        U = eig_out[1]
        D = eig_out[0]
        V = np.linalg.inv(U)
        V = V.T
    
        innerVal = np.dot(-x1_use,x0_use.T) + np.dot(T,np.dot(x0_use,x0_use.T))
        P = np.dot(np.dot(U.T,innerVal),V)
        
        F_mat = np.zeros((D.shape[0],D.shape[0]),dtype=np.complex128)
        for alpha in range(0,D.shape[0]):
            for beta in range(0,D.shape[0]):
                if D[alpha] == D[beta]:
                    F_mat[alpha,beta] = np.exp(D[alpha])
                else:
                    F_mat[alpha,beta] = (np.exp(D[beta])-np.exp(D[alpha]))/(D[beta]-D[alpha])
        
        fp = np.multiply(F_mat,P)
        Q1 = np.dot(V,fp)
        Q = np.dot(Q1,U.T)
        c_grad = np.real(np.dot(np.reshape(Q,-1,order='F'),Psi) + zeta*np.sign(c))
    except:
        c_grad = np.zeros((M))
        print("Failed to generate T")
    return c_grad

def infer_transOpt_coeff(x0,x1,Psi,zeta,randMin,randMax):
    """
    Infer the transport operator coefficients
    
    Inputs:
        - x0:       Starting point for transport operator path [N]
        - x1:       Ending point for transport operator path [N]
        - Psi:      Transport operator dictionarys [N^2 x M]
        - zeta:     Weight on the l1 coefficient regularizer
        - randMin:  Minimium value for the uniform distribution used to intialize coefficeints
        - randMax:  Maximum value for the uniform distribition used to initializer the coefficeints
        
    Outputs:
        - c_est:    Final inferred coefficients
    """
    M = Psi.shape[1]
    c0 = np.random.uniform(randMin,randMax,M)
    opt_out = minimize(transOptObj_c,c0,args=(Psi,x0,x1,zeta),method = 'CG',jac=transOptDerv_c,options={'maxiter':100,'disp':False},tol = 10^-10)
    c_est = opt_out['x']

    return c_est

