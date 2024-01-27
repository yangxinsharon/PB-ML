#### PBML Project, Math Department, Southern Methodist University
#### This script aims to compute several kinds of graph kernels and protein forces
#### version: use numba to speed up 

import sys
import math
import numpy as np
from scipy.spatial import cKDTree

import numba
from numba import jit, int8, prange
numba.config.NUMBA_DEFAULT_NUM_THREADS=8
from numba.typed import List

import time
import warnings
warnings.filterwarnings("ignore")

mat_idx = [0, 1, 2, 3, 4, \
           1, 5, 6, 7, 8, \
           2, 6, 9,10,11, \
           3, 7,10,12,13, \
           4, 8,11,13,14]
criterion = int8(18)


@jit(nopython=True, fastmath=True, parallel=True)
def _exp_Lor_fcn(tau,kappa,nu,atomN,atomT,atomx,atomy,atomz,vdWRa,charg,mat_idx):
    expfcn = [0.0]*15
    Lorfcn = [0.0]*15
    expqfcn = [0.0]*15
    Lorqfcn = [0.0]*15
    VDW = [0.0]*15
    CLB = [0.0]*15
    for i in range(atomN):
        # print(atomx[i], atomy[i], atomz[i])
        for j in range(i+1, atomN):
            idx = mat_idx[5*atomT[i]+atomT[j]]
            dist = math.sqrt((atomx[i]-atomx[j])**2 \
                       + (atomy[i]-atomy[j])**2 \
                       + (atomz[i]-atomz[j])**2)
            eta = tau*(vdWRa[i]+vdWRa[j])
            if eta == 0:
                continue
            temp = dist/eta
            _charge = charg[i]*charg[j]
            ratio =  (vdWRa[i] + vdWRa[j])**2/dist
            _vdw = np.power(ratio, 12) - 2.*np.power(ratio, 6)
            _clb = charg[i] * charg[j]/dist
            if temp < criterion:
                _expfcn = math.exp(-(temp)**kappa)
                _Lorfcn = 1./(1.+temp**nu)
                expfcn[idx] += _expfcn
                Lorfcn[idx] += _Lorfcn
                expqfcn[idx] += _charge*_expfcn
                Lorqfcn[idx] += _charge*_Lorfcn
                VDW[idx] += _vdw
                CLB[idx] += _clb
    return expfcn,Lorfcn,expqfcn,Lorqfcn,VDW,CLB


@jit(nopython=True, fastmath=True, parallel=True)
def _exp_fcn(tau,pwr,atomN,atomT,atomx,atomy,atomz,vdWRa,charg,mat_idx): # exponential function without charge
    expfcn = [0.0]*15
    expqfcn = [0.0]*15
    for i in range(atomN):
        for j in range(i, atomN):
            idx = mat_idx[5*atomT[i]+atomT[j]]
            dist = math.sqrt((atomx[i]-atomx[j])**2 \
                       + (atomy[i]-atomy[j])**2 \
                       + (atomz[i]-atomz[j])**2)
            eta = tau*(vdWRa[i]+vdWRa[j])
            temp = dist/eta
            _charge = charg[i]*charg[j]
            if temp < criterion:
                _expfcn = math.exp(-(temp)**pwr)
                expfcn[idx] += _expfcn
                expqfcn[idx] += _charge*_expfcn
    return expfcn,expqfcn



@jit(nopython=True, fastmath=True, parallel=True)
def _lor_fcn(tau,pwr,atomN,atomT,atomx,atomy,atomz,vdWRa,charg,mat_idx):
    Lorfcn = [0.0]*15
    Lorqfcn = [0.0]*15
    for i in range(atomN):
        for j in range(i, atomN):
            idx = mat_idx[5*atomT[i]+atomT[j]]
            dist = math.sqrt((atomx[i]-atomx[j])**2 \
                       + (atomy[i]-atomy[j])**2 \
                       + (atomz[i]-atomz[j])**2)
            eta = tau*(vdWRa[i]+vdWRa[j])
            temp = dist/eta
            _charge = charg[i]*charg[j]
            if temp < criterion:
                _Lorfcn = 1./(1.+temp**pwr)
                Lorfcn[idx] += _Lorfcn
                Lorqfcn[idx] += _charge*_Lorfcn 
    return Lorfcn,Lorqfcn



class protein():
    def __init__(self):
        self.atomN = 0 # total number of atoms
        self.atomT = []  # atom type_encoded (ONCSH - 01234)
        self.atomx = [] 
        self.atomy = [] 
        self.atomz = [] 
        self.vdWRa = [] # atom radius
        self.charg = [] # atom charge
        self.expfcn = []
        self.Lorfcn = []
        self.expqfcn = []
        self.Lorqfcn = []
        self.AtomPos = [] # position x,y,z for all atoms
        self.VDW = []
        self.CLB = []

        filename = 'pro.pqr'
        with open(filename) as fp:
            for line in fp:
                if line[0:4] == 'ATOM':
                    self.atomN += 1
                    # self.atomx.append(float(line[30:38]))
                    # self.atomy.append(float(line[38:46]))
                    # self.atomz.append(float(line[46:54]))
                    # self.charg.append(float(line[54:62]))
                    # self.vdWRa.append(float(line[62:70]))
                    # self.AtomPos.append([float(line[30:38]),float(line[38:46]), float(line[46:54])])
                    ## change for prep_bind data format ##
                    self.atomx.append(float(line.split()[5]))
                    self.atomy.append(float(line.split()[6]))
                    self.atomz.append(float(line.split()[7]))
                    self.charg.append(float(line.split()[8]))
                    self.vdWRa.append(float(line.split()[9]))
                    self.AtomPos.append([float(line.split()[5]),float(line.split()[6]), float(line.split()[7])])
                    line_split = line.split()
                    AtomType = line_split[2]
                    if AtomType[0] == 'O':
                        self.atomT.append(0)
                    elif AtomType[0] == 'N':
                        self.atomT.append(1)
                    elif AtomType[0] == 'C':
                        self.atomT.append(2)
                    elif AtomType[0] == 'S':
                        self.atomT.append(3)
                    elif AtomType[0] == 'H':
                        self.atomT.append(4)
                    else:
                        print('Error in pqr file line %d\n'%(self.atomN))
                        sys.exit()

    def exp_Lor_fcn(self,tau,kappa,nu):
        self.expfcn,self.Lorfcn,self.expqfcn,self.Lorqfcn,self.VDW,self.CLB = \
        _exp_Lor_fcn(tau,kappa,nu,self.atomN,self.atomT,self.atomx,self.atomy,self.atomz,self.vdWRa,self.charg,mat_idx)
   
    def exp_fcn(self,tau,pwr): # exponential function without charge
        self.expfcn,self.expqfcn = \
        _exp_fcn(tau,pwr,self.atomN,self.atomT,self.atomx,self.atomy,self.atomz,self.vdWRa,self.charg,mat_idx)
        
    def lor_fcn(self, tau, pwr):
        self.Lorfcn,self.Lorqfcn = \
        _lor_fcn(tau,pwr,self.atomN,self.atomT,self.atomx,self.atomy,self.atomz,self.vdWRa,self.charg,mat_idx)



if __name__ == "__main__":
    one = time.time()
    # filename = sys.argv[1]
    p = protein()
    kernels = [['E',0.3,5,'1'], ['E',4.7,5,'q'], ['L',4.2,2,'1']]
    output = open('VDW_CLB_FRI.txt', 'w')
    output_list = []
    for k in kernels:
        p.exp_Lor_fcn(k[1], k[2], k[2])
        if k[0] == 'E':
            if k[3] == 'q':
                output_list += p.expqfcn
            else:
                output_list += p.expfcn
        if k[0] == 'L':
            if k[3] == 'q':
                output_list += p.Lorqfcn
            else:
                output_list += p.Lorfcn
    output_list += p.VDW # yang
    output_list += p.CLB # yang
    output.write(str(output_list)[1:-1])
    output.close()
    print(time.time() - one)
