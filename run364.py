import warnings
warnings.filterwarnings('ignore')
import numpy as np
# import time
from tensorflow.keras.models import Sequential, load_model
import sys
import os
from timeit import default_timer as timer
from sklearn.preprocessing import StandardScaler
from generate_feature import *
from multiprocessing import Process,Manager,Pool
import multiprocessing
import tensorflow as tf

def fetchPDBlist(input_file):
    PDBlist = []
    openfile = open(input_file,'r')
    lines = openfile.readlines()
    for line in lines:
        PDBID = line[:4]
        PDBlist.append(PDBID)
    openfile.close()
    return PDBlist

def copyPQR(PDBID): ## option = 2
    os.system('cp '+rootdir+'/../../pbml_xu/'+'CoreSet'+'/'+PDBID+'/pro.pqr '+rootdir+'/CoreSet/'+PDBID+'/pro.pqr')

def getXtest(PDBID):
    os.system('python feature.py')
    os.system('bornRadius -pqr pro.pqr -surf_density 13 -energy total_solv > bornRadius.txt')
    os.system('MS_Intersection pro.pqr 1.4 0.8 1')  

    X_test = generate_features(rootdir+'/CoreSet/'+PDBID)
    return X_test

# def normalization(X_train,X_test):
    # scaler = StandardScaler()
    # scaler.fit(X_test)
    # X_test_norm = scaler.transform(np.array(X_test).reshape(1,-1))

    # X_test = []
    # X_test_lines = open(rootdir+'/X_test.txt', 'r').readlines()
    # for line in X_test_lines:
    #     row = [float(x) for x in line.split(', ')]
    #     X_test.append(row)
    #     break
    # print(X_test)

########## need a further check: why use X_train for normalization ##########
    # X_train = []
    # X_train_lines = open(rootdir+'/X_train.txt', 'r').readlines()
    # for line in X_train_lines:
    #     row = [float(x) for x in line.split(', ')]
    #     X_train.append(row)

    # scaler = StandardScaler()
    # scaler.fit(X_train)
    # X_train_norm = scaler.transform(X_train)
    # print (np.array(X_test).shape)
    # # print("X_train: ", np.array(X_train).shape) # (4294,427)
    # X_test_norm = scaler.transform(np.array(X_test).reshape(1,-1))
 ################################################   

    # X_test = []
    # bornRadius_file = open('VDW_CLB_FRI.txt','r')
    # for line in bornRadius_file.readlines():
    #     list = line.split(',')
    #     for item in list:
    #         X_test.append(float(item))
    # # print(X_test)
    # return X_test_norm

def runDNN(X_test_norm):
    pbml_model = load_model(rootdir+'/saved_model/364')
    # print(pbml_model.summary())

    yhat = pbml_model.predict(X_test_norm)
    yhat = yhat.flatten()
    print(yhat)
    return yhat


def output2file(PDBID, yhat, MIBPB, GB, abserr, time):
    f = open(PDBID + '_err.txt', 'w')
    f.write('MIBPB: '+ str(MIBPB) +'\n')
    f.write('GB: '+ str(GB) +'\n')
    f.write('yhat: '+str(yhat)+'\n')
    f.write('abs error: '+str(abserr)+'\n')
    f.write('time: '+str(time)+'\n')
    f.close()

def main4parallel(PDBID, MIBPB, GB):
    os.system('mkdir CoreSet/'+PDBID)
    copyPQR(PDBID)
    os.chdir('CoreSet/'+PDBID)
    # if os.path.isfile('GB.result') == True:
    #     os.system('rm GB.result')

    start = timer()
    X_test = getXtest(PDBID)
    X_test_norm = scaler.transform(np.array(X_test).reshape(1,-1))
    yhat = runDNN(X_test_norm)
    end = timer()
    time = end-start
    abserr = abs(float(MIBPB) - (float(yhat)+float(GB)))
    output2file(PDBID, yhat, MIBPB, GB, abserr, time)

    # return yhat, time  

if __name__ == '__main__':
    rootdir = os.path.dirname(os.path.abspath(__file__))
    # input_file = rootdir+'/../../pbml_xu/'+'/ylabel/MIBPBCore.txt'
    # PDBlist = fetchPDBlist(input_file)
    # PDBlist = ['3muz']
######################################################
    # # read in X_train for StandardScaler
    X_train = []
    X_train_lines = open(rootdir+'/X_train.txt', 'r').readlines()
    for line in X_train_lines:
        row = [float(x) for x in line.split(', ')]
        X_train.append(row)
    scaler = StandardScaler()
    scaler.fit(X_train)
######################################################
    # # read in MIBPB and GB 
    # MIBPB_core = []; GB_core = []
    # MIBPB_lines = open(rootdir+'/MIBPB_core.txt', 'r').readlines()
    # for line in MIBPB_lines:
    #     MIBPB_core.append(float(line))
    # GB_lines = open(rootdir+'/GB_core.txt', 'r').readlines()
    # for line in GB_lines:
    #     GB_core.append(float(line))

    # # for parallel
    # numCPT = int(multiprocessing.cpu_count())
    # print("numCPT is: "+str(numCPT))
    # pool = Pool(processes = numCPT)
    # # idx = 0
    # idx = 128
    # for PDBID in PDBlist[128:]:
    #     print("current PDB is: "+str(PDBID)+' '+str(idx)+'\n')
    #     MIBPB = MIBPB_core[idx]
    #     GB = GB_core[idx]
    #     pool.apply_async(main4parallel, (PDBID,MIBPB,GB,))
    #     idx += 1  
    # pool.close()
    # pool.join()
    # main4parallel('3muz',-80090.353037555993,-79325.6)

    # # compute MAPE
    # yhatlist = []
    # for PDBID in PDBlist:
    #     result = open(rootdir + '/CoreSet/' + PDBID + '/' + PDBID + '_err.txt', 'r')
    #     lines = result.readlines()
    #     # print (PDBID)
    #     for line in lines:
    #         if 'yhat' in line:
    #             yhat = line.split(': ')[1][1:-2]
    #             # print(yhat)
    #     yhatlist.append(float(yhat))

    # print(len(yhatlist))
    # print(len(MIBPB_core)) 
    # print(len(GB_core)) 

    # MAPEerr = getError(yhatlist, MIBPB_core, GB_core)


    # yhatlist = []
    # idx = 0
    # #############
    # for PDBID in PDBlist[:1]:
    #     os.system('mkdir CoreSet/'+PDBID)
    #     copyPQR(PDBID)
    #     os.chdir('CoreSet/'+PDBID)
    #     if os.path.isfile('GB.result') == True:
    #         os.system('rm GB.result')

    #     start = timer()
    #     X_test = getXtest(PDBID)
    #     X_test_norm = scaler.transform(np.array(X_test).reshape(1,-1))
    #     yhat = runDNN(X_test_norm)
    #     end = timer()
    #     abserr = abs(MIBPB_core[idx] - (yhat+GB_core[idx]))
    #     time = end-start
    #     output2file(PDBID, yhat, MIBPB_core[idx], GB_core[idx], abserr, time)
    #     idx += 1
    #     # for all 195 proteins
    #     yhatlist.append(yhat)

    # result = getError(yhatlist, MIBPB_core, GB_core)


    # # X_test = generate_features(rootdir+'/CoreSet/'+PDBID)
    # start = timer()
    # # X_test = getXtest(PDBID)
    # X_test = generate_features('prep_bind/'+'1b2u/'+'')
    # X_test_norm = scaler.transform(np.array(X_test).reshape(1,-1))
    # yhat = runDNN(X_test_norm)
    # end = timer()
    # time = end-start

    # abserr = abs(MIBPB - (yhat+GB))
    # output2file(PDBID, yhat, MIBPB, GB, abserr, time)

#####################################################################
    # show protein transferability on dna and drug pqr
    PDBlist = []
    for filename in os.listdir(os.path.join(rootdir,'../binding_pqrs/data-set2')):
        PDBlist.append(filename[:4])
    
    for PDBID in np.unique(PDBlist):
        print(PDBID)
        # os.system('mkdir prep_bind/data-set2/nase_n/'+PDBID)
        # os.system('cp '+rootdir+'/../binding_pqrs/data-set2/'+PDBID+'_barnase_n.pqr '+rootdir+'/prep_bind/data-set2/nase_n/'+PDBID+'/pro.pqr')      
        os.chdir(rootdir+'/prep_bind/data-set2/nase_n/'+PDBID)

        # run MIB for comparison
        os.system('cp '+rootdir+'/prep_bind/rMIB.exe '+rootdir+'/prep_bind/data-set2/nase_n/'+PDBID)
        os.system('cp '+rootdir+'/prep_bind/usrdata.in '+rootdir+'/prep_bind/data-set2/nase_n/'+PDBID)
        os.system('mkdir '+rootdir+'/prep_bind/data-set2/nase_n/'+PDBID+'/test_proteins')
        os.system('mkdir '+rootdir+'/prep_bind/data-set2/nase_n/'+PDBID+'/surfaces')
        os.system('cp ./pro.pqr ./test_proteins')
        os.system('./rMIB.exe')

        # generate feature and run ML model
        start = timer()
        os.system('python '+ rootdir+'/feature.py')
        os.system('bornRadius -pqr pro.pqr -surf_density 13 -energy total_solv > bornRadius.txt')
        os.system('MS_Intersection pro.pqr 1.4 0.8 1')  
        X_test = generate_features(rootdir+'/prep_bind/data-set2/nase_n/'+PDBID)
        X_test_norm = scaler.transform(np.array(X_test).reshape(1,-1))
        yhat = runDNN(X_test_norm)
        end = timer()
        time = end-start 

        GBfile = open(rootdir+'/prep_bind/data-set2/nase_n/'+PDBID+'/bornRadius.txt','r')
        lines = GBfile.readlines()
        GB = lines[4].split()[3]
        GBfile.close()

        MIBfile = open(rootdir+'/prep_bind/data-set2/nase_n/'+PDBID+'/output.txt','r')
        lines = MIBfile.readlines()
        MIB = lines[0].split()[0]
        MIBfile.close()            

        # abserr = abs(MIB - (yhat+GB))
        output2file(PDBID, yhat, MIB, GB, 0, time)
        os.chdir(rootdir)

    for PDBID in np.unique(PDBlist):
        print(PDBID)
        # os.system('mkdir prep_bind/data-set2/star_n/'+PDBID)
        # os.system('cp '+rootdir+'/../binding_pqrs/data-set2/'+PDBID+'_barstar_n.pqr '+rootdir+'/prep_bind/data-set2/star_n/'+PDBID+'/pro.pqr')      
        os.chdir(rootdir+'/prep_bind/data-set2/star_n/'+PDBID)

        # run MIB for comparison
        os.system('cp '+rootdir+'/prep_bind/rMIB.exe '+rootdir+'/prep_bind/data-set2/star_n/'+PDBID)
        os.system('cp '+rootdir+'/prep_bind/usrdata.in '+rootdir+'/prep_bind/data-set2/star_n/'+PDBID)
        os.system('mkdir '+rootdir+'/prep_bind/data-set2/star_n/'+PDBID+'/test_proteins')
        os.system('mkdir '+rootdir+'/prep_bind/data-set2/star_n/'+PDBID+'/surfaces')
        os.system('cp ./pro.pqr ./test_proteins')
        os.system('./rMIB.exe')

        # generate feature and run ML model
        start = timer()
        os.system('python '+ rootdir+'/feature.py')
        os.system('bornRadius -pqr pro.pqr -surf_density 13 -energy total_solv > bornRadius.txt')
        os.system('MS_Intersection pro.pqr 1.4 0.8 1')  
        X_test = generate_features(rootdir+'/prep_bind/data-set2/star_n/'+PDBID)
        X_test_norm = scaler.transform(np.array(X_test).reshape(1,-1))
        yhat = runDNN(X_test_norm)
        end = timer()
        time = end-start 

        GBfile = open(rootdir+'/prep_bind/data-set2/star_n/'+PDBID+'/bornRadius.txt','r')
        lines = GBfile.readlines()
        GB = lines[4].split()[3]
        GBfile.close()

        MIBfile = open(rootdir+'/prep_bind/data-set2/star_n/'+PDBID+'/output.txt','r')
        lines = MIBfile.readlines()
        MIB = lines[0].split()[0]
        MIBfile.close()            

        # abserr = abs(MIB - (yhat+GB))
        output2file(PDBID, yhat, MIB, GB, 0, time)
        os.chdir(rootdir)



    for PDBID in np.unique(PDBlist):
        print(PDBID)     
        os.chdir(rootdir+'/prep_bind/data-set2/combined/'+PDBID)

        # run MIB for comparison
        os.system('cp '+rootdir+'/prep_bind/rMIB.exe '+rootdir+'/prep_bind/data-set2/combined/'+PDBID)
        os.system('cp '+rootdir+'/prep_bind/usrdata.in '+rootdir+'/prep_bind/data-set2/combined/'+PDBID)
        os.system('mkdir '+rootdir+'/prep_bind/data-set2/combined/'+PDBID+'/test_proteins')
        os.system('mkdir '+rootdir+'/prep_bind/data-set2/combined/'+PDBID+'/surfaces')
        os.system('cp ./pro.pqr ./test_proteins')
        os.system('./rMIB.exe')

        # generate feature and run ML model
        start = timer()
        os.system('python '+ rootdir+'/feature.py')
        os.system('bornRadius -pqr pro.pqr -surf_density 13 -energy total_solv > bornRadius.txt')
        os.system('MS_Intersection pro.pqr 1.4 0.8 1')  
        X_test = generate_features(rootdir+'/prep_bind/data-set2/combined/'+PDBID)
        X_test_norm = scaler.transform(np.array(X_test).reshape(1,-1))
        yhat = runDNN(X_test_norm)
        end = timer()
        time = end-start 

        GBfile = open(rootdir+'/prep_bind/data-set2/combined/'+PDBID+'/bornRadius.txt','r')
        lines = GBfile.readlines()
        GB = lines[4].split()[3]
        GBfile.close()

        MIBfile = open(rootdir+'/prep_bind/data-set2/combined/'+PDBID+'/output.txt','r')
        lines = MIBfile.readlines()
        MIB = lines[0].split()[0]
        MIBfile.close()            

        # abserr = abs(MIB - (yhat+GB))
        output2file(PDBID, yhat, MIB, GB, 0, time)
        os.chdir(rootdir)

