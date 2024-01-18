#### Weihua Geng, Yongjia Xu, Xin (Sharon) Yang
#### PBML Project, Math Department, Southern Methodist University
#### This script has functions to generate features
#### input: 


import warnings
warnings.filterwarnings('ignore')
import numpy as np
import sys
import os

AminoA_w2a = {"ARG":"R", "HIS":"H", "LYS":"K", "ASP":"D",
              "GLU":"E", "SER":"S", "THR":"T", "ASN":"N",
              "GLN":"Q", "CYS":"C", "GLY":"G", "PRO":"P",
              "ALA":"A", "VAL":"V", "ILE":"I", "LEU":"L",
              "MET":"M", "PHE":"F", "TYR":"Y", "TRP":"W"}
Hydro = ['A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W']
PolarAll = ['S','T','N','Q','R','H','K','D','E']
PolarUncharged = ['S','T','N','Q']
PolarPosCharged = ['R','H','K']
PolarNegCharged = ['D','E']
SpecialCase = ['C','U','G','P']
AAvolume = {'A': 88.6, 'R':173.4, 'D':111.1, 'N':114.1, 'C':108.5, \
            'E':138.4, 'Q':143.8, 'G': 60.1, 'H':153.2, 'I':166.7, \
            'L':166.7, 'K':168.6, 'M':162.9, 'F':189.9, 'P':112.7, \
            'S': 89.0, 'T':116.1, 'W':227.8, 'Y':193.6, 'V':140.0}
AAhydropathy = {'A': 1.8, 'R':-4.5, 'N':-3.5, 'D':-3.5, 'C': 2.5, \
                'E':-3.5, 'Q':-3.5, 'G':-0.4, 'H':-3.2, 'I': 4.5, \
                'L': 3.8, 'K':-3.9, 'M': 1.9, 'F': 2.8, 'P':-1.6, \
                'S':-0.8, 'T':-0.7, 'W':-0.9, 'Y':-1.3, 'V': 4.2}
AAarea = {'A':115., 'R':225., 'D':150., 'N':160., 'C':135., \
          'E':190., 'Q':180., 'G': 75., 'H':195., 'I':175., \
          'L':170., 'K':200., 'M':185., 'F':210., 'P':145., \
          'S':115., 'T':140., 'W':255., 'Y':230., 'V':155.}
AAweight = {'A': 89.094, 'R':174.203, 'N':132.119, 'D':133.104, 'C':121.154, \
            'E':147.131, 'Q':146.146, 'G': 75.067, 'H':155.156, 'I':131.175, \
            'L':131.175, 'K':146.189, 'M':149.208, 'F':165.192, 'P':115.132, \
            'S':105.093, 'T':119.12 , 'W':204.228, 'Y':181.191, 'V':117.148}
AApharma = {'A':[0,1,3,1,1,1],'R':[0,3,3,2,1,1],'N':[0,2,4,1,1,0],'D':[0,1,5,1,2,0],\
            'C':[0,2,3,1,1,0],'E':[0,1,5,1,2,0],'Q':[0,2,4,1,1,0],'G':[0,1,3,1,1,0],\
            'H':[0,3,5,3,1,0],'I':[0,1,3,1,1,2],'L':[0,1,3,1,1,1],'K':[0,2,4,2,1,2],\
            'M':[0,1,3,1,1,2],'F':[1,1,3,1,1,1],'P':[0,1,3,1,1,1],'S':[0,2,4,1,1,0],\
            'T':[0,2,4,1,1,1],'W':[2,2,3,1,1,2],'Y':[1,2,4,1,1,1],'V':[0,1,3,1,1,1]}
Groups = [Hydro, PolarAll, PolarUncharged, PolarPosCharged, PolarNegCharged, SpecialCase]


def AAcharge(AA):
    if AA in ['D','E']:
        return -1.
    elif AA in ['R','H','K']:
        return 1.
    else:
        return 0.

def res_feature(pro_lines):
    FeatureEnv = []
    NearSeq = []
    cur_resID = -1
    for pro_line in pro_lines:
        if pro_line[0:4] == 'ATOM':
            pro_line_list = pro_line.strip().split()
            if pro_line_list[3] in AminoA_w2a:
                AA = AminoA_w2a[pro_line_list[3]]
                resID = pro_line_list[4]
                if resID != cur_resID:
                    NearSeq.append(AA)
                    cur_resID = resID
    for Group in Groups:
        cnt = 0.
        for AA in NearSeq:
            if AA in Group:
                cnt += 1.
        FeatureEnv.append(cnt)
        FeatureEnv.append(cnt/max(1., float(len(NearSeq))))
    Vol = []; Hyd = []; Area = []; Wgt = []; Chg = []
    phara = [0, 0, 0, 0, 0, 0]
    for AA in NearSeq:
        Vol.append(AAvolume[AA])
        Hyd.append(AAhydropathy[AA])
        Area.append(AAarea[AA])
        Wgt.append(AAweight[AA])
        Chg.append(AAcharge(AA))
        for i in range(6):
            phara[i] += AApharma[AA][i]
    Vol = np.asarray(Vol)
    Hyd = np.asarray(Hyd)
    Area = np.asarray(Area)
    Wgt = np.asarray(Wgt)

    if len(NearSeq) == 0:
        FeatureEnv.extend([0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.])
    else:
        FeatureEnv.extend([np.sum(Vol), np.sum(Vol)/float(len(NearSeq)), np.var(Vol)])
        FeatureEnv.extend([np.sum(Hyd), np.sum(Hyd)/float(len(NearSeq)), np.var(Hyd)])
        FeatureEnv.extend([np.sum(Area), np.sum(Area)/float(len(NearSeq)), np.var(Area)])
        FeatureEnv.extend([np.sum(Wgt), np.sum(Wgt)/float(len(NearSeq)), np.var(Wgt)])
    FeatureEnv.append(sum(Chg))
    FeatureEnv.extend(phara)
    return FeatureEnv

def generate_features(folder):
    row = []
    # VDW_CLB_FRI
    f = open(folder+'/VDW_CLB_FRI.txt','r')
    lines = f.readlines()
    row = [float(x) for x in lines[0].split(', ')]
    f.close()
    
    #GB.result
    f = open(folder+'/GB.result', 'r')
    lines = f.readlines()
    GB_list = []
    for line in lines:
        line_list = line.split()
        for x in line_list:
            GB_list.append(float(x))
    
    if len(GB_list) != 240:
        GB_list = GB_list[:240]
        print("GB_list isn't 240: " + str(folder) + "\n")
    
    # Area+Charge+abs Charge
    area_file = open(folder+'/partition_area.txt','r')
    area_lines = area_file.readlines()
    pro_file =  open(folder+'/pro.pqr','r')
    pro_lines = pro_file.readlines()
    area_i = 0
    area_features = [0]*7
    charge_features = [0]*7
    abs_charge_features = [0]*7
    for i in range(len(pro_lines)):
        pro_line = pro_lines[i]
        if pro_line[0:4] == 'ATOM':
            area_line = area_lines[area_i]
            area_i += 1
            pro_line_list = pro_line.strip().split()
            area_line_list = area_line.strip().split()
            atom = pro_line_list[2][0]
            atom_area = float(area_line_list[1])
            charge = float(pro_line_list[8])
            if atom == 'C':
                area_features[0] += atom_area; charge_features[0] += charge; abs_charge_features[0] += abs(charge)
                area_features[5] += atom_area; charge_features[5] += charge; abs_charge_features[5] += abs(charge)
                area_features[6] += atom_area; charge_features[6] += charge; abs_charge_features[6] += abs(charge)
            if atom == 'N':
                area_features[1] += atom_area; charge_features[1] += charge; abs_charge_features[1] += abs(charge)
                area_features[5] += atom_area; charge_features[5] += charge; abs_charge_features[5] += abs(charge)
                area_features[6] += atom_area; charge_features[6] += charge; abs_charge_features[6] += abs(charge)
            if atom == 'O':
                area_features[2] += atom_area; charge_features[2] += charge; abs_charge_features[2] += abs(charge)
                area_features[5] += atom_area; charge_features[5] += charge; abs_charge_features[5] += abs(charge)
                area_features[6] += atom_area; charge_features[6] += charge; abs_charge_features[6] += abs(charge)
            if atom == 'S':
                area_features[3] += atom_area; charge_features[3] += charge; abs_charge_features[3] += abs(charge)
                area_features[5] += atom_area; charge_features[5] += charge; abs_charge_features[5] += abs(charge)
                area_features[6] += atom_area; charge_features[6] += charge; abs_charge_features[6] += abs(charge)
            if atom == 'H':
                area_features[4] += atom_area; charge_features[4] += charge; abs_charge_features[4] += abs(charge)
                area_features[6] += atom_area; charge_features[6] += charge; abs_charge_features[6] += abs(charge)
    FeatureEnv = res_feature(pro_lines)
    feature = row+GB_list+area_features+charge_features+abs_charge_features+FeatureEnv
    # feature = row+area_features+charge_features+abs_charge_features+FeatureEnv # without GB list
    return feature

# get error
def getError(yhat, MIBPB_core, GB_core):
    GBDTresult = []
    for i in range(len(GB_core)):
        GBDTresult.append(yhat[i] + GB_core[i])
    ML_MAPE = 0; GB_MAPE = 0
    
    for i in range(len(GBDTresult)):
        # if i != 143:
        temp1 = abs(MIBPB_core[i]-GBDTresult[i])/abs(MIBPB_core[i])
        temp2 = abs(MIBPB_core[i]-GB_core[i])/abs(MIBPB_core[i])
        if np.isnan(temp1) == True:
            print(str(temp1)+ ' ' + str(i) + ' temp1')
        elif np.isnan(temp2) == True:
            print(str(temp2)+ ' ' + str(i) + ' temp2')
        else:
            ML_MAPE += temp1
            GB_MAPE += temp2
    ML_MAPE /= len(GBDTresult)
    GB_MAPE /= len(GBDTresult)
    print("ML MAPE: {}, GB MAPE: {}".format(ML_MAPE, GB_MAPE))

