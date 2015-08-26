import os
import abm
import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt

sim = 'bristol25'
p = abm.Places(sim)

import pickle

def loadR(label):
    file = open('analysis/{0}/!R/{1}'.format(sim,label), 'r')
    R[label] = pickle.load(file)
    file.close()

def saveR(label):
    file = open('analysis/{0}/!R/{1}'.format(sim,label), 'w')
    pickle.dump(R[label],file)
    file.close()    

def loadCity(thing,city):
    file = open('analysis/{0}/{1}/{2}'.format(sim,thing,city), 'r')
    r = pickle.load(file)
    file.close()
    return r

R = {}

# Load constants
R['N'] = np.array([],int)
R['W'] = np.array([],float)

# Load metrics that change
lab = ['Dmax','Ds','Dm','D50','D90','Tmax','Tmaxf','Ts','Tm','T90','T90f','T90f1i','T90f2i','T90b','Qmax','Qs','Qm','Qmf','Qmf1','Qmf2','Qmb','Q50','Q50f','Q50f1','Q50f2','Q50b','Q90','Q90f','Q90f1','Q90f2','Q90b','Qp']


# Pretty label
pl = {
    'N':'N',
    'W':'W',
    'Dmax':'D^{max}',
    'Ds':'D^{\sigma}',
    'Dm':'\overline{D}',
    'D50':'D^{50\%}',
    'D90':'D^{90\%}',
    'Tmax':'T^{max}',
    'Tmaxf':'T^{max}_f',
    'Ts':'T^{\sigma}',
    'Tm':'\overline{T}',
    'T90':'T^{90\%}',
    'T90f':'T^{90\%}_f',
    'T90f1i':'T^{90\%}_{f1}',
    'T90f2i':'T^{90\%}_{f2}',
    'T90b':'T^{90\%}_b',
    'Qmax':'Q^{max}',
    'Qs':'Q^{\sigma}',
    'Qm':'\overline{Q}',
    'Qmf':'\overline{Q}_f',
    'Qmf1':'\overline{Q}_{f1}',
    'Qmf2':'\overline{Q}_{f2}',
    'Qmb':'\overline{Q}_b',
    'Q50':'Q^{50\%}',
    'Q50f':'Q^{50\%}_f',
    'Q50f1':'Q^{50\%}_{f1}',
    'Q50f2':'Q^{50\%}_{f2}',
    'Q50b':'Q^{50\%}_b',
    'Q90':'Q^{90\%}',
    'Q90f':'Q^{90\%}_f',
    'Q90f1':'Q^{90\%}_{f1}',
    'Q90f2':'Q^{90\%}_{f2}',
    'Q90b':'Q^{90\%}_b',
    'Qp':'Q_c',     
     }

unit = {
    'N':'[ped]',
    'W':'[m]',
    'Dmax':'[m]',
    'Ds':'[m]',
    'Dm':'[m]',
    'D50':'[m]',
    'D90':'[m]',
    'Tmax':'[s]',
    'Tmaxf':'[s]',
    'Ts':'[s]',
    'Tm':'[s]',
    'T90':'[s]',
    'T90f':'[s]',
    'T90f1i':'[s]',
    'T90f2i':'[s]',
    'T90b':'[s]',
    'Qmax':'[ped/(ms)]',
    'Qs':'[ped/(ms)]',
    'Qm':'[ped/(ms)]',
    'Qmf':'[ped/(ms)]',
    'Qmf1':'[ped/(ms)]',
    'Qmf2':'[ped/(ms)]',
    'Qmb':'[ped/(ms)]',
    'Q50':'[ped/(ms)]',
    'Q50f':'[ped/(ms)]',
    'Q50f1':'[ped/(ms)]',
    'Q50f2':'[ped/(ms)]',
    'Q50b':'[ped/(ms)]',
    'Q90':'[ped/(ms)]',
    'Q90f':'[ped/(ms)]',
    'Q90f1':'[ped/(ms)]',
    'Q90f2':'[ped/(ms)]',
    'Q90b':'[ped/(ms)]',
    'Qp':'[ped/(ms)]',
     }

for l in lab:
    R[l] = np.array([],float) 

X = {} # holds the index of the destination node for the city
Xi = {} # holds the index of the index of destination node in the list

this = 0

for city in p.names:
    print 'Loading', city
    n = np.array(loadCity('N',city))
    w = np.array(loadCity('W',city))
    x = np.array(loadCity('X',city))
    
    # Filter catchment areas that meet the following condition
    condition = n>100
    
    R['N'] = np.hstack((R['N'], n[condition]))
    R['W'] = np.hstack((R['W'], w[condition]))
    
    X[city] = x[condition]
    Xi[city] = range(this,this+len(X[city]))
    this = this+len(X[city])
    
    d = loadCity('_',city)
    
    for l in lab:
        R[l] = np.hstack((R[l], np.array(d[l])[condition]))

# Add the labels not originally included        
lab=['N','W']+lab