from init import abm

import os
import gzip
import time
import pickle
import logging
import multiprocessing

import numpy as np
import scipy.stats as ss

sim = 'bristol25'
log_file = 'logs/{0}.log'.format(time.ctime())

# get logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# create a file handler
handler = logging.FileHandler(log_file)
handler.setLevel(logging.INFO)

# create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(handler)

places = abm.Places(sim).names
# Use this to test a few palces
# places = ['Telford and Wrekin']

# These are the scenarios we want to process
scenarios = ['k5','k5-original','k6','k7','k9']

# File naming convention for quick retrieval for analysis
# common file looks like
    # 'analysis/bristol25/_/City of Bristol.C'
    # 'analysis/bristol25/_/City of Bristol.DX'
# scenario specific file looks like 
    # 'analysis/bristol25/k5/City of Bristol.R'
    # 'analysis/bristol25/k5/City of Bristol.T'
    # 'analysis/bristol25/k5/City of Bristol.Q'
fname = {}
for scenario in scenarios:
    fname[scenario] = 'analysis/{}/{}'.format(sim,scenario)
fname['_'] = 'analysis/{}/_'.format(sim)

# Create folders if they do not already exist
for k in fname:
    if not os.path.isdir(fname[k]):
        os.makedirs(fname[k])    

# Variables that are common between all scenarios
common_lab = ['N','W','X','Dmax','Ds','Dm','D50','D90','T90f','Qp']
# Variables that are unique to each scenario
unique_lab = ['Tmax','Tmaxf','Ts','Tm','T90','T90f1i','T90f2i','T90b','Qmax','Qs','Qm','Qmf','Qmf1','Qmf2','Qmb','Q50','Q50f','Q50f1','Q50f2','Q50b','Q90','Q90f','Q90f1','Q90f2','Q90b']

# Process pool output
def job(place):
    # Check if all scenarios have been processed
    complete = True
    for scenario in scenarios:
        # Check this file to make sure that a city has been completed        
        complete = complete and os.path.isfile('{}/{}.T.pickle'.format(fname[scenario],place))

    # Only process if not already processed
    if not complete:
        # Load path and the default scenario here!
        s = abm.Sim(sim=sim,place=place)
        s.load_agent_meta()

        # DX_destin is a vector containing distances of all agents away from the exit
        # Useful to cache in the COMMON folder
        folder = fname['_'].format(scenario)
        with open('{}/{}.DX.pickle'.format(fname['_'],place), 'w') as file:
            pickle.dump(s.DX_destin, file)

        # Output common in all files
        X = list(set(s.h.destins))
        N = [] # Number of agents per destin
        W = [] # Width per destin
        for x in X:
            # There has to be a width if a destin node exists
            w = s.h.destin_width_dict[x]
            W.append(w)
            # Sometimes, a destin node may not have any agent allocated to it in which case we call it zero
            try:
                n = s.n_destin[x]
            except KeyError: 
                n = 0
            N.append(n)

        for scenario in scenarios:
            s.init_scenario(scenario)
            s.load_result_meta()

            C = {}
            for l in common_lab:
                C[l] = np.array([]) # Initialise empty numpy array

            R = {}            
            for l in unique_lab:
                R[l] = np.array([]) # Initialise empty numpy array

            for n,w,x in zip(N,W,X):
                r = {}
                if n > 0:
                    # All the variables below are common to all scenarios
                    # ---------------------------------------------------                    
                    r['N'] = n
                    r['W'] = w
                    r['X'] = x

                    # List of agent distances to exit 'x'
                    d = s.DX_destin[x]
                    r['Dmax'] = max(d)
                    r['Ds'] = np.std(d)
                    r['Dm'] = np.mean(d)
                    r['D50'] = np.percentile(d,50)
                    r['D90'] = np.percentile(d,90)

                    # in minutes
                    r['T90f'] = r['D90']/s.fd.v_ff/s.fd.speedup

                    # predictive flow
                    r['Qp'] = n/r['T90f']/w

                    # All the variables below are unique to each scenario
                    # ---------------------------------------------------
                    # in minutes
                    t = s.T_destin[x]
                    r['Tmax'] = max(t)
                    r['Tmaxf'] = r['Dmax']/s.fd.v_ff/s.fd.speedup
                    r['Ts'] = np.std(t)
                    r['Tm'] = np.mean(t)
                    r['T90'] = np.percentile(t,90)
                    # Integer form so that it can be used as index
                    # np.ceil to avoid value error for q90f
                    r['T90f1i'] = int(np.ceil(r['T90f']/2))
                    r['T90f2i'] = int(np.ceil(r['T90f']))
                    r['T90b'] = r['T90']-r['T90f']

                    # The entire flow range
                    q = s.Q_destin[x]
                    r['Qmax'] = max(q)            
                    r['Qs'] = np.std(q)                    
                    r['Qm'] = np.mean(q)
                    r['Q50'] = np.percentile(q,50)                    
                    r['Q90'] = np.percentile(q,90)

                    # The whole free flow range
                    r['Qmf'] = np.mean(q[:r['T90f2i']])                    
                    r['Q50f'] = np.percentile(q[:r['T90f2i']],50)                
                    r['Q90f'] = np.percentile(q[:r['T90f2i']],90)  

                    # First free flow range
                    r['Qmf1'] = np.mean(q[:r['T90f1i']])                    
                    r['Q50f1'] = np.percentile(q[:r['T90f1i']],50)
                    r['Q90f1'] = np.percentile(q[:r['T90f1i']],90)

                    # Second free flow range
                    qf1f2 = q[r['T90f1i']:r['T90f2i']]
                    if qf1f2:
                        r['Qmf2'] = np.mean(q[r['T90f1i']:r['T90f2i']])                    
                        r['Q50f2'] = np.percentile(qf1f2,50)
                        r['Q90f2'] = np.percentile(q[r['T90f1i']:r['T90f2i']],90)                    
                    else:
                        r['Qmf2'] = r['Qmf1']
                        r['Q50f2'] = r['Q50f1']
                        r['Q90f2'] = r['Q90f1']                    

                    # Bottleneck range
                    qb = q[r['T90f2i']:]                    
                    if qb:
                        r['Qmb'] = np.mean(qb)   
                        r['Q50b'] = np.percentile(qb,50)
                        r['Q90b'] = np.percentile(q[r['T90f2i']:],90)                    
                    else:
                        r['Qmb'] = 0. 
                        r['Q50b'] = 0.
                        r['Q90b'] = 0.

                    # Stack the results together
                    for l in common_lab:
                        C[l] = np.hstack((C[l],r[l]))
                    for l in unique_lab:
                        R[l] = np.hstack((R[l],r[l]))

            # Save the common files
            with open('{}/{}.C.pickle'.format(fname['_'],place), 'w') as file:
                pickle.dump(C, file)

            with open('{}/{}.R.pickle'.format(fname[scenario],place), 'w') as file:
                pickle.dump(R, file)

            with open('{}/{}.Q.pickle'.format(fname[scenario],place), 'w') as file:
                pickle.dump(s.T_destin, file)

            with open('{}/{}.T.pickle'.format(fname[scenario],place), 'w') as file:
                pickle.dump(s.Q_destin, file)

        return 'now processed.'
    else:
        return 'available.'

# Call this and disable multiprocessing to debug issues
# action(places[0])

# ---------------------------------------------------------
# Start multi processing

if __name__ == '__main__':
    pool = Pool(processes=cpu_count()) 
    pool_outputs = pool.map(job, places)    

    # Show results of the processing
    for place,output in zip(places,pool_outputs):
        print place,output

# ---------------------------------------------------------
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

# Composite labels
common_lab.remove('X')
lab = common_lab + unique_lab

# ---------------------------------------------------------
# Module to load from file

def load(scenario,city,extension):
    file = open('{}/{}.{}.pickle'.format(fname[scenario],city,extension), 'r')
    r = pickle.load(file)
    file.close()
    return r

# ---------------------------------------------------------
# Retrieve into a dict

import metrics_subgraph as ms

def retrieve_X_Xi():
    X = {} # holds the index of the destination node for the city
    Xi = {} # holds the index of the index of destination node in the list

    this = 0
    tot = 0

    for city in places:
        c = load('_',city,'C')
        
        # Load variables unique to the scenario
        X[city] = c['X']
        Xi[city] = range(this,this+len(X[city]))

        this = this+len(X[city])

    return X,Xi

def retrieve_M():
    M = {}
    X,Xi = retrieve_X_Xi()
    for city in places:
        for x in X[city]:
            # print x
            summary = ms.load_json('summary',city,int(x))
            for key,value in summary.iteritems():
                try:
                    M[key] = np.hstack((M[key],value))
                except KeyError:
                    M[key] = np.array(value)

    return M

def retrieve_R(scenario):
    ''' Returns R for an input scenario
    '''
    R = {}
    # Load constants
    for l in lab:
        R[l] = np.array([],float) 

    X,Xi = retrieve_X_Xi()


    for city in places:
        print 'Loading', city
        # Load common

        c = load('_',city,'C')
        r = load(scenario,city,'R')

        for l in common_lab:
            R[l] = np.hstack((R[l], c[l]))

        for l in unique_lab:
            R[l] = np.hstack((R[l], r[l]))

    return R