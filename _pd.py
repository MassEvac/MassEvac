import os
import abm
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

p = abm.Places(sim)

ftemp = 'analysis/{0}/{1}'.format(sim,'{0}')

fname = {'_':ftemp.format('_'),
        'X':ftemp.format('X'),
        'N':ftemp.format('N'),
        'W':ftemp.format('W'),
        'Q':ftemp.format('Q'),
        'T':ftemp.format('T'),
        'D':ftemp.format('D')}
        
# Create folders if they dont already exist
for k in fname:
    if not os.path.isdir(fname[k]):
        os.makedirs(fname[k])    

# Check this file to make sure that a city has been completed
check = 'X'

def action(place):
    N = []
    W = []
    X = []
    Q = []
    T = []
    D = []
        
    if not os.path.isfile('{0}/{1}'.format(fname[check],place)):
        # Load path and the default scenario here!
        s = abm.Sim(sim=sim,place=place)
        s.init_scenario('ia')
        s.load_agent_meta()
        s.load_result_meta()
        
        X = s.h.destins
        Q = s.Q_destin
        T = s.T_destin
        D = s.DX_destin
        
        for x in X:
            # There has to be a width if a destin node exists
            w = s.h.destin_width_dict[x]
            W.append(w)
            
            # Sometimes, a destin node may not have any agent allocated to it in which case we call it zero
            try:
                n = s.n_destin[x]              
            except KeyError: 
                n = 0.
            N.append(n)
            
        file = open('{0}/{1}'.format(fname['N'],place), 'w')
        pickle.dump(N, file)
        file.close()

        file = open('{0}/{1}'.format(fname['W'],place), 'w')
        pickle.dump(W, file)
        file.close()

        file = open('{0}/{1}'.format(fname['X'],place), 'w')
        pickle.dump(X, file)
        file.close()

        file = open('{0}/{1}'.format(fname['Q'],place), 'w')
        pickle.dump(Q, file)
        file.close()

        file = open('{0}/{1}'.format(fname['T'],place), 'w')
        pickle.dump(T, file)
        file.close()
    
        file = open('{0}/{1}'.format(fname['D'],place), 'w')
        pickle.dump(D, file)
        file.close()
    else:
        file = open('{0}/{1}'.format(fname['N'],place), 'r')
        N = pickle.load(file)
        file.close()

        file = open('{0}/{1}'.format(fname['W'],place), 'r')
        W = pickle.load(file)
        file.close()

        file = open('{0}/{1}'.format(fname['X'],place), 'r')
        X = pickle.load(file)
        file.close()

        file = open('{0}/{1}'.format(fname['Q'],place), 'r')
        Q = pickle.load(file)
        file.close()

        file = open('{0}/{1}'.format(fname['T'],place), 'r')
        T = pickle.load(file)
        file.close()
    
        file = open('{0}/{1}'.format(fname['D'],place), 'r')
        D = pickle.load(file)
        file.close()        

    lab = ['Dmax','Ds','Dm','D50','D90','Tmax','Tmaxf','Ts','Tm','T90','T90f','T90f1i','T90f2i','T90b','Qmax','Qs','Qm','Qmf','Qmf1','Qmf2','Qmb','Q50','Q50f','Q50f1','Q50f2','Q50b','Q90','Q90f','Q90f1','Q90f2','Q90b','Qp']

    R = {}
    
    for l in lab:
        R[l] = np.array([]) # Initialise empty numpy array

    for n,w,x in zip(N,W,X):

        r = {}
        for l in lab:
            r[l] = 0 # Initialise the values to 0

        if n > 0:
            d = D[x]
            r['Dmax'] = max(d)
            r['Ds'] = np.std(d)
            r['Dm'] = np.mean(d)
            r['D50'] = np.percentile(d,50)
            r['D90'] = np.percentile(d,90)

            # in minutes
            t = T[x]
            r['Tmax'] = max(t)
            r['Tmaxf'] = r['Dmax']/abm.fd.vFf/60
            r['Ts'] = np.std(t)
            r['Tm'] = np.mean(t)
            r['T90'] = np.percentile(t,90)                        
            r['T90f'] = r['D90']/abm.fd.vFf/60
            # integer form so that it can be used as index
            # np.ceil to avoid value error for q90f
            r['T90f1i'] = int(np.ceil(r['T90f']/2))
            r['T90f2i'] = int(np.ceil(r['T90f']))
            r['T90b'] = r['T90']-r['T90f']

            # mean
            q = Q[x]
            r['Qmax'] = max(q)            
            r['Qs'] = np.std(q)
            r['Qm'] = np.mean(q)
            r['Qmf'] = np.mean(q[:r['T90f2i']])
            r['Qmf1'] = np.mean(q[:r['T90f1i']])
            try:
                r['Qmf2'] = np.mean(q[r['T90f1i']:r['T90f2i']])
            except IndexError:
                r['Qmf2'] = r['Qmf1']
            try:            
                r['Qmb'] = np.mean(q[r['T90f2i']:])
            except IndexError:
                r['Qmb'] = 0.

            # 50th percentile
            r['Q50'] = np.percentile(q,50)                
            r['Q50f'] = np.percentile(q[:r['T90f2i']],50)                
            r['Q50f1'] = np.percentile(q[:r['T90f1i']],50)
            try:
                r['Q50f2'] = np.percentile(q[r['T90f1i']:r['T90f2i']],50)
            except IndexError:
                r['Q50f2'] = r['Q50f1']
            try:
                r['Q50b'] = np.percentile(q[r['T90f2i']:],50)
            except IndexError:
                r['Q50b'] = 0.


            # 90th percentile   
            r['Q90'] = np.percentile(q,90)
            r['Q90f'] = np.percentile(q[:r['T90f2i']],90)                        
            r['Q90f1'] = np.percentile(q[:r['T90f1i']],90)
            try:
                r['Q90f2'] = np.percentile(q[r['T90f1i']:r['T90f2i']],90)
            except IndexError:            
                r['Q90f2'] = r['Q90f1']
            try:
                r['Q90b'] = np.percentile(q[r['T90f2i']:],90)
            except IndexError:
                r['Q90b'] = 0.

            r['Qp'] = n/r['T90f']/w

        # Stack the results together            
        for l in r.keys():
            R[l] = np.hstack((R[l],r[l]))

    file = open('{0}/{1}'.format(fname['_'],place), 'w')
    pickle.dump(R, file)
    file.close()
    
    print place ,'done.'

action('City of Bristol')

# def start_process():
#     print 'Starting', multiprocessing.current_process().name

# if __name__ == '__main__':
#     pool_size = multiprocessing.cpu_count() * 1

#     try:
#         pool = multiprocessing.Pool(processes=pool_size,
#                                     initializer=start_process,
#                                     )
#         pool_outputs = pool.map(action, p.names)
#         pool.close() # no more tasks
#         pool.join()  # wrap up current task
#         print 'Pool closed and joined normally.'
#     except KeyboardInterrupt:
#         print 'KeyboardInterrupt caught in parent...'
#         pool.terminate()
#         pool.join()  # wrap up current task
#         print 'Pool terminated and joined due to an exception'