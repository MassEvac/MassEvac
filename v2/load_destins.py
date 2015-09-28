import scipy.stats as ss
import cities
import massevac as me
import pickle
import numpy as np

sim_name = 'massevac'

def action(city):
    # Load path and the default scenario here!
    common = me.abm(sim=sim_name,city=city)
    common.load_agent_meta()
    common.load_result_meta()

    N = []
    W = []

    DX90 = []

    T90ff = []
    T90 = []


    Qmeff = []
    Qmebn = []
    
    Qmdff = []
    Qmdbn = []

    Q90ff = []
    Q90bn = []    
    
    Q90 = []
    
    Qpred = []

    for n,w,d in zip(common.n_destin,common.p.destin_width,common.p.destins):
        try:
            dx90 = ss.scoreatpercentile(common.DX_destin[d],90)
                        
            t90ff = int(dx90/common.f.vMax/60)
            t90 = ss.scoreatpercentile(common.T_destin[d],90)                                    
            
            qmeff = np.mean(common.Q_destin[d][:t90ff])
            qmebn = np.mean(common.Q_destin[d][t90ff:])            

            qmdff = np.median(common.Q_destin[d][:t90ff])
            qmdbn = np.median(common.Q_destin[d][t90ff:])                
            
            q90ff = ss.scoreatpercentile(common.Q_destin[d][:t90ff],90)
            q90bn = ss.scoreatpercentile(common.Q_destin[d][t90ff:],90)

            q90 = ss.scoreatpercentile(common.Q_destin[d],90)

            qpred = n/t90ff/w
            
        except KeyError:
             dx90, t90ff,t90, qmeff, qmebn, qmdff, qmdbn, q90ff, q90bn, q90, qpred  = 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.

        N.append(n)
        W.append(w)

        DX90.append(dx90)

        T90ff.append(t90ff)
        T90.append(t90)

        Qmeff.append(qmeff)
        Qmebn.append(qmebn)        

        Qmdff.append(qmdff)
        Qmdbn.append(qmdbn)        
        
        Q90ff.append(q90ff)
        Q90bn.append(q90bn)        

        Q90.append(q90)

        Qpred.append(qpred)

    file = open('analysis/destins/{0}'.format(city), 'w')
    pickle.dump([common.p.destins,N,W,DX90,T90ff,T90,Qmeff,Qmebn,Qmdff,Qmdbn,Q90ff,Q90bn,Q90,Qpred], file)
    file.close()
    
    file = open('analysis/destins_Q/{0}'.format(city), 'w')
    pickle.dump(common.Q_destin, file)
    file.close()

    file = open('analysis/destins_T/{0}'.format(city), 'w')
    pickle.dump(common.T_destin, file)
    file.close()
    
    file = open('analysis/destins_DX/{0}'.format(city), 'w')
    pickle.dump(common.DX_destin, file)
    file.close()

import multiprocessing

def start_process():
    print 'Starting', multiprocessing.current_process().name

if __name__ == '__main__':
    pool_size = multiprocessing.cpu_count() * 1

    try:
        pool = multiprocessing.Pool(processes=pool_size,
                                    initializer=start_process,
                                    )
        pool_outputs = pool.map(action, cities.all)
        pool.close() # no more tasks
        pool.join()  # wrap up current task
        print 'Pool closed and joined normally.'
    except KeyboardInterrupt:
        print 'KeyboardInterrupt caught in parent...'
        pool.terminate()
        pool.join()  # wrap up current task
        print 'Pool terminated and joined due to an exception'
        
# CHECK: happ3
