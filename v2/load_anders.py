import scipy.stats as ss
import cities
import massevac_v2 as me
import pickle
import numpy as np

sim_name = 'massevac'
scenario = 'ff'

def action(city):
    # Load path and the default scenario here!

    this = me.abm(sim=sim_name,city=city,scenario=scenario,p_factor=cities.p_factor,n=None,destins=None)
    this.load_agent_meta()
    this.load_result_meta()    
    
    n = this.n

    total = np.sum(this.D_tstep,axis=1)
    remaining = this.n
    
    width = np.sum(this.p.destin_width)
    DX90 = ss.scoreatpercentile(this.DX,90)
    
    removed = []    
    
    for left in total:
        removed.append(remaining-left)
        remaining = left
        
    removed.append(remaining)
    
    fname = 'analysis/anders/{0}/{1}'.format(scenario,city)
    
    file = open(fname, 'w')
    pickle.dump([removed,width,DX90,n], file)
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
