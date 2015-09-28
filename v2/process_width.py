import scipy.stats as ss
import cities
import massevac as me
import pickle
import numpy as np

sim_name = 'massevac'

def action(city):
    # Load path and the default scenario here!
    sim = me.abm(sim=sim_name,city=city,p_factor=cities.p_factor,n=None,destins=None)
    sim.p.init_path()
    sim.load_agent_meta()
    sim.load_result_meta()
    sim.p.init_EM()

    # Establish maximum distances to exit
    # Using the agent distance is not suitable because we want to use the full length of the edges
    # Using full length of the edges should make the operations faster
    DX_max = {}
    for destin,edge in zip(sim.X,sim.E):
        # Edge ends at this node
        to = sim.p.DAM.col[edge] 

        # Distance to exit
        dx = sim.p.path_length[destin][to]+sim.p.DAM.data[edge] 
    
        try:
            if DX_max[destin]<dx:
                DX_max[destin] = dx
        except KeyError:
            DX_max[destin] = dx
        
    W = {}
    for destin in DX_max:
        W[destin] = [0]*int(round(DX_max[destin]))
    
    done = []

    # Edges and the destins that they belong to
    edges,index=np.unique(sim.E,return_index=True)
    destins = np.array(sim.X)[index]

    for edge,destin in zip(edges,destins):
        while edge not in done:
            # Edge begins at this node
            fr = sim.p.DAM.row[edge]    
        
            # Edge ends at this node
            to = sim.p.DAM.col[edge] 
        
            dist = sim.p.path_length[destin][to]
            leng = sim.p.DAM.data[edge]
            longest = int(round(dist+leng))
            shortest = int(round(dist))
            width = sim.p.width[sim.p.HAM.data[edge]-1]
    
            for i in range(shortest,longest):
                W[destin][i] += width
        
            done.append(edge)
        
            if to not in sim.p.destins:
                fr = to
                to = sim.p.path[destin][fr]
                edge = sim.p.EM[fr][to]
                
    file = open('analysis/width/{0}'.format(city), 'w')
    pickle.dump(W, file)
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
    
# Darn it1
