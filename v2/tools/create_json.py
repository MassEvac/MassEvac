import abm
import os
import time 
import logging 
import cities
import sys
import collections

sim = 'density'
logging.basicConfig(filename='logs/{0}.log'.format(sim),level=logging.DEBUG)

# <codecell>

my_cities = cities.cities

# <codecell>

description = {'ff':'No Interaction, No Intervention (NN)','ia':'With Interaction, No Intervention (IN)','cd':'With Interaction, With Intervention (II)'}

# <codecell>

# Save each exit to a different file
def do_calculations(city):
    try:
        return_code = 1
        run = abm.ABM(sim=sim,city=city,p_factor=cities.p_factor,loadpath=False,n=None,destins=None)
        content = collections.OrderedDict()
        t90 = {}
        
        for scenario in ['ff','ia','cd']:
            if run.init_file(scenario=scenario):
                print run.df_agents_file
                sys.stdout.flush()
                
                # Only if the results cannot be loaded,
                # which means that the simulation is not complete,
                # run the simulation!
                if run.load_results():
                    content['Video: {0}'.format(description[scenario])]='{0}.mp4'.format(run.df_agents_file)
                    content['Histo: {0}'.format(description[scenario])]='{0}-et.png'.format(run.df_agents_file)
                    t90[scenario] = run.et_stats()[3]
                    # run.et_figure(lognorm=True)
            else:
                raise ValueError('Output files cannot be initialised for some reason (e.g. no destinations available).')
        centroid = run.p.nodes[run.p.centroid,:]
        IN_NN = int(100*t90['ia']/t90['ff'])
        II_NN = int(100*t90['cd']/t90['ff'])
        return {'content':content,'lon':centroid[0],'lat':centroid[1],'pop':run.n,'IN_NN':IN_NN,'II_NN':II_NN}        
    except KeyboardInterrupt:
        print 'KeyboardInterrupt caught in child...'
        return
    except (ValueError, AttributeError, NameError, IOError) as e:
        print e
        print '{0}: {1}'.format(city, e)
        logging.critical(city)
        logging.exception(e)
        return 0

import multiprocessing

def start_process():
    print 'Starting', multiprocessing.current_process().name

if __name__ == '__main__':
    pool_size = multiprocessing.cpu_count() * 1

    try:
        pool = multiprocessing.Pool(processes=pool_size,
                                    initializer=start_process,
                                    )
        pool_outputs = pool.map(do_calculations, my_cities)
        pool.close() # no more tasks
        pool.join()  # wrap up current task          
        print 'Pool closed and joined normally.'
    except KeyboardInterrupt:
        print 'KeyboardInterrupt caught in parent...'
        pool.terminate()
        pool.join()  # wrap up current task          
        print 'Pool terminated and joined due to an exception'

# <codecell>

data = {}
for city, info in zip(my_cities,pool_outputs):
    if info['pop']> 100000:
        data[city] = info

# <codecell>

import json

with open('file.json', 'w') as outfile:
  json.dump(data, outfile,indent=True)

