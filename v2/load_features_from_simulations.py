import abm
import os
import time
import logging
import cities
import sys

from matplotlib import mlab
import scipy.io
import scipy.stats as ss
from collections import OrderedDict
import numpy as np

sim = 'density'
logging.basicConfig(filename='logs/{0}_features.log'.format(sim),level=logging.DEBUG)

my_cities = cities.cities

sim_features = ['population',
            'no_destin_nodes',
            'no_destin_edges',
            'destin_width',
            'mean_destin_dist',
            'std_destin_dist',
            'mean_agents_per_destin',
            'std_agents_per_destin'
            ]

df_features_template = ['T_mu_{0}',
                        'T_sigma_{0}',
                        'T_median_{0}',
                        'T_ninetieth_{0}',
                        ]

scenarios = ['ff','ia','cd']

df_features = []
for s in scenarios:
    for d in df_features_template:
        df_features.append(d.format(s))

df_ratios = [['ia','ff'],['cd','ff'],['ia','cd']]

for ratio in df_ratios:
    for template in df_features_template:
        df_features.append(template.format('_'.join(ratio)))

# Save each exit to a different file
def do_calculations(city):
    out = dict(zip(sim_features,[None]*6))
    try:
        print '{0}: Loading simulation features...'.format(city)
        sys.stdout.flush()

        run = abm.ABM(sim=sim,city=city,p_factor=cities.p_factor,loadpath=True)
        run.load_agents()

        # Prepare a list of distance from every node to their nearest destin nodes
        destin_dist = []
        for destin_path in run.p.path_length:
            destin_dist.extend([run.p.path_length[destin_path][length] for length in run.p.path_length[destin_path]])

        # Count the number of agents per destin
        X_count = dict(zip(run.p.destins,[0]*len(run.p.destins)))
        for x in run.X:
            X_count[x] = X_count[x] + 1

        out = {
                  'population': run.n,
                  'no_destin_nodes': len(run.p.destins),
                  'no_destin_edges': len(run.p.all_destin_edges),
                  'destin_width': sum([run.width[run.p.HAM.data[i]-1] for i in run.p.all_destin_edges]),
                  'mean_destin_dist': np.mean(destin_dist),
                  'std_destin_dist': np.std(destin_dist),
                  'mean_agents_per_destin': np.mean(X_count.values()),
                  'std_agents_per_destin': np.std(X_count.values()),
                }

        # Results specific evacuation time statistics
        for scenario in scenarios:
            run.init_file(scenario = scenario)

            print run.df_agents_file
            sys.stdout.flush()

            run.load_results()
            out.update(zip([template.format(scenario) for template in df_features_template],run.et_stats()))

        for ratio in df_ratios:
            for template in df_features_template:
                top = template.format(ratio[0])
                bot = template.format(ratio[1])
                name = template.format('_'.join(ratio))
                out.update({name:out[top]/out[bot]})

        return out
    except KeyboardInterrupt:
        print 'KeyboardInterrupt caught in child...'
        return
    except (ValueError, AttributeError, NameError, IOError) as e:
        print '{0}: {1}'.format(city, e)
        sys.stdout.flush()
        logging.critical(city)
        logging.exception(e)
        return out

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

features = {}
for feature in sim_features+df_features:
    feature_all_cities = []
    for city in pool_outputs:
        feature_all_cities.append(city[feature])
    features[feature]=feature_all_cities

scipy.io.savemat('analysis/features-{0}-cities-{1}.mat'.format(len(features),len(my_cities)),features)
