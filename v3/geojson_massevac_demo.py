import sys
sys.path.append('core')
import abm
reload(abm)

import os
import gzip
import time
import pickle
import logging
import multiprocessing
import collections

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
scenario = ['k5']

description = {'k5':'Evacuation (E)'}

from shapely.geometry import LineString,Point,mapping

# Save each exit to a different file
def job(place):
    try:
        s = abm.Sim(sim=sim,place=place)
        
        if s.load_agents():
            s.load_agent_meta()
            s.init_scenario(scenario=scenario)
            print s.agents_file()
            sys.stdout.flush()
            s.load_result_meta()

            detail = []
            for d in s.DX_destin.keys():
                detail.append({
                    "type": "Feature",
                    "properties": {
                        "name": place,
                        "destin": d,
                        'pop': s.n_destin[d],
                        'D10%':np.percentile(s.DX_destin[d],10),
                        'D50%':np.percentile(s.DX_destin[d],50),
                        'D90%':np.percentile(s.DX_destin[d],90),
                        'T10%':np.percentile(s.T_destin[d],10),
                        'T50%':np.percentile(s.T_destin[d],50),
                        'T90%':np.percentile(s.T_destin[d],90),                                    
                    },
                    "geometry": mapping(Point(s.h.nodes[d]))
                })
            overview = [{
                "type": "Feature",
                "properties": {
                    "name": place,
                    "pop": s.n,
                    "no_of_destins": len(s.DX_destin.keys()),
                    "video": s.video_file(),                    
                    'D10%': np.percentile(s.DX,10),
                    'D50%': np.percentile(s.DX,50),
                    'D90%': np.percentile(s.DX,90),
                    'T10%': np.percentile(s.T,10),
                    'T50%': np.percentile(s.T,50),
                    'T90%': np.percentile(s.T,90),
                },
                "geometry": mapping(s.h.boundary)
            }]
            print overview
            print detail                                       
        else:
            raise ValueError('Output files cannot be initialised for some reason (e.g. no destinations available).')
        return overview,detail
    except KeyboardInterrupt:
        print 'KeyboardInterrupt caught in child...'
    except (ValueError, AttributeError, NameError, IOError) as e:
        print e
        print '{}: {}'.format(place, e)
        logging.critical(place)
        logging.exception(e)

# overview,detail = job('City of Bristol')

import multiprocessing

def start_process():
    print 'Starting', multiprocessing.current_process().name

if __name__ == '__main__':
    pool_size = multiprocessing.cpu_count() * 1

    try:
        pool = multiprocessing.Pool(processes=pool_size,
                                    initializer=start_process,
                                    )
        pool_outputs = pool.map(job, places)
        pool.close() # no more tasks
        pool.join()  # wrap up current task          
        print 'Pool closed and joined normally.'
    except KeyboardInterrupt:
        print 'KeyboardInterrupt caught in parent...'
        pool.terminate()
        pool.join()  # wrap up current task          
        print 'Pool terminated and joined due to an exception'

overview = []
detail = []
for o,d in pool_outputs:
    overview.extend(o)
    detail.extend(d)

import json

with open('../../Sites/MassEvacDemo/overview.json', 'w') as file:
    out = {
        "type": "FeatureCollection",
        "features": overview
    }
    json.dump(out, file,indent=True)


with open('../../Sites/MassEvacDemo/detail.json', 'w') as file:
    out = {
        "type": "FeatureCollection",
        "features": detail
    }
    json.dump(out, file,indent=True)    