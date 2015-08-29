import abm
import db
reload(abm)
reload(db)

import numpy as np

s=abm.Sim('test','City of Bristol')
s.n=500000
s.load_agents()

def agent_times(s):
    '''Iterates through the results and gathers agent times.
    '''
    properties = {}
    for scenario in s.scenarios:
        s.scenario = scenario
        s.load_results()
        T = {}
        # Iterate through all edges and gather agent times
        for e,t in zip(s.E, s.T):
        	# Create a list of time per edge
        	try:
        		T[e].append(t)
        	except KeyError:
        		T[e] = [t]
        # Consolidate into a single edge dictionary
        for i in T:
            try:
                properties[i]
            except KeyError:
                properties[i] = {}
            properties[i]["{0}_mean_time".format(scenario)] = np.mean(T[i])
            properties[i]["{0}_stdv_time".format(scenario)] = np.std(T[i])
    return properties

properties = agent_times(s)

fname = '{0}/time.json'.format(s.agents_file())
s.h.geojson_edges('json/time.json',properties)