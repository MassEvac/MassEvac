from init import db,abm
reload(db),reload(abm)

import os
import time
import logging
import pandas

sim = 'bristol25'
log_file = 'logs/{0}.txt'.format(time.ctime())

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

# Always make sure that my_cities is a list, not a plain string!
# eg. when doing just one place, do [10:11] rather than just [10]
places = sorted(abm.Places(sim).names)

# places.remove('Birmingham')

"""
['Aberdeen City',
 'Amber Valley',
 'Arun',
 'Birmingham',
 'Bridgend',
 'Bromsgrove',
 'Caerphilly',
 'Charnwood',
 'Chiltern',
 'Chorley',
 'City of Bristol',
 'City of Edinburgh',
 'Dacorum',
 'Darlington',
 'Fylde',
 'Great Yarmouth',
 'Guildford',
 'Hart',
 'Kettering',
 'Lewes',
 'Medway',
 'Mole Valley',
 'Newcastle-under-Lyme',
 'Newport',
 'North East Derbyshire',
 'North East Lincolnshire',
 'North Warwickshire',
 'North West Leicestershire',
 'Redcar and Cleveland',
 'Renfrewshire',
 'Rochford',
 'Rotherham',
 'Sefton',
 'Solihull',
 'Stockton-on-Tees',
 'Swindon',
 'Tandridge',
 'Telford and Wrekin',
 'Thurrock',
 'Tonbridge and Malling',
 'Warrington',
 'Warwick District',
 'West Dunbartonshire',
 'Wigan',
 'Windsor and Maidenhead',
 'Wirral District',
 'Wokingham',
 'Wyre Forest',
 'Yell',
 'York']
"""

# Iterate through these scenarios
scenarios = ['k7', 'k6', 'k5', 'k7-idp', 'k6-idp', 'k5-idp']

# Check if a job is complete
def not_complete(place,scenarios):
    not_complete = scenarios[:]
    try:
        hdf = pandas.HDFStore('metadata/bristol25/{}/agents.hdf'.format(place))
        for s in scenarios:
            if '/{}'.format(s) in hdf.keys():
                not_complete.remove(s)
        hdf.close()
    except IOError:
        pass
    return not_complete        

# Save each exit to a different file
def job(place):
    '''
    Multiprocessing runs this subroutine for each place in parallel

    Parameters
    ----------

    place : string
        A string that describes name of place boundary in OSM.

    Returns
    -------

    The number of successfully completed scenarios in this run.
    '''

    try:
        nc = not_complete(place,scenarios)
        if nc:
            s = abm.Sim(sim=sim,place=place,fresh=False,speedup=60)
            s.scenarios = nc
            r = s.run_by_agent(rerun=True,metadata=True,log_events_percent=5)
            return r
    except KeyboardInterrupt:
        print 'KeyboardInterrupt caught in child...'
    except Exception as e:
        logger.critical(place)
        logger.exception(e)
        print 'Exception caught in run.py'
        print e
        return 0

# job('City of Bristol')
# places.remove('City of Bristol')

import multiprocessing

def start_process():
    '''
    Multiprocessing calls this before starting.
    '''
    print 'Starting', multiprocessing.current_process().name

# Start the multiprocessing unit
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

print '----------------------------------------------------------------------'
print 'Summary (City, Scenarios complete)'
print '----------------------------------------------------------------------'

total = 0
for p in abm.Places('bristol25').names:
    for s in not_complete(p,scenarios):
        print p,s
        total += 1
print total, 'not complete.'        
