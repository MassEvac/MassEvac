from init import db,abm
reload(db),reload(abm)

import os
import time
import logging

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
places = abm.Places(sim).names

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
# scenarios = ['k5','k6','k7']
# scenarios = ['k5-invdistprob']

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
        h = db.Highway(place=place)
        h.fig_highway()
        # s.scenarios = scenarios
        return 1
    except KeyboardInterrupt:
        print 'KeyboardInterrupt caught in child...'
    except Exception as e:
        logger.critical(place)
        logger.exception(e)
        print 'Exception caught in run.py'
        print e
        return 0

# job('Rochford')

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
for p, po in zip(places, pool_outputs):
    if po < 1:
        print "***"
    print p,po