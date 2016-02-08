import sys
sys.path.append('core')

import abm
import os
import time
import logging

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

# Always make sure that my_cities is a list, not a plain string!
# eg. when doing just one place, do [10:11] rather than just [10]
places = abm.Places(sim).names

# ['Telford and Wrekin',
#  'Caerphilly',
#  'City of Edinburgh',
#  'York',
#  'Renfrewshire',
#  'Medway',
#  'Bridgend',
#  'Redcar and Cleveland',
#  'City of Bristol',
#  'Swindon',
#  'Newport',
#  'Yell',
#  'Stockton-on-Tees',
#  'North East Lincolnshire',
#  'Windsor and Maidenhead',
#  'Darlington',
#  'Aberdeen City',
#  'Thurrock',
#  'West Dunbartonshire',
#  'Warrington',
#  'Wokingham',
#  'Lewes',
#  'North Warwickshire',
#  'Rotherham',
#  'Warwick District',
#  'North West Leicestershire',
#  'Charnwood',
#  'North East Derbyshire',
#  'Guildford',
#  'Birmingham',
#  'Amber Valley',
#  'Rochford',
#  'Mole Valley',
#  'Wirral District',
#  'Tandridge',
#  'Tonbridge and Malling',
#  'Kettering',
#  'Arun',
#  'Bromsgrove',
#  'Hart',
#  'Dacorum',
#  'Newcastle-under-Lyme',
#  'Sefton',
#  'Chorley',
#  'Chiltern',
#  'Wyre Forest',
#  'Wigan',
#  'Fylde',
#  'Great Yarmouth',
#  'Solihull']

# Iterate through these scenarios
# scenarios = ['k5','k6','k7']
scenarios = ['k5-invdistprob']

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
        s = abm.Sim(sim=sim,place=place)
        s.scenarios = scenarios
        return s.run_sim(fps=20,bitrate=2000,video=True,live_video=False)
    except KeyboardInterrupt:
        print 'KeyboardInterrupt caught in child...'
    except Exception as e:
        logger.critical(place)
        logger.exception(e)
        print 'Exception caught in run.py'
        print e
        return 0

# job(places[0])
job('City of Bristol')

# import multiprocessing

# def start_process():
#     '''
#     Multiprocessing calls this before starting.
#     '''
#     print 'Starting', multiprocessing.current_process().name

# # Start the multiprocessing unit
# if __name__ == '__main__':
#     pool_size = multiprocessing.cpu_count() * 1

#     try:
#         pool = multiprocessing.Pool(processes=pool_size,
#                                     initializer=start_process,
#                                     )
#         pool_outputs = pool.map(job, places)
#         pool.close() # no more tasks
#         pool.join()  # wrap up current task
#         print 'Pool closed and joined normally.'
#     except KeyboardInterrupt:
#         print 'KeyboardInterrupt caught in parent...'
#         pool.terminate()
#         pool.join()  # wrap up current task
#         print 'Pool terminated and joined due to an exception'

# print '----------------------------------------------------------------------'
# print 'Summary (City, Scenarios complete)'
# print '----------------------------------------------------------------------'
# for place, success in zip(places, pool_outputs):
#     if success == len(scenarios):
#         print '{0}: {1} successful scenario(s)'.format(place, success)
#     else:
#         print '  INCOMPLETE: {0}: {1} successful scenario(s)'.format(place, success)
