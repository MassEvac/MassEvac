import abm
import os
import time
import logging
import cities
import sys
import traceback

sim = 'massevac'
logfile = 'logs/{0}.{1}'.format(time.ctime(),sim)
logging.basicConfig(filename=logfile,level=logging.DEBUG)

# Always make sure that my_cities is a list, not a plain string!
# eg. when doing just one city, do [10:11] rather than just [10]
my_cities = cities.all#[11:12]
my_cities = ['Truro']

# Save each exit to a different file
def do_calculations(city):
    '''
    Multiprocessing runs this subroutine for each city in parallel

    Parameters
    ----------

    city : string
        A string that describes name of city boundary in OSM.

    Returns
    -------

    return_code: integer
        A number that signifies the outcome of the simulation.
            0: One of (ValueError, AttributeError, NameError, IOError)
            1: Successful
            2: Already done
    '''

    # Number of completed scenarios for this city in this run
    return_code = 0
    scenario = 'undefined'

    try:
        simulation = abm.Sim(sim=sim,city=city,p_factor=cities.p_factor,n=None,destins=None,fresh_place=False)

        # Iterate through the scenarios
        for scenario in simulation.scenarios:
            simulation.init_scenario(scenario=scenario)

            # If the results cannot be loaded, run simulation
            if not simulation.load_results():
                simulation.run_sim(fps=20,bitrate=4000,video=True,live_video=False)
            else:
                print '{0}[{1}]: This scenario has already been simulated with the given parameters!'.format(city, scenario)
                sys.stdout.flush()

            return_code += 1

        return return_code
    except KeyboardInterrupt:
        print 'KeyboardInterrupt caught in child...'
        return
    except Exception as e:
        logging.critical(city)
        logging.exception(e)
        print '{0}[{1}]: {2}'.format(city,scenario,e)
        print '----------------------------------------------------------------------'
        print 'Following has been logged to [{0}]'.format(logfile)
        print '----------------------------------------------------------------------'
        print traceback.format_exc()
        print '----------------------------------------------------------------------'
        return return_code

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
        pool_outputs = pool.map(do_calculations, my_cities)
        pool.close() # no more tasks
        pool.join()  # wrap up current task
        print 'Pool closed and joined normally.'
    except KeyboardInterrupt:
        print 'KeyboardInterrupt caught in parent...'
        pool.terminate()
        pool.join()  # wrap up current task
        print 'Pool terminated and joined due to an exception'

print '----------------------------------------------------------------------'
print '----------------------------------------------------------------------'
print 'Summary (City, Scenarios complete)'
print '----------------------------------------------------------------------'
for city, scenarios in zip(my_cities, pool_outputs):
    if scenarios == 4:
        print '{0}: {1} scenario(s)'.format(city, scenarios)
    else:
        print 'INCOMPLETE: {0}: {1} scenario(s)'.format(city, scenarios)
