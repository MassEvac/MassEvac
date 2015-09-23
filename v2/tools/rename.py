# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import abm, gc, os, time, logging, shutil, cities
reload(abm)

sim = 'parallel'
logging.basicConfig(filename='logs/{0}.log'.format(sim),level=logging.DEBUG)

# Best estimate of population multiplier
# Assumes that population growth spread is uniform across the UK\
p2000 = 58.459
# Source: http://en.wikipedia.org/wiki/List_of_countries_by_population_in_2000
p2015 = 63.935
# Source: http://en.wikipedia.org/wiki/List_of_countries_by_future_population_(Medium_variant)
p_factor = p2015/p2000

# <codecell>

my_cities = cities.seemingly_done(sim)

# <codecell>

# Save each exit to a different file
def do_calculations(city):
    try:
        run = abm.ABM(sim=sim,city=city,p_factor=p_factor,speedup=60,loadpath=False,n=None,destins=None)
        for df in [0,1]:
            run.density_factor = df
            if run.make_agents_file():
                print run.df_agents_file
                print run.old_df_agents_file                
                sys.stdout.flush()
                
                old_formats = ['{0}.cache','{0}.result','{0}.mp4','{0}-et.png']
                new_formats = ['{0}.cache','{0}.result','{0}.mp4','{0}-et.png']
                
                for i,j in enumerate(old_formats):
                    try:
                        os.rename(old_formats[i].format(run.old_df_agents_file), new_formats[i].format(run.df_agents_file))
                    except OSError:
                        logging.critical('Error in {0} for {1}'.format(city,old_formats[i].format(run.old_df_agents_file)))
                
                try:
                    os.rename('{0}.EA'.format(run.old_agents_file), '{0}.EA'.format(run.agents_file))
                except OSError:
                    logging.critical('Error in {0} for {1}'.format(city,'{0}.EA'.format(run.old_agents_file)))
                
                #run.run_sim(fps=20,bitrate=4000,video=True,live_video=False)
                #run.load_agents('result')
                #run.save_agents('result')
                #run.et_figure(count=100,lognorm=1) 
            else:
                raise ValueError('No destinations present for this city.')
        return True
    except KeyboardInterrupt:
        print 'KeyboardInterrupt caught in child...'
        return
    except (ValueError, AttributeError, NameError, IOError) as e:
        logging.critical(city)
        logging.exception(e)
        return False

# <codecell>

import multiprocessing
import os
import sys

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

print zip(pool_outputs,my_cities)

