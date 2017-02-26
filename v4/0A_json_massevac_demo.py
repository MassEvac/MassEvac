import sys
sys.path.append('core')
import abm, db
reload(abm)
reload(db)

import logging
from collections import OrderedDict

sim = 'bristol50'

logging.basicConfig(filename='logs/{0}.log'.format(sim),level=logging.DEBUG)

places = abm.Places(sim).names

description = {'ff':'Normal Scenario','ia':'Evacuation Scenario'}

city = 'City of Bristol'
self = abm.Sim(sim,city)

# Save each exit to a different file
def do_calculations(city):
    try:
        return_code = 1
        # self = abm.Sim(sim,city)
        content = OrderedDict()
        t90 = {}

        for scenario in ['ff','ia']:
            self.scenarios = [scenario]
            self.run()
            content['Video: {0}'.format(description[scenario])]='{0}.mp4'.format(self.df_agents_file)
            content['Histo: {0}'.format(description[scenario])]='{0}-et.png'.format(self.df_agents_file)
            t90[scenario] = self.et_stats()[3]
            # self.et_figure(lognorm=True)
        return {'content':content,'lon':(self.h.l+self.h.r)/2,'lat':(self.h.t+self.h.b)/2,'pop':self.n,'t90_ia':t90['ia'],'t90_ff':t90['ff']}
    except KeyboardInterrupt:
        print 'KeyboardInterrupt caught in child...'
        return
    except (ValueError, AttributeError, NameError, IOError) as e:
        print e
        print '{0}: {1}'.format(city, e)
        logging.critical(city)
        logging.exception(e)
        return 0

sim='bristol50'
do_calculations(city)

import multiprocessing

def start_process():
    print 'Starting', multiprocessing.current_process().name

if __name__ == '__main__':
    pool_size = multiprocessing.cpu_count() * 1

    try:
        pool = multiprocessing.Pool(processes=pool_size,
                                    initializer=start_process,
                                    )
        pool_outputs = pool.map(do_calculations, places)
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
for city, info in zip(places,pool_outputs):
    if info['pop']> 100000:
        data[city] = info

# <codecell>

import json

with open('json/file.json', 'w') as outfile:
  json.dump(data, outfile,indent=True)

