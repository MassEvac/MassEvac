# Gryff Holland
# MassEvac v4
import db
import pdb
import pickle
import os
import random
import math
import hashlib
import six
import time
import logging
import shutil
import gzip
import numpy as np
import networkx as nx
import scipy.stats as ss
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Best estimate of population multiplier
# Assumes that population growth spread is uniform across the UK\
p2000 = 58.459
# Source: http://en.wikipedia.org/wiki/List_of_countries_by_population_in_2000
p2015 = 63.935
# Source: http://en.wikipedia.org/wiki/List_of_countries_by_future_population_(Medium_variant)
p_factor = p2015/p2000

class FundamentalDiagram:
    def __init__(self,speedup):
        ''' This class defines the fundamental diagram.

            Inputs
            ------
                speedup: float
                    Factor to widen the simulation speed by in seconds
            Properties
            ----------
                self.speedup: float
                    The specified speedup factor
                self.k_vmin: float
                    The density cap threshold for minimum velocity
                self.v_ff: float
                    The free flow velocity
                self.bins: int
                    Number of bins to allocate simulation results
                self.bin_mean: list
                    The mean bin density for each bin
                self.v: dict
                    Dictionary lookup of velocity using parameter density
                self.k_lim: float
                    Limit density
                self.KOpt: float
                    Optimum density which the flow is maximum
        '''
        # Free flow velocity (m/s)
        self.v_ff = 1.34
        # Flat density
        self.k_vmin = 4.4
        # Density cap
        self.k_lim = 9.0
        # Congested flow (agent/m^2)
        # Must be less than this
        # This value CANNOT be looked up in v_dict[k]!
        # Obtain the speedup factor
        self.speedup = speedup
        # Number of decimal places in the velocity lookup dictionary v_dict[k]
        self.dp = 4
        # Precompute results - 14x faster to use dict
        self.precomputation()
        # Labels for the figures
        self.fontsize = 20
        self.densityLabel = '$\mathrm{k \ [ped/m^2]}$'
        self.velocityLabel = '$\mathrm{v \ [m/s]}$'
        self.flowLabel = '$\mathrm{Q \ [ped/(ms)]}$'
        
    def precomputation(self):
        # Create a list of density
        self.k = [float(i)/10**self.dp for i in range(int(self.k_lim*10**self.dp))]
        # Maximum density that can be looked up
        self.k_max = max(self.k)
        # Create a list of velocity
        self.v = [self.velocity(k) for k in self.k]
        # Create a list of flow
        self.q = [v*k for v,k in zip(self.v,self.k)]
        # Velocity lookup dictionary
        self.v_dict = dict(zip(self.k,self.v))
        # Optimum density where flow is maximum
        self.k_opt = self.k[self.q.index(max(self.q))]

    # This should form the basis for calculation of speed from density using,
    def velocity(self, k):
        ''' Calculate velocity from the input density within the boundary limits.
        '''
        if k == 0:
            # Return free flow velocity if k = 0.0 agents per metre square
            v = self.v_ff
        else:
            # Assume velocity is constant at density over 4.4 ped/m^2
            if k > self.k_vmin:
                k = self.k_vmin
            # Weidmann 1993
            v = self.v_ff*(1.0-np.exp(-1.913*(1.0/k-1.0/5.4)))
        return v * self.speedup
    
    def figure(self):
        x_offset = 0.5
        y_offset = 0.2
        k = self.k
        v = [v/self.speedup for v in self.v]
        q = [q/self.speedup for q in self.q]
        fig, (ax1,ax2) = plt.subplots(2, sharex=False, figsize=(8,8))        
        ax1.plot(k,v,'r-',linewidth=4,label='$\mathrm{v_{max} \ = \ %0.2f \ [m/s]}$'%max(v))
        ax1.set_xlim(0,self.k_max+x_offset)
        ax1.set_ylabel(self.velocityLabel,fontsize=self.fontsize)
        ax1.set_ylim(0,self.v_ff+y_offset)
        #ax.legend(loc=2,fontsize=self.fontsize)
        ax2.plot(k,q,'g--',linewidth=4,label='$\mathrm{Q_{max} = %0.2f \ [ped/(ms)]}$'%max(q))
        ax2.set_xlim(0,self.k_max+x_offset)
        ax2.set_xlabel(self.densityLabel,fontsize=self.fontsize)
        ax2.set_ylabel(self.flowLabel,fontsize=self.fontsize)
        ax2.set_ylim(min(q),max(q)+y_offset)

        ax1.axvline(self.k_opt,linestyle='-.',linewidth=2,label='$\mathrm{{k_{opt} = %0.2f} \ [ped/m^2]}$'%self.k_opt)
        ax2.axvline(self.k_opt,linestyle='-.',linewidth=2)        

        # ax1.axvline(self.k_vmin,c='g',linestyle='-.',linewidth=2,label='$\mathrm{{k_{v,min} = %0.2f} \ [ped/m^2]}$'%self.k_vmin)
        # ax2.axvline(self.k_vmin,c='g',linestyle='-.',linewidth=2)

        # ax1.axvline(self.k_lim,c='r',linestyle='-.',linewidth=2,label='$\mathrm{{k_{lim} = %0.2f} \ [ped/m^2]}$'%self.k_lim)        
        # ax2.axvline(self.k_lim,c='r',linestyle='-.',linewidth=2)

        ax1.legend(loc=1)
        ax2.legend(loc=1)

        plt.gcf().subplots_adjust(bottom=0.15)
        plt.savefig('figs/fd.pdf')
        return fig

# Import fundamental diagram
fd = FundamentalDiagram(speedup=60)

''' Scenario Definitions
    --------------------
        ff
            Free flow - no agent interaction
        ia
            Interaction mode - agents interact with each other
        df
            Density factor
        label
            Scenario label
'''
settings = {
    'ia':{
        'df':1.0,
        'label':'With Interaction, No Intervention',
    },
    'ff':{
        'df':0.0,
        'label':'Free flow',
    },
}

class Places:
    def __init__(self,sim,fresh=False):
        ''' Class to query list of places that fit within an input criteria.

            Inputs
            ------
                sim: String
                    String that contains name of simulation a query belongs to.
                fresh: Boolean
                    If true, generates a new cache of query result.
            Properties
            ----------
                self.osm_id: list of int
                    OSM id of the places in the list.
                self.names: list of String
                    Name of the places.
                self.admin_level: int
                    Administrative level on OSM.
                self.area: float
                    Area of the place.
                self.polygon_count: int
                    Number of points on the boundary polygon
        '''
        self.sim = sim
        self.fresh = fresh
        self.queries = {
                    # +/- 25% area that of 'City of Bristol' where 235816681.819764 is the area.
                    # Single polygon
                    # Less than admin level 10
                    'bristol25':'''SELECT * FROM
                        (SELECT p.osm_id, p.name, p.admin_level, SUM(ST_Area(p.way,false))  AS area, COUNT(p.osm_id) AS polygon_count FROM planet_osm_polygon AS p WHERE boundary='administrative' GROUP BY p.osm_id, p.name,p.admin_level ORDER BY admin_level, area DESC) AS q
                        WHERE polygon_count = 1 AND CAST(admin_level AS INT) < 10 AND name != '' AND area != 'NaN' AND area BETWEEN 235816681.819764*3/4 AND 235816681.819764*5/4'''
                    }
        fname = 'places/{0}'.format(self.sim)
        if os.path.isfile(fname) and not self.fresh == True:
            print 'Loading {0}'.format(fname)
            with open(fname, 'r') as f:
                self.result=pickle.load(f)
        else:
            print 'Processing {0}'.format(fname)            
            self.result = db.Query(self.queries[sim]).result
            print 'Writing {0}'.format(fname)
            with open(fname, 'w') as f:
                pickle.dump(self.result, f)
        self.osm_id,self.names,self.admin_level,self.area,self.polygon_count = zip(*self.result)

class Agent:
    def __init__(self,id,initial_edge,initial_position):
        ''' Agent class.'''
        self.id = id
        self.initial_edge = initial_edge
        self.initial_position = initial_position
        self.edge = initial_edge
        self.position = initial_position
        self.in_buffer = False
    def __repr__(self):
        return '[A{} P{:0.2f} E{}]'.format(self.id,self.position,self.edge)

class Track:
    def __init__(self,events,G):
        self.events = events[:]
        self.cursor = 0
        self.G = G
        self.update()        

    def update(self):
        self.this_time, self.this_node = self.events[self.cursor]
        self.that_time, self.that_node = self.events[self.cursor+1]
        self.this_position = np.array(self.G.node[self.this_node])
        self.that_position = np.array(self.G.node[self.that_node])
        self.velocity = (self.that_position-self.this_position)/(self.that_time-self.this_time)
        self.speed = self.G[self.this_node][self.that_node]['distance']/(self.that_time-self.this_time)/fd.speedup

    def position(self,time):
        while True:
            # The most likely scenario is that cursor hasn't moved
            if time >= self.this_time and time < self.that_time:
                return self.this_position+self.velocity*(time-self.this_time)
            # Cursor has moved up is the next likely scenario
            if time >= self.that_time:
                if self.cursor == len(self.events)-2:
                    return False
                else:
                    self.cursor += 1
            # Sometimes we might want to backtrack                    
            elif time < self.this_time:
                if self.cursor == 0 or time < 0:
                    return False
                else:
                    self.cursor -= 1
            # Since we only arrive here if the cursor has moved, update the position
            self.update()

class Sim:
    def __init__(self,sim,place,fresh=False):
        ''' Queries the population raster table on PostgreSQL database.
        
            Inputs
            ------
                sim: string
                    String that is a name of the folder where we save the simulation results
                place: string or tuple (xmin, ymin, xmax, ymax)
                    Name of the polygon on OpenStreetMap being queried
                    Alternatively, input tuple with boundary box coordinates
                fresh: boolean
                    False: (default) Read processed highway graph from cache file
                    True: Read and construct highway graph from the database (may take longer)
        '''        
        self.sim = sim
        self.place = str(place)
        self.fresh = fresh
        # Current scenario
        self.scenario = None
        # List of scenarios available
        self.scenarios = settings.keys()
        # Folder to store the simulation and logging folder
        self.folder = 'abm/{0}/{1}'.format(self.sim,self.place)
        if not os.path.isdir(self.folder):
            os.makedirs(self.folder)
        self.log_folder = '{0}/logs'.format(self.folder)
        if not os.path.isdir(self.log_folder):
            os.makedirs(self.log_folder)
        # Name of log file
        self.log_file = '{0}/{1}.log'.format(self.log_folder,time.ctime())
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        # create a file handler
        handler = logging.FileHandler(self.log_file)
        handler.setLevel(logging.INFO)
        # create a logging format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        # add the handlers to the logger
        self.logger.addHandler(handler)
        # Place Initialisation
        # --------------------
        # Init highway
        self.h = db.Highway(place,fresh=self.fresh)
        # Mark the blocked edges, If True, reduces the agent velocity to vMin on that edge (doesn't affect edge capacity)
        for u,v in self.h.G.edges_iter():
            self.h.G[u][v]['blocked'] = False
        # Agent Population Dependent Initialisation
        # -----------------------------------------
        # Init population
        self.p = db.Population(place,fresh=self.fresh)
        # If the number of agents is not defined, determine the population by multiplying the 2000 population by p_factor
        self.n = int(sum(self.p.pop) * p_factor)
        # Use buffer
        self.use_buffer = False
        
    def destin_hash(self):
        ''' Generate a hash for the list of destins.
        '''
        if self.h.destins:
            destins_text = str.join('\n',[str(x) for x in sorted(self.h.destins)])
            this_hash = hashlib.md5(destins_text).hexdigest()[:3]
            # Will have to check if the file already exists in the future
            with open('{0}/{1}.destins'.format(self.folder,this_hash), 'w') as file:
                file.write(destins_text)
            return this_hash
        else:
            raise ValueError('{0}[{1}]: Agent file could not be made because destin nodes are not available!'.format(self.place,self.scenario))
    
    def log_print(self,message):
        ''' Logs the input message into a simulation specific log file.

            Input
            -----
                message: String
        '''        
        line = '[{0}:{1}] {2}'.format(self.scenario,self.place,message);
        self.logger.info(line)
        print line
    
    def init_scenario(self,scenario):
        ''' Set the current scenario to the input scenario label.
        '''
        self.scenario = scenario
        self.log_print('Current scenario set to ({0}).'.format(settings[self.scenario]['label']))

        # Count the number of blocked
        self.agents_file = '{0}/n={1}.d={2}'.format(self.folder,self.n,self.destin_hash())
    
        # Name the scenario file
        self.scenario_file = '{0}/{1}'.format(self.agents_file,scenario)

        # Name the events file
        if not os.path.isdir(self.scenario_file):
            os.makedirs(self.scenario_file)

        self.events_file = '{0}/events'.format(self.scenario_file)

    def save_results(self):
        ''' Save the results to result files.
        '''
        fname = '{0}/T'.format(self.scenario_file)
        self.log_print('Writing {0}'.format(fname))
        with open(fname, 'w') as file:
            pickle.dump(self.T, file)
        if self.tracked_agent:
            fname = '{0}/tracked_agent'.format(self.scenario_file)
            self.log_print('Writing {0}'.format(fname))
            with open(fname, 'w') as file:
                pickle.dump(self.tracked_agent, file)
    
    def load_results(self):
        ''' Load the results from result files.
        
            Output
            ------
                success: Boolean    
                    True  - if successful
                    False - if unsuccessful
        '''
        fname = '{0}/T'.format(self.scenario_file)
        if os.path.isfile(fname):
            self.log_print('Loading {0}'.format(fname))
            with open(fname, 'r') as file:
                self.T = pickle.load(file)
            return True
        else:
            self.log_print('There are no results for this scenario.')            
            return False
    
    def init_agents(self):
        ''' Function to initialise agent properties.
        '''
        # NOTE: Need to be able to raise error if no destination
        for u,v,d in self.h.G.edges_iter(data=True):
            # Determine number of agents that are allowed on a given edge
            self.h.G[u][v]['capacity'] = int(fd.k_max*d['area'])
            # Initialise the queue length
            self.h.G[u][v]['queue_length'] = 0
            # Initialise the buffer length
            self.h.G[u][v]['buffer_length'] = 0
        # The following initial state of agents takes time to compute, save to cache
        # --------------------------------------------------------------------------
        fname = '{0}/initial_state'.format(self.agents_file)
        if os.path.isfile(fname) and not self.fresh:
            self.log_print('Loading {0}'.format(fname))
            with open(fname, 'r') as file:
                initial_state = pickle.load(file)
        else:
            edgelist = self.h.G.edges()
            creation_loop = 0
            agent_id = 0
            initial_state = {}
            while agent_id < self.n:
                self.log_print('Entering creation loop {0}. {1} of {2} agents created.'.format(creation_loop,agent_id,self.n))
                # Randomly shuffle the order in which edges are called
                random.shuffle(edgelist)
                for u,v in edgelist:
                    try:
                        # Proceed if nearest_destin is available for the edge
                        self.h.G[u][v]['nearest_destin']
                        # This is how many agents are supposed to be on this edge
                        # Function that rounds by using probability based on decimal places of the number.
                        # E.g. 6.7 has a 70% probability of being rounded to 7 and 30% probability of being rounded to 6.
                        x = self.h.G[u][v]['pop_dist']*self.n
                        x = int(x) + (random.random() < x - int(x))
                        # Number of agents left to create
                        total_capacity = self.n - agent_id
                        if x > total_capacity:
                            x = total_capacity
                        # Initialise the edge to 0 if it hasnt been yet
                        try:
                            initial_state[(u,v)]
                        except KeyError:
                            initial_state[(u,v)] = 0
                        # Check capacity of the link
                        link_capacity = self.h.G[u][v]['capacity']-initial_state[(u,v)]
                        if x > link_capacity:
                            x = link_capacity
                        initial_state[(u,v)] += x
                        agent_id += x
                        if agent_id==self.n:
                            break
                    except KeyError:
                        pass
                creation_loop += 1
            self.log_print('Initial state of {0} agents established in {1} loop(s)!'.format(agent_id,creation_loop))
            self.log_print('Writing {0}'.format(fname))
            with open(fname, 'w') as file:
                pickle.dump(initial_state, file)
        # Construct the agents
        agent_id = 0
        self.agents = []
        self.agents_per_tstep = {}    
        self.events = []    
        for u,v in initial_state:
            number_of_agents_in_this_edge = initial_state[(u,v)]
            for i in range(number_of_agents_in_this_edge):
                # When a new agent is introduced, the position is proportional to the order of the agent
                # Position is a value between 0 and 1 which shows how the agent is progressing through an edge
                # position = 1 means agent is at u
                # position = 0 means agent is at v
                position = float(i+1)/number_of_agents_in_this_edge
                agent = Agent(agent_id,initial_edge=(u,v),initial_position=position)
                # Determine the node that the agent is travelling to
                # Save the initial position of the agent
                # Add to the number of agents on that link
                self.h.G[u][v]['queue_length'] += 1
                density = self.density(self.h.G[u][v])
                velocity = fd.v_dict[density]
                agent.last_action_time = 0
                agent.action_time = position*self.h.G[u][v]['distance']/velocity
                try:
                    self.agents_per_tstep[int(agent.action_time)].append(agent)
                except KeyError:
                    self.agents_per_tstep[int(agent.action_time)] = [agent]
                self.agents.append(agent)
                # What was the time when agent was at u?
                # This is likely to go into negatives
                time_at_u = (position-1)*self.h.G[u][v]['distance']/velocity
                self.events.append([(u,round(time_at_u,2))])
                agent_id += 1
        # If use of Journey Time is enabled, construct a journey time matrix.
        # It is currently not being used as it is not fully mature.
        if False:
            # Construct a profile of journey time based on number of agents on various links
            for u,v,d in range(self.h.G.edges_iter(data=True)):
                velocity = fd.v_dict[self.density(d)]
                self.h.G[u][v]['journey_time'] = d['distance']/velocity
    
    def run(self,rerun=False,agent_progress_bar=False):
        ''' Function to animate the agents.
        '''
        success = 0
        # Load this as all scenarios share the same EM EA and route
        # This is within run_sim because they are only required for the simulation.
        self.h.init_route()
        # Iterate through scenarios
        for scenario in self.scenarios:
            # Initiate the given scenario
            self.init_scenario(scenario)

            if self.load_results() and not rerun:
                self.log_print('Scenario has already been simulated with these parameters!')
            else:
                self.init_agents()
                self.log_print('Starting this simulation...')                
                self.sim_complete = False                  

                self.tstep = 0
                self.tstep_length = 1        

                self.agents_left = self.n

                print '\n Time: {:0.4f} Agents Left: {}'.format(self.tstep, self.agents_left)

                # If the events folder exists, remove and create a new one
                if os.path.isdir(self.events_file):
                    shutil.rmtree(self.events_file)
                os.makedirs(self.events_file)

                # Start the timer
                start_time = time.time()
                while self.agents_left:
                    # Event log for this timestep
                    self.loop_this = True
                    try:
                        random.shuffle(self.agents_per_tstep[self.tstep])
                    except KeyError:
                        # There are no agents awaiting action in this timestep so pass
                        self.loop_this = False
                    if self.loop_this:
                        print 'Agents acting in tstep {}: {} of {}'.format(self.tstep,len(self.agents_per_tstep[self.tstep]),self.agents_left)
                        # Loop through every agent due action in this timestep
                        for self.agent in self.agents_per_tstep[self.tstep]:
                            # Loop until the agent action_time exceeds the sim_time
                            self.add_agent = True
                            while self.agent.action_time < self.tstep + self.tstep_length:
                                # Determine the edge that the agent is on
                                u,v = self.agent.edge
                                edge = self.h.G[u][v]
                                # print self.agent.action_time, self.tstep
                                # If this is the last edge
                                if v == edge['nearest_destin']:
                                    # Decrement the number of agents left
                                    self.agents_left -= 1
                                    # Remove the agent from the queue
                                    edge['queue_length'] -= 1
                                    if self.use_buffer and self.agent.in_buffer:
                                        edge['buffer_length'] -= 1
                                        self.agent.in_buffer = False
                                    self.agent.destin = v
                                    # Log the time spent on last edge
                                    self.events[self.agent.id].append((v,round(self.agent.action_time,2)))
                                    # Print the simulation time
                                    if agent_progress_bar:
                                        print '\n Time: {:0.4f} Agents Left: {}'.format(self.agent.action_time, self.agents_left)
                                    self.add_agent = False
                                    # print 'break 1'
                                    break
                                else:
                                    # Determine the new edge
                                    new_u = v
                                    new_v = self.h.route[edge['nearest_destin']][v]
                                    new_edge = self.h.G[new_u][new_v]
                                    # Only move the agent if there is capacity in the new edge
                                    if new_edge['capacity'] > new_edge['queue_length']:
                                        # Remove the agent from the old edge
                                        edge['queue_length'] -= 1
                                        if self.use_buffer and self.agent.in_buffer:
                                            edge['buffer_length'] -= 1
                                            self.agent.in_buffer = False
                                        # Add agent to the new edge
                                        new_edge['queue_length'] += 1
                                        # Log the time spent on last edge
                                        self.events[self.agent.id].append((v,round(self.agent.action_time,2)))
                                        # Determine the next time to take action depending on:
                                        #   - Link length
                                        #   - Link density
                                        self.agent.action_time += new_edge['distance']/fd.v_dict[self.density(new_edge)]
                                        # Assign new edge to the agent
                                        self.agent.edge = (new_u,new_v)
                                    else:
                                        # If there is no capacity, wait till the next time step                                        
                                        self.agent.action_time += self.tstep_length
                                        # Use self.use_buffer to specify whether to use additional buffer time depending                                        
                                        if self.use_buffer:
                                            # If the agent is not already in buffer, add to buffer
                                            if self.agent.in_buffer is False:
                                                    self.agent.in_buffer = True
                                                    edge['buffer_length'] += 1
                                            # on the buffer ratio and length of time required to traverse the next edge. For example:
                                            # - If buffer_length/capacity in the next edge ---> 0, buffer_time ---> 0
                                            # - If buffer_length/capacity in the next edge ---> 1, buffer_time ---> random(0,new_traversal_time)
                                            # The logic is: if all agents in the next edge are waiting in the buffer,
                                            #   chance that any of the agents in this edge can move until the ones in the next
                                            #   edge have cleared is pretty low. So, the minimum time required to traverse the
                                            #   next edge depends on the density of that edge and is given by new_traversal_time.
                                            # At the moment, using the buffer seems to underestimate the time obtained without it.
                                            # - I need to think of a way to implement buffer that produces a similar result to unbuffered time.
                                            # - As such, unbuffered is a more conservative estimate.
                                            # - And as such, only proceed with using buffer only if the need for performance outweighs accuracy.
                                            self.agent.action_time += random.random()*new_edge['distance']/fd.v_dict[self.density(new_edge)]*new_edge['buffer_length']/new_edge['capacity']
                                        break
                            if self.add_agent:
                                try:
                                    self.agents_per_tstep[int(self.agent.action_time)].append(self.agent)
                                except KeyError:
                                    self.agents_per_tstep[int(self.agent.action_time)] = [self.agent]
                    # Delete list of agents that were processed in this timestep to free up memory
                    # try:
                    #     del self.agents_per_tstep[self.tstep]
                    # except KeyError:
                    #     pass
                    # Increment timestep
                    self.tstep += self.tstep_length
                # Determine the execution time
                self.execution_time = time.time()-start_time
                print 'Execution took {:0.3f} seconds.'.format(self.execution_time)
                # Log events

                self.tracks_file(mode='cache')

                success += 1
        return success

    def tracks_file(self,mode):
        ''' Read cache events or load tracks 
        '''
        fname = '{}.txt.gz'.format(self.events_file)
        if mode == 'cache':
            to_write = ''
            for id,event in enumerate(self.events):
                for node,tstep in event:
                    to_write += '{}:{} '.format(tstep,node)
                to_write += '\n'
            with gzip.open(fname,'wb') as file:
                file.write(to_write)
        elif mode == 'load':
            self.tracks = []
            with gzip.open(fname,'rb') as file:
                for line in file.readlines():
                    this = []
                    for pair in line.split(' '):
                        if not pair == '\n':
                            tstep,node = pair.split(':')
                            this.append((float(tstep),int(node)))
                    self.tracks.append(Track(this,self.h.G))

    def dist2exit(self,agent):
        ''' Function to calculate agent distance to exit. Input is the agent index.
        '''
        x = self.X[agent]
        e = self.E[agent]
        l = self.L[agent]
        n1,n2,d = self.h.edges[e]
        return d['distance'] - l + self.h.route_length[x][n2]

    def density(self,edge):
        ''' Function to calculate link density for the input edge.
            If df = 0, returns 0 density so that we can determine free flow evacuation time.
        '''
        if edge['blocked']:
            k = fd.k_max
        else:
            k = settings[self.scenario]['df']*edge['queue_length']/edge['area']
        return round(k,fd.dp)


class Viewer:
    def __init__(self,sim,t0=0.0,tstep=0.5,denominator=50,video=True):
        ''' Sim tracks viewer.
        
            Inputs
            ------
                sim: Sim object
                    Sim object as defined by class Sim
                t0: float
                    Time at which to start animating the simulation
                tstep: float
                    Amount of timestep to lapse between frames
                denominator: int
                    Show 1 in number of agents indicated by the denominator
                    eg. 1 = all agents, 10 = 1 in 10 agents
        '''                
        self.s = sim
        try:
            self.s.tracks
        except AttributeError:
            self.s.tracks_file(mode='load')


        self.t0 = t0
        self.tstep = tstep
        self.denominator = denominator
        self.sample_size = len(self.s.tracks)/self.denominator
        self.sample_tracks = random.sample(self.s.tracks,self.sample_size)
        if video:
            self.video()

    def video(self):
        # Load figure background
        self.s.h.fig_highway()
        fig = plt.gcf()
        ax = plt.gca()
        # Set Axes limit
        ax.set_xlim(self.s.h.l, self.s.h.r) 
        ax.set_ylim(self.s.h.b, self.s.h.t) 
        # Load colourmap
        cm = plt.cm.get_cmap('Spectral')
        # Initialise points
        self.points = ax.scatter([0],[0],c=[0],marker='o',edgecolors='none',cmap=cm,alpha=0.5,clim=[0.1,fd.v_ff],norm=LogNorm(vmin=0.1, vmax=fd.v_ff))
        # Initialise text
        self.time_text = ax.text(0.02, 0.94, '', transform=ax.transAxes,alpha=0.5,size='large')
        # Draw colorbar and label
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", "5%", pad="3%")
        cb = fig.colorbar(self.points,cax=cax,ticks=[fd.v_dict[k]/fd.speedup for k in range(int(fd.k_lim))], format='$%.2f$')
        cb.set_label("Velocity $m/s$", rotation=90)

        plt.ion()
        plt.show()

        ani = animation.FuncAnimation(fig, self.animate, frames=self.frames(),
            interval=1000/25, blit=False, repeat=False, init_func=self.init_animation, save_count=100000)
        self.video_file = '{}/1_in_{}.mp4'.format(self.s.scenario_file,self.denominator)
        print 'Saving video to {}'.format(self.video_file)
        ani.save(self.video_file, fps=10, bitrate=2000)

    def init_animation(self):
        # Required for the blit state
        P,S = [],np.array([])
        self.points.set_offsets(P)
        self.points.set_array(S)
        self.time_text.set_text('')
        return self.points, self.time_text

    def animate(self,t):
        P,S = self.PS(t)
        self.agents_left = len(S) * self.denominator
        self.points.set_offsets(P)
        self.points.set_array(S)
        self.time_text.set_text('T:{0} A:{1}'.format(t,self.agents_left))
        return self.points, self.time_text

    def frames(self):
        t = self.t0
        self.agents_left = self.sample_size * self.denominator        
        # Generate frames until simulation has ended
        while self.agents_left > 0:
            yield t
            print t, self.agents_left        
            t = t + self.tstep

    def PS(self,t):
        try:
            P,S = zip(*np.vstack([(track.position(t),track.speed) for track in self.sample_tracks if track.position(t) is not False]))
        except ValueError:
            P,S = [],[]
        P,S = np.array(P),np.array(S)
        return P,S

    def path(self,which='time'):
        all_P = []
        all_S = []
        all_t = []
        for t in self.frames():
            P,S = self.PS(t)
            self.agents_left = len(S)
            if self.agents_left > 0:
                all_P.append(P)
                all_S.append(S)
                all_t.append(np.array([t]*self.agents_left))
        self.all_P = np.vstack(all_P)
        self.all_S = np.hstack(all_S)
        self.all_t = np.hstack(all_t)
        if which == 'time':
            colors = self.all_t
        elif which == 'speed':
            colors = self.all_S
        plt.scatter(self.all_P[:,0],self.all_P[:,1],c=colors,marker='.',edgecolors='none',alpha=0.2)
        plt.colorbar()
        plt.show()