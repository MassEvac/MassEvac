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
import numpy as np
import networkx as nx
import scipy.stats as ss
import matplotlib.pyplot as plt
from sympy import *
from matplotlib import mlab, animation
from IPython.display import HTML
from shapely.geometry import mapping, LineString
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import LogNorm
from numpy.lib.function_base import iterable
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
                self.kCf: float
                    The density cap for the edges
                self. vFf: float
                    The free flow velocity
                self.bins: int
                    Number of bins to allocate simulation results
                self.bin_mean: list
                    The mean bin density for each bin
                self.v: dict
                    Dictionary lookup of velocity using parameter density
                self.kMax: float
                    Maximum density
                self.KOpt: float
                    Optimum density which the flow is maximum
        '''
        # Free flow velocity (m/s)
        self.vFf = 1.34
        # Congested flow (agent/m^2)
        # Must be less than this
        # This value CANNOT be looked up in v_dict[k]!
        self.kCf = 5.0
        # Number of bins to profile the density with
        self.bins = int(self.kCf)
        # This is the average density in each bin
        # Note that the last bin is 5.25 = (5+5.5)/2
        self.bin_mean = []
        for i in range(self.bins):
            self.bin_mean.append(i+0.5)
        # Obtain the speedup factor
        self.speedup = speedup
        # Number of decimal places in the velocity lookup dictionary v_dict[k]
        self.dp = 4
        # Create a list of density
        self.k = [float(i)/10**self.dp for i in range(int(self.kCf*10**self.dp))]
        # Create a list of velocity
        self.v = []
        for k in self.k:
            self.v.append(self.velocity(k))
        # Create a list of flow
        self.q=[v*k for v,k in zip(self.v,self.k)]
        # Velocity lookup dictionary
        self.v_dict = dict([(round(k,self.dp),v) for k,v in zip(self.k,self.v)])
        # Optimum density where flow is maximum
        self.kOpt = self.k[self.q.index(max(self.q))]
        # Maximum density that can be looked up
        self.kMax = max(self.k)
        # Labels for the figures
        self.fontsize = 20
        self.densityLabel = '$\mathrm{k \ [ped/m^2]}$'
        self.velocityLabel = '$\mathrm{v \ [m/s]}$'
        self.flowLabel = '$\mathrm{Q \ [ped/(ms)]}$'
        
    # This should form the basis for calculation of speed from density using,
    def velocity(self, k):
        ''' Calculate velocity from the input density within the boundary limits.
        '''
        try:
            # Weidmann 1993
            v = self.vFf*(1.0-np.exp(-1.913*(1.0/k-1.0/5.4)))            
        except ZeroDivisionError:
            # Return free flow velocity if k = 0.0 agents per metre square
            v = self.vFf
        return v * self.speedup
    
    def figure(self):
        offset = 0.1
        k = self.k
        v = [v/self.speedup for v in self.v]
        q = [q/self.speedup for q in self.q]
        fig, (ax1,ax2) = plt.subplots(2, sharex=False, figsize=(8,8))        
        ax1.plot(k,v,'r-',linewidth=4,label='$\mathrm{v_{max} \ = \ %0.2f \ [m/s]}$'%max(v))
        ax1.set_xlim(0,self.kCf+offset)
        ax1.set_ylabel(self.velocityLabel,fontsize=self.fontsize)
        ax1.set_ylim(0,self.vFf+offset)
        #ax.legend(loc=2,fontsize=self.fontsize)
        ax2.plot(k,q,'g--',linewidth=4,label='$\mathrm{Q_{max} = %0.2f \ [ped/(ms)]}$'%max(q))
        ax2.set_xlabel(self.densityLabel,fontsize=self.fontsize)                
        ax2.set_ylabel(self.flowLabel,fontsize=self.fontsize)
        ax2.set_ylim(min(q),max(q)+offset)
        #ax2.legend(loc=0,fontsize=self.fontsize)
        ax1.axvline(self.kOpt,linestyle='-.',label='$\mathrm{{k_{opt} = %0.2f} \ [ped/m^2]}$'%self.kOpt)
        ax2.axvline(self.kOpt,linestyle='-.')        
        ax1.legend()
        ax2.legend()
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.savefig('fd.pdf')
        return fig
    
    def which_bin(self,k):
        ''' Sometimes, the initial position of the agents may lead the density to exceed the
            maximum density, in which case revert the bin index to less than maximum bin index
        '''
        bin = int(k)
        if not bin < self.bins:
            bin = self.bins - 1
        return bin

# Import fundamental diagram
fd = FundamentalDiagram(speedup=60)

''' Scenario Definitions
    --------------------
        ff
            Free flow - no agent interaction
        ia
            Interaction mode - agents interact with each other
        c0
            Capped density - density capped at maximum flow
        c1
            Capped density+1 - density capped at maximum flow+1
        kCap
            Link density cap
        df
            Density factor
        label
            Scenario label
'''
settings = {
    'ia':{
        'kCap':fd.kMax,
        'df':1.0,
        'label':'With Interaction, No Intervention',
    },
    'c0':{
        'kCap':fd.kOpt,
        'df':1.0,
        'label':'With Intervention (Optimum Flow)',
    },
    'c1':{
        'kCap':fd.kOpt+1,
        'df':1.0,
        'label':'With Intervention (Optimum Flow+1)'
    },
    'ff':{
        'kCap':fd.kMax,
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
    def __repr__(self):
        return '[A{} P{:0.2f} E{}]'.format(self.id,self.position,self.edge)

class Sim:
    def __init__(self,sim,place,fresh=False):
        ''' Queries the population raster table on PostgreSQL database.
        
            Inputs
            ------
                place: string or tuple (xmin, ymin, xmax, ymax)
                    Name of the polygon on OpenStreetMap being queried
                    Alternatively, input tuple with boundary box coordinates
                fresh: boolean
                    False: (default) Read processed highway graph from cache file
                    True: Read and construct highway graph from the database (may take longer)
            Properties
            ----------
                self.lon: NumPy array
                    Array of longitudes
                self.lat: NumPy array
                    Array of latitudes
                self.pop: NumPy array
                    Array of population
        '''        
        self.sim = sim
        self.place = str(place)
        self.fresh = fresh
        # Current scenario
        self.scenario = None
        # List of scenarios available
        # self.scenarios = settings.keys()
        self.scenarios = ['ia'] # For now, just consider interaction case
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
        # Which method to use? None is default. 'dict' uses the dict method
        self.method = None

        
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

    def agents_file(self):
        ''' Return the name of the scenario specific agent file name.
        '''
        # Count the number of blocked
        blocked_count = 0
        for u,v,d in self.h.G.edges_iter(data=True):
            if d['blocked']:
                blocked_count+= 1
        # Generate blocked text if some of the roads are blocked
        if blocked_count > 0:
            blocked_text = '.b={0}'.format(blocked_count)
            if self.random_successor:
                blocked_text += '.rs'
        else:
            blocked_text = ''
        fname = '{0}/n={1}.d={2}{3}'.format(self.folder,self.n,self.destin_hash(),blocked_text)
        if not os.path.isdir(fname):
            os.makedirs(fname)
        return fname
    
    def scenario_file(self,scenario=None):
        ''' Return the name of the scenario specific agent file name.
        '''
        if scenario == None:
            scenario = self.scenario
        fname = '{0}/{1}'.format(self.agents_file(),scenario)
        if not os.path.isdir(fname):
            os.makedirs(fname)
        return fname

    def save_results(self):
        ''' Save the results to result files.
        '''
        fname = '{0}/T'.format(self.scenario_file())
        self.log_print('Writing {0}'.format(fname))
        with open(fname, 'w') as file:
            pickle.dump(self.T, file)
        if self.tracked_agent:
            fname = '{0}/tracked_agent'.format(self.scenario_file())
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
        fname = '{0}/T'.format(self.scenario_file())
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
            self.h.G[u][v]['capacity'] = int(settings[self.scenario]['kCap']*d['area'])
            # Reset the current agent count
            self.h.G[u][v]['queue'] = []
            self.h.G[u][v]['queue_length'] = 0
        # The following initial state of agents takes time to compute, save to cache
        # --------------------------------------------------------------------------
        fname = '{0}/initial_state'.format(self.agents_file())
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
        for u,v in initial_state:
            number_of_agents_in_this_edge = initial_state[(u,v)]
            for i in range(number_of_agents_in_this_edge):
                # When a new agent is introduced, the position is proportional to the order of the agent
                # Position is a value between 0 and 1 which shows how the agent is progressing through an edge
                position = float(i+1)/number_of_agents_in_this_edge
                agent = Agent(agent_id,initial_edge=(u,v),initial_position=position)
                # Determine the node that the agent is travelling to
                # Save the initial position of the agent
                # Add to the number of agents on that link
                self.h.G[u][v]['queue'].append(agent)
                self.h.G[u][v]['queue_length'] += 1
                density = self.density(self.h.G[u][v])
                velocity = fd.v_dict[density]
                agent.action_time = position*self.h.G[u][v]['distance']/velocity
                try:
                    self.agents_per_tstep[int(agent.action_time)].append(agent)
                except KeyError:
                    self.agents_per_tstep[int(agent.action_time)] = [agent]
                self.agents.append(agent)
                agent_id += 1
        # If use of Journey Time is enabled, construct a journey time matrix.
        # It is currently not being used as it is not fully mature.
        if False:
            # Construct a profile of journey time based on number of agents on various links
            for u,v,d in range(self.h.G.edges_iter(data=True)):
                velocity = fd.v_dict[self.density(d)]
                self.h.G[u][v]['journey_time'] = d['distance']/velocity
    
    def run(self,video=True,live_video=False,bitrate=4000,fps=20,rerun=False):
        ''' Function to animate the agents.
        '''
        success = 0
        # Load this as all scenarios share the same EM EA and route
        # This is within run_sim because they are only required for the simulation.
        self.h.init_route()
        # Iterate through scenarios
        for scenario in ['ia']:
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

                edge_list = self.h.G.edges(data=True)

                start_time = time.time()
                if self.method == 'dict':
                    while self.agents_left:
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
                                        self.agents_left = self.agents_left - 1
                                        # Remove the agent from the queue
                                        edge['queue_length'] = edge['queue_length'] - 1
                                        # edge['queue'].remove(self.agent)
                                        if edge['queue_length'] < 0:
                                            raise ValueError
                                        self.agent.destin = v
                                        # print self.agent
                                        # Print the simulation time
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
                                            edge['queue_length'] = edge['queue_length'] - 1
                                            # edge['queue'].remove(self.agent)
                                            # Add agent to the new edge
                                            new_edge['queue_length'] = new_edge['queue_length'] + 1
                                            if edge['queue_length'] < 0:
                                                raise ValueError
                                            # new_edge['queue'].append(self.agent)
                                            # Determine the next time to take action depending on:
                                            #   - Link length
                                            #   - Link density
                                            # print self.density(new_edge), new_u,new_v,' density'
                                            self.agent.action_time += new_edge['distance']/fd.v_dict[self.density(new_edge)]
                                            # Assign new edge to the agent
                                            self.agent.edge = (new_u,new_v)
                                        else:
                                            # If there is no capacity, wait till the next time step
                                            self.agent.action_time += self.tstep_length
                                            # print new_edge['capacity'], new_edge['queue_length']
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
                else:
                    # Loop until the number of agents_left is 0
                    while self.agents_left:
                        # Increment the timestep
                        self.tstep += 1
                        # Randomise the edge list
                        random.shuffle(edge_list)
                        # Loop through all the edges
                        for u,v,edge in edge_list:
                            # loop_edge is True if the queue is not empty
                            loop_edge = True
                            while loop_edge and edge['queue']:
                                # Loop through every agent within sim_time
                                self.agent = edge['queue'][0]
                                # If the this agent's action time exceeds the sim time,
                                # Assume that the rest of the agents of the queue do too
                                # So stop looping over this edge
                                if self.agent.action_time > self.tstep:
                                    break
                                # Loop until the agent action_time exceeds the sim_time
                                while self.agent.action_time <= self.tstep:
                                    # print agent.action_time, self.tstep
                                    # If this is the last edge
                                    if v == edge['nearest_destin']:
                                        # Decrement the number of agents left
                                        self.agents_left -= 1
                                        # Remove the agent from the queue                                    
                                        edge['queue'].remove(self.agent)
                                        self.agent.destin = v
                                        # Print the simulation time
                                        print '\n Time: {:0.4f} Agents Left: {}'.format(self.agent.action_time, self.agents_left)
                                        break
                                    else:
                                        # Determine the new edge
                                        new_u = v
                                        new_v = self.h.route[edge['nearest_destin']][v]
                                        new_edge = self.h.G[new_u][new_v]
                                        # Only move the agent if there is capacity in the new edge
                                        if new_edge['capacity'] > len(new_edge['queue']):
                                            # Remove the agent from the old edge
                                            edge['queue'].remove(self.agent)
                                            # Add agent to the new edge
                                            new_edge['queue'].append(self.agent)
                                            # Determine the next time to take action depending on:
                                            #   - Link length
                                            #   - Link density
                                            self.agent.action_time += new_edge['distance']/fd.v_dict[self.density(new_edge)]
                                            # Assign new edge to the agent
                                            u,v,edge = new_u,new_v,new_edge
                                        else:
                                            # If there is no capacity, wait till the next time step
                                            self.agent.action_time += self.tstep_length
                                            loop_edge = False
                                            break
                # Determine the execution time
                self.execution_time = time.time()-start_time
                print 'Execution took {:0.3f} seconds.'.format(self.execution_time)
                success += 1
        return success

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
            k = fd.kMax
        else:
            if self.method is 'dict':
                k = settings[self.scenario]['df']*edge['queue_length']/edge['area']
            else:
                k = settings[self.scenario]['df']*len(edge['queue'])/edge['area']
        return round(k,fd.dp)