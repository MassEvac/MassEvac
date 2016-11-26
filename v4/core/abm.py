# MassEvac v4
import db
import pdb
import pickle
import os
import pdb
import random
import math
import hashlib
import six
import time
import pandas
import logging
import shutil
import shelve
import gzip
import numpy as np
import networkx as nx
import scipy.stats as ss
import matplotlib.pyplot as plt
from operator import itemgetter
from matplotlib import animation
from collections import deque
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import LinearLocator, FormatStrFormatter

class FundamentalDiagram:
    def __init__(self,speedup,k_vmin,k_lim):
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
                self.K_opt: float
                    Optimum density which the flow is maximum
        '''
        # Free flow velocity (m/s)
        self.v_ff = 1.34
        # Flat density
        self.k_vmin = k_vmin
        # Density cap
        self.k_lim = k_lim
        # Congested flow (agent/m^2)
        # Must be less than this
        # This value CANNOT be looked up in v_dict[k]!
        # Obtain the speedup factor
        self.speedup = speedup
        # Sped up velocity
        self.v_ff_spedup = self.v_ff * speedup
        # Number of decimal places in the velocity lookup dictionary v_dict[k]
        self.dp = 4
        # Precompute results - 14x faster to use dict
        self.precomputation()
        # Labels for the figures
        self.ticksize = 20
        self.fontsize = 25
        self.densityLabel = '$k \ \mathrm{[ped/m^2]}$'
        self.velocityLabel = '$v \ \mathrm{[m/s]}$'
        self.flowLabel = '$Q \ \mathrm{[ped/(ms)]}$'
        
    def precomputation(self):
        # Create a list of density
        self.k = [float(i)/10**self.dp for i in range(int(self.k_lim*10**self.dp)+1)]
        # Maximum density that can be looked up
        self.k_max = max(self.k)
        # Create a list of velocity
        self.v = [self.velocity(k) for k in self.k]
        # Create a list of flow
        self.q = [v*k for v,k in zip(self.v,self.k)]
        # Velocity lookup dictionary
        self.v_dict = dict(zip(self.k,self.v))
        # Flow lookup dictionary
        self.q_dict = dict(zip(self.k,self.q))
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
    
    def figure(self,metrics=True):
        x_offset = 0.6
        y_offset = 0.6
        k = self.k
        v = [v/self.speedup for v in self.v]
        q = [q/self.speedup for q in self.q]
        fig, (ax1,ax2) = plt.subplots(2, sharex=False, figsize=(12,12),dpi=100)
        fig.set_tight_layout(True)
        ax1.plot(k,v,'r-',linewidth=4,label='$v=f(k)$')
        ax1.set_xlim(0,self.k_max+x_offset)
        ax1.set_ylabel(self.velocityLabel,fontsize=self.fontsize)
        ax1.set_ylim(0,self.v_ff+y_offset)
        ax1.tick_params(axis='both', labelsize=self.ticksize)
        ax1.yaxis.set_major_locator(LinearLocator(4))
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        #ax.legend(loc=2,fontsize=self.fontsize)
        ax2.plot(k,q,'g--',linewidth=4,label='$Q = k.v(k)$')
        ax2.set_xlim(0,self.k_max+x_offset)
        ax2.set_xlabel(self.densityLabel,fontsize=self.fontsize)
        ax2.set_ylabel(self.flowLabel,fontsize=self.fontsize)
        ax2.set_ylim(min(q),max(q)+y_offset)
        ax2.tick_params(axis='both', labelsize=self.ticksize)        
        ax2.yaxis.set_major_locator(LinearLocator(4))
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))        

        ax1.axhline(max(v),c='r',linestyle=':',linewidth=2,label='$v_{max} = %0.2f \ \mathrm{[m/s]}$'%max(v))
        ax2.axhline(max(q),c='g',linestyle=':',linewidth=2,label='$Q_{max} = %0.2f \ \mathrm{[ped/(ms)]}$'%max(q))
        for ax in [ax1,ax2]:
            ax.axvline(self.k_opt,c='b',linestyle='-.',linewidth=2,label='$\mathrm{{k_{opt} = %0.2f} \ [ped/m^2]}$'%self.k_opt)

        if metrics:
            fname = 'pre2015/fd-revision.pdf'        
            for k_lim,c in zip([5,6,7],['y','c','m']):
                for ax in [ax1, ax2]:
                    ax.axvline(k_lim,c=c,linestyle='-.',linewidth=2,label='$\mathrm{{k_{lim} = %0.2f} \ [ped/m^2]}$'%k_lim)
        else:
            fname = 'figs/fd-simple-kvmin-{}-klim-{}.pdf'.format(self.k_vmin,self.k_lim)
        
        ax1.legend(loc=1,fontsize=self.ticksize)
        ax2.legend(loc=1,fontsize=self.ticksize)            
        fig.savefig(fname,bbox_inches='tight')
        return fig

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
    'k5-idp':{ # Density limit of 5, uses inverse distance probability
        'path':'invdistprob',
        'df':1.0,
        'k_vmin':5,
        'k_lim':5,
        'label':'With Interaction, No Intervention',
    },
    'k6-idp':{ # Density limit of 5, uses inverse distance probability
        'path':'invdistprob',
        'df':1.0,
        'k_vmin':5,
        'k_lim':6,
        'label':'With Interaction, No Intervention',
    },
    'k7-idp':{ # Density limit of 5, uses inverse distance probability
        'path':'invdistprob',
        'df':1.0,
        'k_vmin':5,
        'k_lim':7,
        'label':'With Interaction, No Intervention',
    },
    'k5':{ # Density limit of 5
        'path':'nearest',
        'df':1.0,
        'k_vmin':5,        
        'k_lim':5,        
        'label':'With Interaction, No Intervention',
    },
    'k6':{ # Density limit of 6
        'path':'nearest',
        'df':1.0,
        'k_vmin':5,        
        'k_lim':6,        
        'label':'With Interaction, No Intervention',
    },
    'k7':{ # Density limit of 7
        'path':'nearest',
        'df':1.0,
        'k_vmin':5,        
        'k_lim':7,
        'label':'With Interaction, No Intervention',
    },
    # 'ff-idp':{ # Density limit of 5 but free flow
    #     'path':'invdistprob',
    #     'df':0.0,
    #     'k_vmin':5, # Not applicable but needs a value
    #     'k_lim':5, # Not applicable but needs a value
    #     'label':'Free flow',
    # },    
    # 'ff':{ # Density limit of 5 but free flow
    #     'path':'nearest',
    #     'df':0.0,
    #     'k_vmin':5, # Not applicable but needs a value
    #     'k_lim':5, # Not applicable but needs a value
    #     'label':'Free flow',
    # },    
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
                        WHERE polygon_count = 1 AND CAST(admin_level AS INT) < 10 AND name != '' AND area != 'NaN' AND area BETWEEN 235816681.819764*3/4 AND 235816681.819764*5/4''',
                    'bristol50':'''SELECT * FROM
                        (SELECT p.osm_id, p.name, p.admin_level, SUM(ST_Area(p.way,false))  AS area, COUNT(p.osm_id) AS polygon_count FROM planet_osm_polygon AS p WHERE boundary='administrative' GROUP BY p.osm_id, p.name,p.admin_level ORDER BY admin_level, area DESC) AS q
                        WHERE polygon_count = 1 AND CAST(admin_level AS INT) < 10 AND name != '' AND area != 'NaN' AND area BETWEEN 235816681.819764*2/4 AND 235816681.819764*6/4'''
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
    def __init__(self,id,initial_edge,initial_position,destin):
        ''' Agent class.'''
        self.id = id
        self.initial_edge = initial_edge
        self.initial_position = initial_position
        self.edge = initial_edge
        self.position = initial_position
        self.destin = destin
    def __repr__(self):
        return '[A{} P{:0.2f} E{}]'.format(self.id,self.position,self.edge)

class Track:
    def __init__(self,track,G,fd):
        self.track = track[:]
        self.cursor = 0
        self.G = G
        self.fd = fd        
        self.update()

    def update(self):
        self.this_node,self.this_time = self.track[self.cursor]
        self.that_node,self.that_time = self.track[self.cursor+1]
        self.this_position = np.array(self.G.node[self.this_node]['pos'])
        self.that_position = np.array(self.G.node[self.that_node]['pos'])            
        self.velocity = (self.that_position-self.this_position)/(self.that_time-self.this_time)
        try:
            self.speed = self.G[self.this_node][self.that_node]['distance']/(self.that_time-self.this_time)/self.fd.speedup
        except ZeroDivisionError:
            self.speed = self.fd.v_ff

    def position(self,time):
        while True:
            # The most likely scenario is that cursor hasn't moved
            if time >= self.this_time and time < self.that_time:
                return self.this_position+self.velocity*(time-self.this_time)
            # Cursor has moved up is the next likely scenario
            if time >= self.that_time:
                if self.cursor == len(self.track)-2:
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
    def __init__(self,sim,place,n=None,graph='lite',fresh=False,fresh_db=False,speedup=1,save_route=True):
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
        self.fresh_db = fresh_db
        self.speedup = speedup
        # Current scenario
        self.scenario = None
        # List of scenarios available
        self.scenarios = settings.keys()
        # Place Initialisation
        # --------------------
        # Init highway
        self.h = db.Highway(place,graph=graph,fresh=self.fresh_db,save_route=save_route)
        # Mark the blocked edges, If True, reduces the agent velocity to vMin on that edge (doesn't affect edge capacity)
        for u,v in self.h.G.edges_iter():
            self.h.G[u][v]['blocked'] = False
        # Agent Population Dependent Initialisation
        # -----------------------------------------
        # Init population
        if n:
            self.n = n            
        else:
            if place:
                # If the number of agents is not defined, determine the population by multiplying the 2000 population by p_factor
                self.n = int(round(sum(self.h.pdb.pop)))
        
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

        # Folder to store the simulation and logging folder
        self.folder = 'abm/{}/{}'.format(self.sim,self.place)
        if not os.path.isdir(self.folder):
            os.makedirs(self.folder)
        self.log_folder = '{}/logs'.format(self.folder)
        if not os.path.isdir(self.log_folder):
            os.makedirs(self.log_folder)
        
        # Name of log file
        self.log_file = '{}/{}.{}.txt'.format(self.log_folder,time.ctime(),self.scenario)
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

        self.log_print('Current scenario set to ({0}).'.format(settings[self.scenario]['label']))

        # Count the number of blocked
        self.agents_file = '{0}/n={1}.d={2}'.format(self.folder,self.n,self.destin_hash())
    
        # Name the scenario file
        self.scenario_file = '{0}/{1}'.format(self.agents_file,scenario)

        # Load fundamental diagram
        self.fd = FundamentalDiagram(speedup=self.speedup,k_vmin=settings[self.scenario]['k_vmin'],k_lim=settings[self.scenario]['k_lim'])

        # Name the events file
        if not os.path.isdir(self.scenario_file):
            os.makedirs(self.scenario_file)

        self.events_file = '{0}/events.txt.gz'.format(self.scenario_file)
    
    def load_initial_state(self):
        """The following initial state of agents takes time to compute, save to cache"""
        # NOTE: Need to be able to raise error if no destination
        for u,v,d in self.h.G.edges_iter(data=True):
            # Determine number of agents that are allowed on a given edgelist
            self.h.G[u][v]['storage_capacity'] = int(self.fd.k_max*d['area'])
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
                    # Proceed if all destinations are reachable from this edge
                    try:
                        if len(set(self.h.destins).difference(self.h.G[u][v]['dist2destin'])) == 0:
                            # This is how many agents are supposed to be on this edge
                            # Function that rounds by using probability based on decimal places of the number.
                            # E.g. 6.7 has a 70% probability of being rounded to 7 and 30% probability of being rounded to 6.
                            x = self.h.G[u][v]['pop_dist']*self.n
                            x = int(x) + (random.random() < x - int(x))
                            # Number of agents left to create
                            total_capacity = self.n - agent_id
                            x = min(total_capacity,x)
                            # Initialise the edge to 0 if it hasnt been yet
                            try:
                                initial_state[(u,v)]
                            except KeyError:
                                initial_state[(u,v)] = 0
                            # Check capacity of the link
                            link_capacity = self.h.G[u][v]['storage_capacity']-initial_state[(u,v)]
                            x = min(link_capacity,x)
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
        nx.set_edge_attributes(self.h.G,'initial_state',initial_state)          
        self.initial_state=initial_state

    def init_by_edge(self):
        ''' Function to initialise agent properties.
        '''
        # Load the initial state of agents
        self.load_initial_state()
        # NOTE: Need to be able to raise error if no destination
        for u,v,d in self.h.G.edges_iter(data=True):
            # Initialise the queue
            self.h.G[u][v]['queue'] = deque()            
        # Construct the agents
        agent_id = 0
        self.agents = []
        self.agents_per_tstep = {}
        self.events = {}
        self.log_print('Using [{0}] allocation'.format(settings[self.scenario]['path']))
        for u,v in self.initial_state:
            edge = self.h.G[u][v]
            for i in range(edge['initial_state']):
                # When a new agent is introduced, the position is proportional to the order of the agent
                # Position is a value between 0 and 1 which shows how the agent is progressing through an edge
                # position = 1 means agent is at u
                # position = 0 means agent is at v
                position = float(i+1)/edge['initial_state']
                if settings[self.scenario]['path'] == 'invdistprob':
                    destin = np.random.choice(edge['invdistprob'].keys(),p=edge['invdistprob'].values())
                elif settings[self.scenario]['path'] == 'nearest':
                    destin = edge['nearest_destin']
                agent = Agent(agent_id,initial_edge=(u,v),initial_position=position,destin=destin)
                # Determine the node that the agent is travelling to
                # Save the initial position of the agent
                # Add to the number of agents on that link
                agent.action_time = position*edge['distance']/self.fd.v_ff_spedup
                self.agents.append(agent)
                edge['queue'].append(agent)
                # dist2u: distance to go back to get go from starting position to node u
                agent.dist2u = (position-1)*edge['distance']                
                # What was the time when agent was at u?
                # This is likely to go into negatives
                agent.time_at_u = agent.dist2u/self.fd.v_ff_spedup
                agent.log_event = False
                agent_id += 1
        # Cache percent of agents as True
        for agent in random.sample(self.agents,self.n*self.log_events_percent/100):
            agent.log_event = True
        # Log initial event for these agents
        for agent in self.agents:
            self.log_event(agent,where=0)                

    def init_by_agent(self):
        ''' Function to initialise agent properties.
        '''
        # Load the initial state of agents
        self.load_initial_state()
        # NOTE: Need to be able to raise error if no destination
        for u,v,d in self.h.G.edges_iter(data=True):
            # Initialise the queue length
            self.h.G[u][v]['queue_length'] = 0

        # Construct the agents
        agent_id = 0
        self.agents = []
        self.agents_per_tstep = {}
        self.events = {}
        self.log_print('Using [{0}] allocation'.format(settings[self.scenario]['path']))
        for u,v in self.initial_state:
            edge = self.h.G[u][v]
            for i in range(edge['initial_state']):
                # When a new agent is introduced, the position is proportional to the order of the agent
                # Position is a value between 0 and 1 which shows how the agent is progressing through an edge
                # position = 1 means agent is at u
                # position = 0 means agent is at v
                position = float(i+1)/edge['initial_state']
                if settings[self.scenario]['path'] == 'invdistprob':
                    destin = np.random.choice(edge['invdistprob'].keys(),p=edge['invdistprob'].values())
                elif settings[self.scenario]['path'] == 'nearest':
                    destin = edge['nearest_destin']
                agent = Agent(agent_id,initial_edge=(u,v),initial_position=position,destin=destin)
                # Determine the node that the agent is travelling to
                # Save the initial position of the agent
                # Add to the number of agents on that link
                edge['queue_length'] += 1
                density = self.density(edge)
                velocity = self.fd.v_dict[density]
                agent.action_time = position*edge['distance']/velocity
                try:
                    self.agents_per_tstep[int(agent.action_time)].append(agent)
                except KeyError:
                    self.agents_per_tstep[int(agent.action_time)] = [agent]
                self.agents.append(agent)
                # Agent distance to exit
                agent.dist2x = self.h.route_length[destin][v] + position*edge['distance']
                # dist2u distance to go back to get go from starting position to node u
                agent.dist2u = (position-1)*edge['distance']
                # What was the time when agent was at u?
                # This is likely to go into negatives
                agent.time_at_u = agent.dist2u/velocity
                # Set this flag to Falase by default
                agent.log_event = False
                # Number of timesteps an agent is stuck
                agent.stuck = 0
                # Up agent count
                agent_id += 1
        # Cache percent of agents as True
        for agent in random.sample(self.agents,self.n*self.log_events_percent/100):
            agent.log_event = True
        # Log initial event for these agents
        for agent in self.agents:
            self.log_event(agent,where=0)                
        # If use of Journey Time is enabled, construct a journey time matrix.
        # It is currently not being used as it is not fully mature.
        if False:
            # Construct a profile of journey time based on number of agents on various links
            for u,v,d in self.h.G.edges_iter(data=True):
                velocity = self.fd.v_dict[self.density(d)]
                self.h.G[u][v]['journey_time'] = d['distance']/velocity
    
    def run_by_edge(self,rerun=False,log_events_percent=100,metadata=False,verbose=False):
        ''' Function to animate the agents.
        '''
        success = 0
        # Load this as all scenarios share the same EM EA and route
        # This is within run_sim because they are only required for the simulation.        
        self.h.init_route()
        # Cache events percent
        self.log_events_percent = log_events_percent
        # Iterate through scenarios        
        for scenario in self.scenarios:
            # Initiate the given scenario
            self.init_scenario(scenario)

            if self.events_file_exists() and not rerun:
                self.log_print('Scenario has already been simulated with these parameters!')
            else:                
                self.init_by_edge()
                self.log_print('Starting this simulation...')                
                self.sim_complete = False                  

                self.tstep = 0

                agents_left = self.n

                self.log_print('Time {}*{} seconds: Agents Left: {}'.format(self.tstep,self.fd.speedup, agents_left))

                # Start the timer
                start_time = time.time()

                while agents_left:
                    """Simulation starts here until there are zero agents left"""
                    total_actors = 0
                    flowing_edges = []
                    # Calculate the number of agents that are allowed to move
                    for u,v,d in self.h.G.edges_iter(data=True):
                        if settings[self.scenario]['df'] == 0:
                            # Flow capacity is infinite in free flow
                            d['flow_capacity'] = np.inf  
                        else:
                            fc = self.fd.q_dict[round(len(d['queue'])/d['area'],self.fd.dp)]*d['assumed_width']
                            d['flow_capacity'] = int(fc) + (random.random() < fc - int(fc))
                        if d['flow_capacity'] > 0:
                            flowing_edges.append((u,v))
                    # Randomise the order in which we access these edges
                    random.shuffle(flowing_edges)
                    for u,v in flowing_edges:
                        edge = self.h.G[u][v]
                        actors = 0
                        while edge['flow_capacity'] >= actors and len(edge['queue'])>0:
                            agent = edge['queue'][0]
                            if agent.action_time > self.tstep:
                                """If it is not turn for first agent in the queue to act yet, skip this edge"""
                                if verbose:
                                    self.log_print('Time {}*{} seconds: Not turn to act yet for agent {} at edge ({},{}), SKIPPING LINK'.format(agent.action_time,self.fd.speedup,agent.id,u,v))
                                break
                            if agent.destin == v:
                                """If agent has reached destination"""
                                # Decrement the number of agents left
                                agents_left -= 1
                                # Remove the agent from the queue
                                edge['queue'].popleft()
                                # Preserve the residual time
                                agent.action_time = self.tstep - agent.action_time + int(agent.action_time)                                
                                # Log the time that v is reached
                                self.log_event(agent)
                                # Print the simulation time
                                if verbose:
                                    self.log_print('Time {}*{} seconds: Agent {} has reached destination'.format(agent.action_time,self.fd.speedup,agent.id))
                                # Keep a tally of acting agents
                                actors += 1
                            else:
                                # Determine the new edge
                                new_u = v
                                new_v = self.h.route[agent.destin][v]
                                new_edge = self.h.G[new_u][new_v]
                                if new_edge['storage_capacity'] > len(new_edge['queue'])*settings[self.scenario]['df']:
                                    """Only move the agent if there is capacity in the new edge"""
                                    # Remove the agent from the old edge
                                    edge['queue'].popleft()
                                    # Add agent to the new edge
                                    new_edge['queue'].append(agent)
                                    # Preserve the residual time
                                    agent.action_time = self.tstep - agent.action_time + int(agent.action_time)
                                    if verbose:
                                        self.log_print('Time {}*{} seconds: Old action time for agent {}'.format(agent.action_time,self.fd.speedup,agent.id))
                                    # Log the time that v is reached
                                    self.log_event(agent)
                                    # Determine the next time to take action depending on:
                                    #   - Link length
                                    #   - Link density
                                    agent.action_time += new_edge['distance']/self.fd.v_ff_spedup
                                    # Assign new edge to the agent
                                    agent.edge = (new_u,new_v)
                                    # Keep a tally of acting agents
                                    actors += 1
                                    if verbose:
                                        self.log_print('{} new action time is {}'.format(agent.id,agent.action_time))
                                else:
                                    """If there is no capacity, skip to next link"""
                                    # if agent.id == 1 and self.scenario == 'ff':
                                    #     pdb.set_trace()
                                    if verbose:
                                        self.log_print('Time {}*{} seconds: No storage capacity on edge ({},{}) for agent {}, SKIPPING LINK'.format(agent.action_time,self.fd.speedup,new_u,new_v,agent.id))
                                    break
                        total_actors += actors
                    # Increment timestep
                    self.log_print('Time {}*{} seconds: Acting {}, Left {}'.format(self.tstep,self.fd.speedup,total_actors,agents_left))
                    self.tstep += 1
                # Determine the execution time
                self.execution_time = time.time()-start_time
                self.log_print('Execution took {:0.4f} seconds.'.format(self.execution_time))
                # Log events

                """
                    Different concepts are:
                        gross capacity is how much an edge can carry
                        gross capacity is calculated from capacity density = 5 person/m^2
                        occupancy is how many agents an edge is carrying right now
                        occupancy must be lower than capacity
                        net capacity = gross capacity - occupancy
                        gross occupancy/area is gross occupancy density
                        flow is the number of agents that are allowed to move
                        flow is calculated from occupancy density
                        flow is the limit of number of people of agents that can move
                            as long as the net capacity > 0 in the next edge
                        when an agent enters an edge, 
                        so at each timestep, there is a flow rate calculated from 
                """
                self.cache_events(metadata=metadata)

                success += 1
        return success

    def run_by_agent(self,rerun=False,log_events_percent=100,metadata=False,verbose=False):
        ''' Function to animate the agents.
        '''
        success = 0
        # Load this as all scenarios share the same EM EA and route
        # This is within run_sim because they are only required for the simulation.
        self.h.init_route()
        # Cache events percent
        self.log_events_percent = log_events_percent
        # Iterate through scenarios
        for scenario in self.scenarios:
            # Initiate the given scenario
            self.init_scenario(scenario)
            if self.events_file_exists() and not rerun:
                self.log_print('Scenario has already been simulated with these parameters!')
            else:
                # Cache percent events
                self.init_by_agent()
                self.log_print('Starting this simulation...')                
                self.sim_complete = False                  

                self.tstep = 0
                # Rather than chaging tstep_length, change fd.speedup to control update interval
                self.tstep_length = 1

                agents_left = self.n

                # Start the time
                start_time = time.time()
                while agents_left:
                    rerouting_in_this_tstep = 0                
                    # Event log for this timestep
                    self.loop_this = True
                    try:
                        random.shuffle(self.agents_per_tstep[self.tstep])
                    except KeyError:
                        # There are no agents awaiting action in this timestep so pass
                        self.loop_this = False
                    # if self.tstep == 3:
                    #     pdb.set_trace()
                    if self.loop_this:
                        # Loop through every agent due action in this timestep
                        for agent in self.agents_per_tstep[self.tstep]:
                            # Loop until the agent action_time exceeds the sim_time
                            add_agent = True
                            while agent.action_time < self.tstep + self.tstep_length:
                                # Determine the edge that the agent is on
                                u,v = agent.edge
                                edge = self.h.G[u][v]
                                # print agent.action_time, self.tstep
                                # If this is the last edge
                                if v == agent.destin:
                                    # Decrement the number of agents left
                                    agents_left -= 1
                                    # Remove the agent from the queue
                                    edge['queue_length'] -= 1
                                    # Log the time spent on last edge
                                    self.log_event(agent)
                                    # Print the simulation time
                                    if verbose:
                                        self.log_print('Time: {:0.2f}*{} Agents Left: {}'.format(agent.action_time,self.speedup, agents_left))
                                    add_agent = False
                                    # print 'break 1'
                                    break
                                else:
                                    # occsucc=lambda(succ): [(k,succ[k]['queue_length']/float(succ[k]['storage_capacity'])) for k in succ]
                                    # Determine the new edge
                                    new_u = v
                                    new_v = None
                                    # If the agent has been stuck for longer than 5 minutes, randomly pick a successor road
                                    if agent.stuck > 5*60/self.fd.speedup:
                                        choices = self.h.G.successors(new_u)
                                        # If there is more than one choice
                                        if len(choices) > 1:
                                            choice = random.choice(choices)
                                            # Conditions:
                                            # If the random choice is the the way we came
                                            if choice is not u:                                    
                                                try:
                                                    # The following will invoke an error if there is
                                                    # still a route to destination from our random choice
                                                    self.h.route[agent.destin][choice]
                                                    rerouting_in_this_tstep += 1
                                                    new_v = choice
                                                except KeyError:
                                                    pass
                                    if new_v == None:
                                        new_v = self.h.route[agent.destin][v]
                                    new_edge = self.h.G[new_u][new_v]
                                    # Only move the agent if there is capacity in the new edge
                                    if new_edge['storage_capacity'] > new_edge['queue_length']*settings[self.scenario]['df']:
                                        # Remove the agent from the old edge
                                        edge['queue_length'] -= 1
                                        # Add agent to the new edge
                                        new_edge['queue_length'] += 1
                                        # Log the time spent on last edge
                                        self.log_event(agent)
                                        # Determine the next time to take action depending on:
                                        #   - Link length
                                        #   - Link density
                                        agent.action_time += new_edge['distance']/self.fd.v_dict[self.density(new_edge)]
                                        # Assign new edge to the agent
                                        agent.edge = (new_u,new_v)
                                        # Not stuck anymore
                                        agent.stuck = 0
                                    else:
                                        # If there is no capacity, wait till the next time step                                        
                                        agent.action_time += self.tstep_length
                                        # Count the number of timesteps an agent has been stuck
                                        agent.stuck += 1
                                        break
                            if add_agent:
                                try:
                                    self.agents_per_tstep[int(agent.action_time)].append(agent)
                                except KeyError:
                                    self.agents_per_tstep[int(agent.action_time)] = [agent]
                        self.log_print('Time {}*{} seconds: Agents Acting {} of {}'.format(self.tstep,self.speedup,len(self.agents_per_tstep[self.tstep]),agents_left))
                        if rerouting_in_this_tstep > 0:
                            self.log_print('    {} rerouting after being stuck for more than 5 mins'.format(rerouting_in_this_tstep))
                        # Delete list of agents that were processed in this timestep to free up memory
                        del self.agents_per_tstep[self.tstep]
                    # Increment timestep
                    self.tstep += self.tstep_length
                # Determine the execution time
                self.execution_time = time.time()-start_time
                self.log_print('Execution took {:0.2f} seconds.'.format(self.execution_time))
                # Log events

                """
                    Different concepts are:
                        gross capacity is how much an edge can carry
                        gross capacity is calculated from capacity density = 5 person/m^2
                        occupancy is how many agents an edge is carrying right now
                        occupancy must be lower than capacity
                        net capacity = gross capacity - occupancy
                        gross occupancy/area is gross occupancy density
                        flow is the number of agents that are allowed to move
                        flow is calculated from occupancy density
                        flow is the limit of number of people of agents that can move
                            as long as the net capacity > 0 in the next edge
                        when an agent enters an edge, 
                        so at each timestep, there is a flow rate calculated from 
                """
                self.cache_events(metadata=metadata)

                success += 1
        return success

    def cache_events(self,metadata):
        ''' Cache events
        '''
        to_write = ''
        # Forget the agent id, only the events matter        
        self.events = self.events.values()        
        for event in self.events:
            for node,tstep in event:
                to_write += '{}:{} '.format(tstep,node)
            to_write += '\n'
        with gzip.open(self.events_file,'wb') as file:
            file.write(to_write)

        # Write metadata if this flag is True
        if metadata:
            # Metadata folder
            metadata_folder = 'metadata/{}/{}'.format(self.sim,self.place)
            if not os.path.isdir(metadata_folder):
                os.makedirs(metadata_folder)
            # Write another file with start and end times only for ALL agents
            # Metadata contains start_node, destin_node, start_pos, dist2exit, endtime
            # Initial node
            I = []
            # Destin node
            X = []
            # dist2u distance since sometimes time is negative
            DX = []
            # Distance to exit
            DU = []
            # Time to exit
            TX = []
            for agent in self.agents:
                I.append(agent.initial_edge)
                X.append(agent.destin)
                DU.append(agent.dist2u)
                DX.append(agent.dist2x)
                TX.append(agent.action_time)

            df=pandas.DataFrame(zip(I,X,DU,DX,TX),columns=['initedge','destin','dist2u','dist2x','time2x'])
            df.to_hdf('{}/agents.hdf'.format(metadata_folder),self.scenario,mode='a',complib='blosc',fletcher32=True)
            
            # Dictionary of destin edges
            destin_width = {}
            for destin in self.h.destins:
                destin_width[destin]= {(i,destin):v['assumed_width'] for i,v in self.h.G.pred[destin].iteritems()}

            f = shelve.open('{}/common.shelve'.format(metadata_folder))
            f['destin_width'] = destin_width
            f.close()

    def events_file_exists(self):
        return os.path.isfile(self.events_file)

    def load_events(self,metadata=None):
        ''' Load events
        '''
        self.events = []
        with gzip.open(self.events_file,'rb') as file:
            for line in file.readlines():
                # print line
                this = []
                for pair in line.split(' '):
                    if not pair == '\n':
                        tstep,node = pair.split(':')
                        this.append((int(node),float(tstep)))
                self.events.append(this)
        self.log_print('Event file contains record for {}% of {} agents'.format(100*len(self.events)/self.n,self.n))

    def log_event(self,agent,where=1):
        """Log event to file
            where = 0 means u
            where = 1 means v
        """
        if agent.log_event:
            if where == 1:
                # Use the default action time
                when = agent.action_time
            elif where == 0:
                # This is referring to the beginning so use the initial time
                when = agent.time_at_u
            e = (agent.edge[where],round(when,self.fd.dp))
            try:
                self.events[agent.id].append(e)
            except KeyError:
                self.events[agent.id] = [e]

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
            k = self.fd.k_max
        else:
            k = settings[self.scenario]['df']*edge['queue_length']/edge['area']
        return round(k,self.fd.dp)


class Viewer:
    def __init__(self,sim,bbox=None,t0=0.0,tstep=1,percent=5,video=True):
        ''' Sim tracks viewer.
        
            Inputs
            ------
                sim: Sim object
                    Sim object as defined by class Sim
                t0: float
                    Time at which to start animating the simulation
                tstep: float
                    Amount of timestep to lapse between frames
                percent: int
                    Percentage of agents to be shown in the video
                    eg. 100 = all agents, 10 = 10 percent of agents
        '''                
        self.s = sim
        self.tracks = [Track(e,self.s.h.G,self.s.fd) for e in self.s.events]
        if bbox:
            self.l,self.r,self.b,self.t = min(bbox[0],bbox[2]), max(bbox[0],bbox[2]), min(bbox[1],bbox[3]), max(bbox[1],bbox[3])
        else:
            self.l,self.r,self.b,self.t = self.s.h.l, self.s.h.r, self.s.h.b, self.s.h.t
        self.t0 = t0
        self.tstep = tstep
        self.percent = percent
        self.sample_size = self.s.n * self.percent / 100
        self.sample_tracks = random.sample(self.tracks,self.sample_size)
        self.video_file = '{}/showing-{}-percent.mp4'.format(self.s.scenario_file,self.percent)
        if video:
            self.video()
        # By all, we mean a sample of P
        all_P = []
        all_S = []
        all_t = []
        for t in self.frames():
            P,S = self.PS(t)
            self.remaining_agents = len(S)
            if self.remaining_agents > 0:
                all_P.append(P)
                all_S.append(S)
                all_t.append(np.array([t]*self.remaining_agents))
        self.all_P = np.vstack(all_P)
        self.all_S = np.hstack(all_S)
        self.all_t = np.hstack(all_t)

    def video(self):
        # Load figure background
        self.s.h.fig_highway()
        fig = plt.gcf()
        ax = plt.gca()
        # Set Axes limit
        ax.set_xlim(self.l,self.r) 
        ax.set_ylim(self.b,self.t)
        # Load colourmap
        cm = plt.cm.get_cmap('Spectral')
        # Initialise points
        self.points = ax.scatter([0],[0],c=[0],marker='o',edgecolors='none',cmap=cm,alpha=0.5,clim=[0.1,self.s.fd.v_ff],norm=LogNorm(vmin=0.1, vmax=self.s.fd.v_ff))
        # Initialise text
        self.time_text = ax.text(0.02, 0.94, '', transform=ax.transAxes,alpha=0.5,size='large')
        # Draw colorbar and label
        cb = fig.colorbar(self.points,shrink=0.5,ticks=[self.s.fd.v_dict[k]/self.s.fd.speedup for k in range(int(self.s.fd.k_lim))], format='$%.2f$',orientation='horizontal')
        cb.set_label("Velocity [m/s]", fontsize=15)
        plt.xlabel('Longitude',fontsize=15)
        plt.ylabel('Latitude',fontsize=15)            
        plt.ion()
        plt.show()

        ani = animation.FuncAnimation(fig, self.animate, frames=self.frames(),
            interval=1000/25, blit=False, repeat=False, init_func=self.init_animation, save_count=100000)
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
        self.remaining_agents = len(S) * 100 / self.percent
        self.points.set_offsets(P)
        self.points.set_array(S)
        self.time_text.set_text('T:{0} A:{1}'.format(t,self.remaining_agents))
        return self.points, self.time_text

    def frames(self):
        t = self.t0
        self.remaining_agents = self.sample_size * 100 / self.percent
        # Generate frames until simulation has ended
        while self.remaining_agents > 0:
            yield t
            print t, self.remaining_agents        
            t = t + self.tstep

    def PS(self,t):
        """Return position and speed"""
        try:
            P,S = zip(*np.vstack([(track.position(t),track.speed) for track in self.sample_tracks if track.position(t) is not False]))
        except ValueError:
            P,S = [],[]
        P,S = np.array(P),np.array(S)
        return P,S

    def path(self,which='time'):
        if which == 'time':
            colors = self.all_t
            clim = [0.1,self.all_t.max()]
            ticks = np.logspace(0.1,np.log10(self.all_t.max()),5)
            format = '$%.0f$'
            norm = LogNorm(vmin=0.1, vmax=self.all_t.max())
        elif which == 'speed':
            colors = self.all_S
            clim = [0.1,self.s.fd.v_ff]
            ticks = [self.s.fd.v_dict[k]/self.s.fd.speedup for k in range(int(self.s.fd.k_lim))]
            format = '$%.2f$'
            norm = LogNorm(vmin=0.1, vmax=self.s.fd.v_ff)
        fig = plt.figure(figsize=(12,8))
        ax = plt.gca()
        # Set Axes limit
        # ax.set_xlim(self.l,self.r)
        # ax.set_ylim(self.b,self.t)
        cm = plt.cm.get_cmap('Spectral')
        # Initialise points
        self.points = ax.scatter(self.all_P[:,0],self.all_P[:,1],c=colors,marker='.',edgecolors='none',cmap=cm,alpha=0.2,clim=clim,norm=norm)
        cb = plt.colorbar(self.points,ticks=ticks, format=format,orientation='horizontal',shrink=0.5)
        if which == 'speed':
            cb.set_label("Velocity [m/s]",fontsize=15)
        elif which == 'time':
            cb.set_label("Time [minutes]",fontsize=15)
        for destin in self.s.h.destins:
            plt.scatter(*self.s.h.G.node[destin]['pos'],s=200,c='g',alpha=0.5,marker='o')
        plt.xlabel('Longitude',fontsize=15)
        plt.ylabel('Latitude',fontsize=15)            
        plt.show()
