# MassEvac V3
import db
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

def monkey_patch_init(self, fig, func, frames=None, init_func=None, fargs=None,
             save_count=None, **kwargs):
    ''' Animation function patch. IGNORE.
        Note
        ----
            Copied directly from the proposed fix
    '''
    if fargs:
        self._args = fargs
    else:
        self._args = ()
    self._func = func
    # Amount of framedata to keep around for saving movies. This is only
    # used if we don't know how many frames there will be: in the case
    # of no generator or in the case of a callable.
    self.save_count = save_count
    # Set up a function that creates a new iterable when needed. If nothing
    # is passed in for frames, just use itertools.count, which will just
    # keep counting from 0. A callable passed in for frames is assumed to
    # be a generator. An iterable will be used as is, and anything else
    # will be treated as a number of frames.
    if frames is None:
        self._iter_gen = itertools.count
    elif six.callable(frames):
        self._iter_gen = frames
    elif iterable(frames):
        self._iter_gen = lambda: iter(frames)
        if hasattr(frames, '__len__'):
            self.save_count = len(frames)
    else:
        self._iter_gen = lambda: xrange(frames).__iter__()
        self.save_count = frames
    # If we're passed in and using the default, set it to 100.
    if self.save_count is None:
        self.save_count = 100000
    self._init_func = init_func
    # Needs to be initialized so the draw functions work without checking
    self._save_seq = []
    animation.TimedAnimation.__init__(self, fig, **kwargs)
    # Need to reset the saved seq, since right now it will contain data
    # for a single frame from init, which is not what we want.
    self._save_seq = []
# Patch the animation function
animation.FuncAnimation.__init__ = monkey_patch_init        

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
        # Use journey time or not?
        self.use_JT = False       
        # Save agent trajectory or not 
        self.dump_N = False
        # Save average velocity
        self.dump_VD = False
        # Agents to track
        self.track_list = []
        # scatter or quiver
        self.agent_marker = 'scatter'
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
        # If the destination nodes are not defined, use the predefined destinations determined from the major leaf nodes
        self.h.init_destins()
        # Mark the blocked edges, If True, reduces the agent velocity to vMin on that edge (doesn't affect edge capacity)
        self.blocked = len(self.h.edges)*[False]
        # If True, pick a random successor in case the next edge happens to be blocked
        self.random_successor = False
        # Agent Population Dependent Initialisation
        # -----------------------------------------
        # Init population
        self.h.init_pop()
        # If the number of agents is not defined, determine the population by multiplying the 2000 population by p_factor
        self.n = int(self.h.total_pop * p_factor)
        
    def destin_hash(self):
        ''' Generate a hash for the list of destins.
        '''
        if self.h.destins:
            destins_text = str.join('\n',[str(x) for x in sorted(self.h.destins)])
            this_hash = hashlib.md5(destins_text).hexdigest()[:3]
            # Will have to check if the file already exists in the future
            f = open('{0}/{1}.destins'.format(self.folder,this_hash), 'w')
            f.write(destins_text)
            f.close()
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
        blocked_count = self.blocked.count(True)
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

    def tstep_file(self,tstep,label):
        ''' Return the name of the scenario specific agent file name.
        '''
        fname = '{0}/tstep/{1}'.format(self.scenario_file(),label)
        if not os.path.isdir(fname):
            os.makedirs(fname)
        return '{0}/{1}'.format(fname,tstep)
    
    def video_file(self):
        ''' Return the name of the scenario specific video file name.
        '''
        return '{0}/video.mp4'.format(self.scenario_file())
    
    def save_agents(self):
        ''' Save the agents to cache files.
        '''
        fname = '{0}/agents'.format(self.agents_file())
        self.log_print('Writing {0}'.format(fname))
        file = open(fname, 'w')
        pickle.dump([self.P, self.L, self.E, self.X, self.N], file)
        file.close()
    
    def load_agents(self):
        ''' Load the agents from cache files.
        
            Output
            ------
                success: Boolean            
                    True  --- if successful
                    False --- if unsuccessful
        '''    
        fname = '{0}/agents'.format(self.agents_file())
        if os.path.isfile(fname) and not self.fresh == True:
            # Load cache
            self.log_print('Loading {0}'.format(fname))
            file = open(fname, 'r')
            self.P, self.L, self.E, self.X, self.N = pickle.load(file)
            file.close()
            return True
        else:
            return False

    def load_agent_times(self):
        ''' Iterates through the results and gathers agent times for
            all scenarios in geojson properties compatiable format.
        '''
        # Load agents
        try:
            self.E
        except AttributeError:
            self.load_agents()
        # Initiate properties
        properties = [{} for _ in range(len(self.h.edges))]
        for scenario in self.scenarios:
            self.scenario = scenario
            self.load_results()
            edge_T = [[] for _ in range(len(self.h.edges))]
            # Iterate through all edges and gather agent times
            # and create a list of agent times per edge
            for e,t in zip(self.E, self.T):
                edge_T[e].append(t)
            # Consolidate into a single list of dictionary
            # where each item refers to an edge in order
            for e in range(len(self.h.edges)):
                if edge_T[e] != []:
                    properties[e]["{0}_mean_time".format(scenario)] = np.mean(edge_T[e])
                    properties[e]["{0}_stdv_time".format(scenario)] = np.std(edge_T[e])
        return properties
    
    def load_result_meta(self):
        ''' Cache of meta data that takes a long time to calculate which is result dependent.
        '''
        self.load_agents()        
        self.load_results()
        # Simulation time grouped by destin
        self.T_destin = {}
        self.T_destin_index = {}
        for i,(t,x) in enumerate(zip(self.T, self.X)):
            try:
                self.T_destin[x].append(t)
                self.T_destin_index[x].append(i)
            except KeyError:
                self.T_destin[x] = [t]
                self.T_destin_index[x] = [i]
        # Flow per destin
        self.Q_destin = {} # Flow of agents per destin
        for d,w in zip(self.h.destins,self.h.destin_width):
            try:
                self.Q_destin[d] = [0]*(int(max(self.T_destin[d]))+1)
                for t in self.T_destin[d]:
                    self.Q_destin[d][int(t)] += 1/w
            except KeyError:
                pass

    def load_agent_meta(self):
        ''' Cache of meta data that takes a long time to calculate.
        '''
        self.load_agents()
        # Calculate distance to exit for every agent
        fname = '{0}/DX'.format(self.agents_file())
        if os.path.isfile(fname) and not self.fresh == True:
            self.log_print('Loading {0}'.format(fname))
            file = open(fname, 'r')
            self.DX = pickle.load(file)
            file.close()
        else:
            self.log_print('Processing {0}'.format(fname))
            self.h.init_route()
            self.DX = []
            for edge,destin,progress in zip(self.E,self.X,self.L):
                node = self.h.C[edge]
                dist = self.h.route_length[destin][node]-progress+self.h.D[edge]
                self.DX.append(dist)
            self.log_print('Writing {0}'.format(fname))
            file = open(fname, 'w')
            pickle.dump(self.DX, file)
            file.close()
        self.DX_destin = {}
        self.n_destin = {}
        for dx,x in zip(self.DX, self.X):
            try:
                # Count the number of agents per destin
                self.n_destin[x] = self.n_destin[x] + 1
                # Distance to exit grouped by destin
                self.DX_destin[x].append(dx)
            except KeyError:
                self.n_destin[x] = 0           
                self.DX_destin[x] = [dx]
        # Numeric representation of destinations
        self.X_num = [self.h.destin_dict[d] for d in self.X]        
            
    def save_results(self):
        ''' Save the results to result files.
        '''
        fname = '{0}/T'.format(self.scenario_file())
        self.log_print('Writing {0}'.format(fname))
        file = open(fname, 'w')
        pickle.dump(self.T, file)
        file.close()
        fname = '{0}/KP_agent'.format(self.scenario_file())
        self.log_print('Writing {0}'.format(fname))
        file = open(fname, 'w')
        pickle.dump(self.KP_agent, file)
        file.close()
        fname = '{0}/KP_tstep'.format(self.scenario_file())
        self.log_print('Writing {0}'.format(fname))
        file = open(fname, 'w')
        pickle.dump(self.KP_tstep, file)
        file.close()
        fname = '{0}/KP_edges'.format(self.scenario_file())
        self.log_print('Writing {0}'.format(fname))
        file = open(fname, 'w')
        pickle.dump(self.KP_edges, file)
        file.close()
        if self.tracked_agent:
            fname = '{0}/tracked_agent'.format(self.scenario_file())
            self.log_print('Writing {0}'.format(fname))
            file = open(fname, 'w')
            pickle.dump(self.tracked_agent, file)
            file.close()            
    
    def load_results(self):
        ''' Load the results from result files.
        
            Output
            ------
                success: Boolean    
                    True  - if successful
                    False - if unsuccessful
        '''
        success = True
        fname = '{0}/T'.format(self.scenario_file())
        if os.path.isfile(fname):
            self.log_print('Loading {0}'.format(fname))
            file = open(fname, 'r')
            self.T = pickle.load(file)
            file.close()
        else:
            success = False
        fname = '{0}/KP_agent'.format(self.scenario_file())
        if os.path.isfile(fname):
            self.log_print('Loading {0}'.format(fname))
            file = open(fname, 'r')
            self.KP_agent = pickle.load(file)
            file.close()
        else:
            success = False
        fname = '{0}/KP_tstep'.format(self.scenario_file())
        if os.path.isfile(fname):
            self.log_print('Loading {0}'.format(fname))
            file = open(fname, 'r')
            self.KP_tstep = pickle.load(file)
            file.close()
        else:
            success = False
        fname = '{0}/KP_edges'.format(self.scenario_file())
        if os.path.isfile(fname):
            # Load cache
            self.log_print('Loading {0}'.format(fname))
            file = open(fname, 'r')
            self.KP_edges = pickle.load(file)
            file.close()
        else:
            success = False
        fname = '{0}/tracked_agent'.format(self.scenario_file())
        if os.path.isfile(fname):
            # Load cache
            self.log_print('Loading {0}'.format(fname))
            file = open(fname, 'r')
            self.tracked_agent = pickle.load(file)
            file.close()
        else:
            success = False
        if not success:
            self.log_print('Some of the results could not be loaded.')
        return success
    
    def init_agents(self):
        ''' Function to initialise agent properties.
        '''
        # NOTE: Need to be able to raise error if no destination
        # Determine number of agents that are allowed on a given edge
        self.EC = [int(settings[self.scenario]['kCap']*area) for area in self.h.EA]
        # The following initial state of agents takes time to compute, save to cache
        # --------------------------------------------------------------------------
        if not self.load_agents():
            # Position of the agents
            # Save to cache only
            self.P = np.zeros((self.n,2))
            # Current location of agents along the link (min = 0, max = h.D[this_edge])
            # Save to cache only
            self.L = [None]*self.n
            # Edge that the agent is on
            # Save to cache only
            self.E = [None]*self.n
            # This is the agent destination array
            # Save to cache only
            self.X = [None]*self.n
            # Matrix with number of agents on a given edge for each timestep
            # Save to cache only
            self.N = [0]*self.h.nedges
            this_agent = 0
            bad_agents = 0
            randomEdges = range(self.h.nedges)
            while this_agent < self.n:
                self.log_print('{0} of {1} agents created.'.format(this_agent,self.n))
                # Randomly determine an edge
                random.shuffle(randomEdges)
                for this_edge in randomEdges:
                    # This is how many agents are supposed to be on this edge
                    x = self.prob_round(this_edge)
                    # If the total number of agents for this link exceeds the total allowed to be created
                    total_capacity = self.n - this_agent
                    if x > total_capacity:
                        x = total_capacity
                    # Check capacity of the link
                    link_capacity = self.EC[this_edge]-self.N[this_edge]                    
                    if x > link_capacity:
                        x = link_capacity                    
                    for j in range(x):
                        # Determine the node that the agent is travelling to
                        ci = self.h.C[this_edge]
                        try:
                            # Find the nearest destination, if unresolvable then throw KeyError
                            val, idx = min((val, idx) for (idx, val) in enumerate([self.h.route_length[destin][ci] for destin in self.h.destins]))
                            # Assign destination to the agent
                            self.X[this_agent] = self.h.destins[idx]
                            # Agent is on this edge
                            self.E[this_agent] = this_edge
                            # Randomly assign a position of the agent on the edge
                            self.L[this_agent] = random.uniform(0,self.h.D[this_edge])
                            # Calculate the position co-ordinates
                            di = self.h.D[this_edge]
                            ri = self.h.R[this_edge]
                            pr = self.L[this_agent]/di
                            this_node = np.array(self.h.nodes[ri])
                            that_node = np.array(self.h.nodes[ci])
                            self.P[this_agent,:] = this_node + (that_node - this_node) * pr
                            # Add to the number of agents on that link
                            self.N[this_edge] += 1
                            # Count the number of agents created
                            this_agent += 1
                        except KeyError:
                            # Count the number of agents that had to be re-created
                            bad_agents += 1
                    # If we have the specified number of agents, stop
                    if this_agent==self.n:
                        break
            self.log_print('{0}% of agents ({1} of {2}) had to be re-created!'.format(bad_agents*100/float(self.n),bad_agents,self.n))
            self.save_agents()
        # No need to save the following
        # -----------------------------
        # Simply produce a list of agents so that they can be dealt with in this order
        self.S = range(self.n)
        # Direction that the agents are facing
        self.U = np.ones((self.n,2))
        # Current density array
        self.K = [self.density(this_edge) for this_edge in self.E]
        # Save the following to result only
        # ---------------------------------
        # Density profile counter to record how often
        # each of the agents are in a dense environment
        self.KP_agent = np.zeros((self.n,fd.bins),dtype=np.int)
        # Record what the density profile looks like per timestep
        self.KP_tstep = []
        # Cumulative agent presence density map for each edge
        self.KP_edges = np.zeros((self.h.nedges,fd.bins),dtype=np.int)
        # No need to record velocity as well, we can determine that from the density array
        # Time at which destination reached
        self.T = [None]*self.n
        # If use of Journey Time is enabled, construct a journey time matrix.
        # It is currently not being used as it is not fully mature.
        if self.use_JT:
            # Construct a profile of journey time based on number of agents on various links
            self.JT=nx.DiGraph()
            for this_edge in range(self.h.nedges):
                ri=self.h.R[this_edge]
                ci=self.h.C[this_edge]
                di=self.h.D[this_edge]
                density = self.density(this_edge)
                velocity = fd.velocity(density)
                self.JT.add_edge(ri,ci,{'weight':di/velocity})
    
    def run_sim(self,video=True,live_video=False,bitrate=4000,fps=20,rerun=False):
        ''' Function to animate the agents.
        '''
        success = 0
        # Load this as all scenarios share the same EM EA and route
        # This is within run_sim because they are only required for the simulation.
        self.h.init_EM()       
        self.h.init_EA()
        self.h.init_route()
        # Iterate through scenarios
        for scenario in self.scenarios:
            # Initiate the given scenario
            self.init_scenario(scenario)
            if self.load_results() and not rerun:
                self.log_print('Scenario has already been simulated with these parameters!')
            else:
                # Load agents, if possible, from cache!
                self.init_agents()
                # First set up the figure, the axis, and the plot element we want to animate                
                fsize = np.array([self.h.r-self.h.l,self.h.t-self.h.b])
                ratio = np.round(fsize*24/sum(fsize))                
                self.log_print('Simulation video aspect ratio will be {0} by {1}!'.format(ratio[0],ratio[1]))
                self.fig = plt.figure(figsize=ratio, dpi=100)
                axs = plt.axes(xlim=(self.h.l, self.h.r), ylim=(self.h.b, self.h.t))
                axs.xaxis.set_major_formatter(FormatStrFormatter('%0.2f'))
                axs.yaxis.set_major_formatter(FormatStrFormatter('%0.2f'))
                plt.xlabel('Longitude')
                plt.ylabel('Latitude')        
                # Load the background
                plt.imshow(self.h.img_highway(), zorder=0, extent=[self.h.l, self.h.r, self.h.b, self.h.t])
                # Load colourmap
                cm = plt.cm.get_cmap('Spectral_r')
                # Initialise the plots
                if self.agent_marker == 'scatter':
                    point = axs.scatter(self.P[:,0],self.P[:,1],c=self.K,marker='o',cmap=cm,alpha=0.5,clim=[0.5,fd.kCf],norm=LogNorm(vmin=0.5, vmax=fd.kCf))
                elif self.agent_marker == 'quiver':
                    point = plt.quiver(self.P[:,0],self.P[:,1],self.U[:,0],self.U[:,1],self.K,cmap=cm,alpha=0.5,clim=[0.5,fd.kCf],norm=LogNorm(vmin=0.5, vmax=fd.kCf))
                agents_text = axs.text(0.02, 0.94, '', transform=axs.transAxes,alpha=0.5,size='large')
                # Draw colorbar and label
                divider = make_axes_locatable(plt.gca())
                cax = divider.append_axes("right", "5%", pad="3%")
                cb = self.fig.colorbar(point,cax=cax,ticks=fd.bin_mean, format='$%.2f$')
                cb.set_label("Agent Local Density $m^{-2}$", rotation=90)
                def sim_update(tstep):
                    ''' Function to get update position of agents.
                    '''
                    # Shuffle the order of the agents around
                    random.shuffle(self.S)
                    removal = []
                    # Record the density profile for this timestep in this list
                    this_tstep = [0]*fd.bins
                    for this_agent in self.S:
                        this_edge = self.E[this_agent]
                        this_density = self.density(this_edge)
                        this_velocity = fd.v_dict[this_density]
                        # Move the agent
                        this_location = self.L[this_agent] + this_velocity
                        di = self.h.D[this_edge]
                        # Travelling from this node (ri)
                        ri = self.h.R[this_edge]
                        # To this node (ci)
                        ci = self.h.C[this_edge]
                        # Has agent gone beyond the length of the link?
                        residual_di = this_location - di
                        # If the agent travels beyond the length of the link      
                        while residual_di > 0:
                            # Determine the amount of time remaining from last run
                            residual_time = residual_di/this_velocity
                            # Agent has reached the destination node
                            if ci == self.X[this_agent]:
                                this_location = di
                                # Agent has reached its destination, record time
                                self.T[this_agent] = tstep + 1 - residual_time
                                # Subtract the agent from this edge
                                self.N[this_edge] -= 1
                                # Remove agent from our list of agents
                                removal.append(this_agent)
                            # Agent has not reached the destination node
                            else:
                                next_ri = ci
                                next_ci = self.h.route[self.X[this_agent]][ci]
                                next_edge = self.h.EM[next_ri][next_ci]
                                # If the next edge is blocked, equal likelihood of picking all available routes
                                if self.blocked[next_edge] and self.random_successor:
                                    # Get list of all successor nodes
                                    all_succ = self.h.G.succ[next_ri]
                                    # Pick a random successor node
                                    next_ci = all_succ.keys()[random.randrange(len(all_succ))]
                                    # Determine the edge 
                                    next_edge = self.h.EM[next_ri][next_ci]
                                # If there is no space, then just wait at the end of the link
                                if self.N[next_edge]+1 > self.EC[next_edge]:
                                    this_location = di
                                # If there is enough space for the agent in the next link, only then proceed
                                else:
                                    # Assign new nodes
                                    ri = next_ri
                                    ci = next_ci
                                    # Subtract agent away from previous edge
                                    self.N[this_edge] -= 1
                                    # Add agent to the new edge
                                    self.N[next_edge] += 1
                                    # Update this edge
                                    this_edge = next_edge
                                    # Update this density
                                    this_density = self.density(this_edge)
                                    # If the next edge has vacancy then progress to the next edge
                                    # if this_density>kMax and random.random()<0.01: # 1% chance of rerouting
                                        # self.log_print('{0} {1} Congestion, rerouting...'.format(this_density, edge))
                                        # path=nx.single_source_dijkstra_path(JT,dest)
                                    # Calculate new velocity
                                    this_velocity = fd.v_dict[this_density]
                                    # Assign the new distance taking into account the remaining time
                                    this_location = this_velocity * residual_time
                                    # Calculate new progress
                                    di = self.h.D[this_edge]
                            # Calculate the residual distance
                            residual_di = this_location - di
                        # Determine new position
                        this_node = np.array(self.h.nodes[ri])
                        that_node = np.array(self.h.nodes[ci])
                        pr = this_location / di
                        offset = (that_node-this_node) * pr
                        self.P[this_agent,:] = this_node + offset
                        # Update array of unit vectors if the agent marker is quiver
                        if self.agent_marker == 'quiver':
                            magnitude = np.linalg.norm(offset)
                            if magnitude > 0:
                                unit = offset/magnitude
                            else:
                                unit = offset
                            self.U[this_agent,:] = unit
                        # Update to current agent location
                        self.L[this_agent] = this_location
                        # Update to current agent edge
                        self.E[this_agent] = this_edge
                        # Update to current agent edge
                        self.K[this_agent] = this_density
                        if this_agent in self.track_list:
                            self.tracked_agent[this_agent].append(self.dist2exit(this_agent))
                        # Update journey time based on velocity on this link
                        if self.use_JT:
                            self.JT[ci][ri]['weight']=di/this_velocity
                        # Get the bin id in which to add the agent
                        bin = fd.which_bin(this_density)
                        # This is the density profile for a given agent over all timestep
                        self.KP_agent[this_agent,bin] = self.KP_agent[this_agent,bin] + 1
                        # This is the density profile for all agents per time step
                        this_tstep[bin] = this_tstep[bin] + 1
                        # This is the density profile for all agents for all edges over all timestep
                        self.KP_edges[this_edge,bin] = self.KP_edges[this_edge,bin] + 1
                    # Append the agglomerate density profile to our list of density profiles per timestep
                    self.KP_tstep.append(this_tstep)
                    # Remove agents that have reached destination
                    for r in removal:
                        self.S.remove(r)
                def sim_init():
                    '''Initialization function: plot the background of each frame.'''
                    self.log_print('Starting this simulation...')
                    self.tracked_agent = {}
                    for tl in self.track_list:
                        self.tracked_agent[tl] = [self.dist2exit(tl)]
                    self.sim_complete = False
                    self.last_agents_left = self.n                    
                def sim_animate(tstep):
                    ''' Animation function.  This is called sequentially.
                    '''
                    if not self.sim_complete:
                        agents_left = len(self.S)
                        # Update the agent location graph
                        point.set_offsets(self.P[self.S,:])
                        if self.agent_marker == 'scatter':
                            # Update the color array
                            point.set_array(np.array(self.K)[self.S])
                        elif self.agent_marker == 'quiver':
                            # Update the color and direction array
                            point.set_UVC(self.U[self.S,0],self.U[self.S,1],np.array(self.K)[self.S])
                        # Text in the corner
                        agents_text.set_text('T:{0}, A:{1}'.format(tstep,agents_left))
                        if tstep%10 == 0 or agents_left == 0:
                            # Print out the progress                            
                            exit_rate = (self.last_agents_left - agents_left)/10
                            self.log_print('Time {0} Agents left {1} Exit per frame {2}'.format(tstep,agents_left,exit_rate))
                            self.last_agents_left = agents_left
                        # Store the number of agents on every edge per timestep
                        if self.dump_N:
                            with open(self.tstep_file(tstep,'N'),'w') as f:
                                pickle.dump(self.N, f)
                        if agents_left == 0:
                            self.sim_complete = True
                            self.log_print('End of this simulation.')
                        else:
                            # Update the position of the agents
                            sim_update(tstep)
                    # Since only the point and agent_text need updating, just return these
                    return point,agents_text
                def sim_frames():
                    ''' Generate a new frame if the simulation is not complete.
                    '''
                    tstep = 0
                    count_last_frames = 0
                    # Generate frames until simulation has ended + additional second is added to the end of the video
                    while not self.sim_complete and count_last_frames < fps:
                        # If there are 0 agents, start counting additional frames
                        if self.sim_complete:
                            count_last_frames = count_last_frames + 1
                        yield tstep
                        tstep = tstep + 1
                # Call the animator.  blit=True means only re-draw the parts that have changed.
                self.anim = animation.FuncAnimation(fig=self.fig, func=sim_animate, init_func=sim_init,
                                       frames=sim_frames(), interval=20, blit=False)
                # Save the animation as an mp4.  This requires ffmpeg or mencoder to be
                # installed.  The extra_args ensure that the x264 codec is used, so that
                # the video can be embedded in html5.  You may need to adjust this for
                # your system: for more information, see
                # http://matplotlib.sourceforge.net/api/animation_api.html
                if live_video:
                    plt.show(block=True)
                if video:
                    self.anim.save(self.video_file(), fps=fps, bitrate=bitrate,extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
                    self.log_print('Saving video to file {0}...'.format(self.video_file()))
                # Resize and save time to file
                # Resizing to 2 decimal places as that is enough information which
                # is accurate enough to render seconds if every time step is a minute
                # which is more than we need.
                self.T = [round(t,2) for t in self.T]
                # Convert this to a numpy array as it will be easier to work with as opposed to a list!
                self.KP_tstep = np.array(self.KP_tstep,dtype=np.int)
                self.save_results()
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

    def density(self,edge,add=0):
        ''' Function to calculate link density for the input edge.
            If df = 0, returns 0 density so that we can determine free flow evacuation time.
        '''
        if self.blocked[edge]:
            k = fd.kMax
        else:
            k = settings[self.scenario]['df']*(self.N[edge]+add)/self.h.EA[edge]
        return round(k,fd.dp)
    
    def prob_round(self,edge):
        ''' Function that rounds by using probability based on decimal places of the number.
            E.g. 6.7 has a 70% probability of being rounded to 7 and 30% probability of being rounded to 6.
        '''
        x = self.h.pop_dist[edge]*self.n
        sign = np.sign(x)
        x = abs(x)
        is_up = random.random() < x-int(x)
        round_func = math.ceil if is_up else math.floor
        return int(sign * round_func(x))
    
    def et_stats(self):
        ''' Calculate the standard mean, standard deviation, median and ninetieth percentile.
        '''
        mu, sigma = ss.norm.fit(self.T)
        median=ss.scoreatpercentile(self.T,50)
        ninetieth=ss.scoreatpercentile(self.T,90)
        return mu, sigma, median, ninetieth
    
    def et_figure(self,lognorm=False,count=100,xlim=None):
        ''' Produce histogram of the evacuation time.
        '''
        fig = plt.figure(figsize=(8,3))
        ax = fig.add_subplot(1,1,1)
        mu, sigma, median, ninetieth = self.et_stats()
        if lognorm:
            s, loc, scale = ss.lognorm.fit(self.T)
        # The histogram of the data
        n, bins, patches = plt.hist(self.T, count, normed=1, facecolor='green', alpha=0.5)
        # Add a 'best fit' line
        if lognorm:
            y = ss.lognorm.pdf( bins, s, loc, scale)
        else:
            y = mlab.normpdf( bins, mu, sigma)
        mode=bins[max(xrange(len(y)),key=y.__getitem__)]
        if lognorm:
            label = r'$\mu=%.1f,\ \sigma=%.1f,\ 50\%%=%.1f,\ 90\%%=%.1f,\ Mo=%.1f,\ sh=%.1f,\ loc=%.1f,\ sc=%.1f$' %(mu,sigma,median,ninetieth,mode,s,loc,scale)
        else:
            label = r'$\mu=%.1f,\ \sigma=%.1f,\ 50\%%=%.1f,\ 90\%%=%.1f$' %(mu, sigma, median, ninetieth)
        l = plt.plot(bins, y, 'r--', linewidth=2)
        ax.set_xlim(0,max(bins))
        # ax.set_xscale('log')
        if xlim:
            ax.set_xlim(0,xlim)
        else:
            ax.set_xlim(0,max(bins))
        # Plot
        ax.set_xlabel('Time')
        ax.set_ylabel('Ratio of Agents')
        ax.set_title(label,fontsize=12)
        plt.savefig('{0}/et.png'.format(self.scenario_file()), dpi=300)
        return fig
    
    # Plot population distribution map
    def highlight_agents(self,agent_list,agent_weight,weight_label,fname,colorbar_ticks=None,sample_size=1000):
        ''' Agent weight must be the same length as the number of agents.
        '''
        fig = self.h.fig_highway()
        pop_sample = random.sample(agent_list, sample_size)
        plt.scatter(self.P[pop_sample,0],self.P[pop_sample,1],c=agent_weight[pop_sample],alpha=0.5)
        cb = plt.colorbar(ticks=colorbar_ticks)
        cb.set_label(weight_label, rotation=90)
        plt.savefig(fname.format(self.scenario_file()), dpi=300)
        plt.close(fig)
        return fig
    
    def highlight_edges(self,edges,edge_weight,weight_label,fname):
        ''' Edge list must be the same length as the edge weight.
        '''
        # Compile a list of node tuples from the edge numbers
        edge_list=zip(self.h.R[edges],self.h.C[edges])
        # Import the map layer
        fig = self.h.fig_highway(theme='greyscale')
        # Draw the edges in the edge list in a new axis layer
        this_layer = nx.draw_networkx_edges(self.h.G,pos=self.h.nodes,arrows=False,edgelist=edge_list,edge_color=edge_weight[edges],width=5,alpha=0.5)
        # Draw the colorbar
        cb = plt.colorbar(this_layer)
        cb.set_label(weight_label, rotation=90)
        plt.savefig(fname.format(self.scenario_file), dpi=300)
        plt.close(fig)
        return fig
    
    def range_KP_edges(self,fr,to):
        ''' Retrieve edges that have density values in the given range.
        '''
        return mlab.find(np.sum(self.KP_edges[:,fr:to],1))
    
    def average_KP_edges(self):
        ''' Average density weighted by units of time spent on that
            edge in total by the agents.
        '''
        average = []
        for kp in self.KP_edges:
            val = sum([fd.bin_mean[i]*kp[i] for i in range(fd.bins)])
            if val > 0:
                val = val/sum(kp)
            average.append(val)
        return np.array(average)
    
    def fraction_of_sum_total_time(self):
        ''' Fraction of sum of total time of all agents spent on this link.
            Note: The sum of all fractions should be approximately 1 where the error
            is accumulated rounding errors when placing the density quantity in bins.
        '''
        return np.array(np.sum(self.KP_edges,1)/sum(self.T))
    
    def range_KP_agent(self,fr,to):
        ''' Retrieve agents that have density values in the given range.
        '''
        agents=mlab.find(np.sum(self.KP_agent[:,fr:to],1))
        return agents
    
    def average_KP_agent(self):
        ''' Average density weighted by number of timesteps.
        '''
        average = []
        for kp in self.KP_agent:
            val = sum([fd.bin_mean[i]*kp[i] for i in range(fd.bins)])
            if val > 0:
                val = val/sum(kp)
            average.append(val)
        return np.array(average)
    
    def average_KP_tstep(self):
        ''' Average density weighted by number of agents left in the simulation.
        '''
        average = []
        for kp in self.KP_tstep:
            val = sum([fd.bin_mean[i]*kp[i] for i in range(fd.bins)])
            if val > 0:
                val = val/sum(kp)
            average.append(val)
        return np.array(average)
    
    def video(self,fname=None):
        ''' Produce HTML inline video of the simulation.
        '''
        if fname==None:
            fname = self.video_file()
        VIDEO_TAG = '''<video width=800 controls autoplay>
         <source src="data:{0}">
         Your browser does not support the video tag.
        </video>'''
        video = open(fname, "rb").read()
        encoded_video = 'video/mp4;base64,' + video.encode("base64")
        return HTML(VIDEO_TAG.format(encoded_video))