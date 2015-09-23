# MassEvac v2
import six
import db
import os
import sys
import math
import numpy
import cities
import pickle
import random
import hashlib
import scipy.io

import scipy as sp
import numpy as np
import networkx as nx
import scipy.stats as ss
import matplotlib.pyplot as plt

from matplotlib import mlab, animation
from IPython.display import HTML
from matplotlib.colors import LogNorm
from matplotlib.ticker import FormatStrFormatter
from numpy.lib.function_base import iterable
from mpl_toolkits.axes_grid1 import make_axes_locatable

class FundamentalDiagram:
    # Maximum velocity (m/s)
    vMax = 1.66   
    
    # Minimum velocity (m/s)
    vMin = vMax/10
    
    # Maximum density (agent/m^2)
    pMax = 5.5
    
    # Minimum density (agent/m^2)    
    pMin = 0.00

    # Density upto which velocity is flat (agent/m^2)
    pFlat = 1.00
    
    # Gradient
    m = (vMax - vMin)/(pMin + pFlat - pMax)
    
    # Offset
    c = vMin - m*pMax
    
    # Optimum density to maximise flow - see explaination below
    pOpt = -c/2/m
    
    # Number of bins to profile the density with
    bins = int(math.ceil(pMax-pMin))
    
    # This is the average density in each bin
    # Note that the last bin is 5.25 = (5+5.5)/2
    bin_mean = []
    for i in range(bins):
        lowerBound = i + pMin
        upperBound = lowerBound + 1
        if upperBound > pMax:
            upperBound = pMax
        bin_mean.append((lowerBound+upperBound)/2)
    
    def __init__(self,speedup=60):
        self.speedup = speedup
        self.Qmax = self.velocity(self.pOpt)*self.pOpt
        self.Qmin = self.velocity(self.pMax)*self.pMax
        
        
    # This should form the basis for calculation of speed from density using,
    # velocity = m*density + c
    def velocity(self, p):
        # Calculate velocity from the input density within the boundary limits
        v = self.m * p + self.c
        if p < self.pFlat:
            v = self.vMax
        elif v < self.vMin:
            v = self.vMin
        return v*self.speedup
    
    def figure(self):
        p = np.linspace(self.pMin,self.pMax,int(self.pMax*10)+1)
        v = p.copy()
        
        for i,j in enumerate(p):
            v[i] = self.velocity(j)/self.speedup
        
        f = v*p
        
        offset = 0.5
        
        fontsize=20
        fig = plt.figure(figsize=(9,6),dpi=70)
        ax = fig.add_subplot(111)
        ax.plot(p,v,'r-',linewidth=4,label='$V_{max}$ = %0.2f $m/s$'%max(v))
        ax.set_xlabel('Density $agent/m^{2}$',fontsize=fontsize)
        ax.set_ylabel('Velocity $m/s$',fontsize=fontsize)
        ax.set_xlim(self.pMin,self.pMax+offset)
        ax.set_ylim(self.vMin,self.vMax+offset)
        #ax.legend(loc=2,fontsize=fontsize)

        ax2 = ax.twinx()
        ax2.set_ylabel('Flow $agent/ms$',fontsize=fontsize)
        ax2.plot(p,f,'g--',linewidth=4,label='$F_{max}$ = %0.2f  $agent/ms$'%max(f))
        ax2.set_ylim(f.min(),f.max()+offset)
        #ax2.legend(loc=0,fontsize=fontsize)
        
        plt.show()
        plt.savefig('fd-model.pdf',dpi=300)
        return fig
    
    def which_bin(self,density):
        bin = int(density-self.pMin)
        
        # Sometimes, the initial position of the agents may lead the density to exceed the
        # maximum density, in which case revert the bin index to less than maximum bin index
        if not bin < self.bins:
            bin = self.bins - 1
            
        return bin


class Lookup:
    # Load the fundamental diagram variables
    fd = FundamentalDiagram(speedup = 1)

    def __init__(self, city):
        # Import place
        self.p = db.Place(city)

        # Create a graph of Congested Flow based on uniform desity probability between 0 and 7
        self.CF=nx.DiGraph()

        # Create a graph of Free Flow based on 0 density
        self.FF=nx.DiGraph()

        # Iterate through the edges and generate a Free Flow and a Congested Flow graph
        for i in xrange(self.p.edges):
            ci=self.p.DAM.col[i]
            ri=self.p.DAM.row[i]
            di=self.p.DAM.data[i]
            density = random.random() * 7
            velocity = self.fd.velocity(density)
            self.CF.add_edge(ri,ci,{'weight':di/velocity})
            density = 0
            velocity = self.fd.velocity(density)
            self.FF.add_edge(ri,ci,{'weight':di/velocity})

        # Initialise the origin and destination nodes
        self.set_origin((self.p.l+self.p.r)/2,(self.p.b+self.p.t)/2)
        self.set_destin((self.p.l+self.p.r)/2,(self.p.b+self.p.t)/2)

        # Initialise dijkstra
        self.new_source()

        # Initialise the first path to show
        self.new_sink()

        # Show figure
        self.figure()

    def figure(self):
        # Create a figure
        self.fig = plt.figure()
        self.axs = plt.axes(xlim=(self.p.l,self.p.r), ylim=(self.p.b,self.p.t))
        plt.imshow(self.p.image(), zorder=0, extent=[self.p.l,self.p.r,self.p.b,self.p.t])

        # Initialise the plots
        self.od, = self.axs.plot([self.ox, self.dx], [self.oy, self.dy], 'ro')
        self.CFtrail, = self.axs.plot(self.CFnodes[:,0], self.CFnodes[:,1], 'c-',linewidth=3.0)
        self.FFtrail, = self.axs.plot(self.FFnodes[:,0], self.FFnodes[:,1], 'g--',linewidth=3.0)
        self.Otext = self.axs.text(self.ox, self.oy, self.Ostring, fontsize=15, color='b')
        self.Dtext = self.axs.text(self.dx, self.dy, self.Dstring, fontsize=15, color='b')

        # Connect event handlers
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

        # Show the figure
        plt.show(self.fig)

    def set_origin(self,x,y):
        self.origin = self.p.nearest_node(x, y)
        self.ox = self.p.lon[self.origin]
        self.oy = self.p.lat[self.origin]

    def set_destin(self,x,y):
        self.destin = self.p.nearest_node(x, y)
        self.dx = self.p.lon[self.destin]
        self.dy = self.p.lat[self.destin]

    def new_source(self):
        self.CFpath=nx.single_source_dijkstra_path(self.CF, self.origin)
        self.FFpath=nx.single_source_dijkstra_path(self.FF, self.origin)
        self.CFlength=nx.single_source_dijkstra_path_length(self.CF,self.origin)
        self.FFlength=nx.single_source_dijkstra_path_length(self.FF,self.origin)
        self.Ostring = str(self.origin)

    def new_sink(self):
        self.CFnodes=self.p.nodes[self.CFpath[self.destin],:]
        self.FFnodes=self.p.nodes[self.FFpath[self.destin],:]
        self.Dstring = str(self.destin)+', CF:'+str(int(self.CFlength[self.destin]))+', FF:'+str(int(self.FFlength[self.destin]))

    def draw_od(self):
        self.od.set_data([self.ox,self.dx], [self.oy,self.dy])

    # Update the origin node and all the paths from that node if the user clicks on the graph
    def on_click(self,event):
        x, y = event.xdata, event.ydata
        if x and y:
            self.set_origin(x,y)
            self.new_source()
            self.draw_od()
            self.Otext.set_text(self.Ostring)
            self.Otext.set_position((self.ox,self.oy))
            self.fig.canvas.draw()

    # Update the destination node as the user hovers over the graph
    def on_mouse_move(self,event):
        x, y = event.xdata, event.ydata
        if x and y:
            self.set_destin(x,y)

            try:
                self.new_sink()
                self.draw_od()
                self.CFtrail.set_data(self.CFnodes[:,0], self.CFnodes[:,1])
                self.FFtrail.set_data(self.FFnodes[:,0], self.FFnodes[:,1])
                self.Dtext.set_text(self.Dstring)
                self.Dtext.set_position((self.dx,self.dy))
                self.fig.canvas.draw()
            except KeyError:
                pass

class Sim:
    def __init__(self,sim,city,p_factor=cities.p_factor,scenario='ia',n=None,destins=None,speedup=60,fresh_place=False):

        self.sim = self.name = sim
        self.city = city
        self.speedup = speedup
        self.use_JT = False
        self.fresh_place = fresh_place

        self.sim_folder = 'abm/{0}'.format(self.sim)
        if not os.path.isdir(self.sim_folder):
            os.mkdir(self.sim_folder)

        self.city_folder = '{0}/{1}'.format(self.sim_folder,self.city)
        if not os.path.isdir(self.city_folder):
            os.mkdir(self.city_folder)


        # Scenario Definitions
        # -------------------------------------------------------------------------------
        # ff - Free flow - no agent interaction
        # ia - Interaction mode - agents interact with each other
        # cd - Capped density - density capped at maximum flow

        # List of scenarios
        self.scenarios = ['ff','ia','cd']

        # Import fundamental diagram
        self.f = FundamentalDiagram(speedup = speedup)

        # Link density cap
        self.pCap = {'ff':self.f.pMax,
                     'ia':self.f.pMax,
                     'cd':self.f.pOpt,
                     }

        # Density factor
        self.df = {'ff':0.0,
                   'ia':1.0,
                   'cd':1.0,
                   }

        # Scenario descriptions
        self.scenario_labels = {'ff':'Free flow',
                                 'ia':'With Interaction, No Intervention',
                                 'cd':'With Intervention (Optimum Flow)'}

        # Initialise scenario
        self.init_scenario(scenario=scenario)

        # Place Initialisation
        # -------------------------------------------------------------------------------

        # Init place
        self.p = db.Place(self.city)

        # If the destination nodes are not defined, use the predefined destinations determined from the major leaf nodes
        if destins==None:
            self.p.init_destins(fresh=fresh_place)
        else:
            self.p.destins = destins

        # Agent Population Dependent Initialisation
        # -------------------------------------------------------------------------------

        # Init population
        self.p.init_pop(fresh=fresh_place)

        # If the number of agents is not defined, determine the population by multiplying the 2000 population by p_factor
        if n==None:
            self.n = int(self.p.total_pop * p_factor)
        else:
            self.n = n

        # What to call the outputs - has the number of agents in the name
        if self.p.destins:
            destins_text = str.join('\n',[str(x) for x in sorted(self.p.destins)])
            self.hash = hashlib.md5(destins_text).hexdigest()[:8]
            f = open('{0}/{1}.destins'.format(self.city_folder,self.hash), 'w')
            f.write(destins_text)
            f.close()
            self.agents_file = '{0}/n-{1}-d-{2}'.format(self.city_folder,self.n,self.hash)
        else:
            raise ValueError('{0}[{1}]: Agent file could not be made because destin nodes are not available!'.format(self.city,self.scenario))

    def init_EM_EA_path(self):

        try:
            self.p.EM
        except AttributeError:
            # Initialise edge matrix
            self.p.init_EM(fresh=self.fresh_place)

        try:
            self.p.EA
        except AttributeError:
            # Initialise the area of edges
            self.p.init_EA(fd=self.f,fresh=self.fresh_place)

        try:
            self.p.path
        except AttributeError:
            # Determine shortest paths by reversing the direction of the paths (from the transpose of the graph)
            # Has a boolean because loading path takes time and not always necessary to know all the shortest paths in the network
            self.p.init_path(fresh=self.fresh_place)

    def init_scenario(self,scenario):
        """Set the current scenario to the input scenario label."""

        self.scenario = scenario
        self.scenario_label = self.scenario_labels[self.scenario]
        print '{0}[{1}]: Current scenario set to ({2}).'.format(self.city, self.scenario, self.scenario_label)

    def scenario_file(self,scenario=None):
        """Return the name of the scenario specific agent file name."""

        if scenario == None:
            scenario = self.scenario

        return '{0}/{1}-n-{2}-d-{3}'.format(self.city_folder,scenario,self.n,self.hash)

    def video_file(self):
        """Return the name of the scenario specific video file name."""

        return '{0}.mp4'.format(self.scenario_file())

    def save_agents(self):
        """Save the agents to cache files."""

        fname = '{0}.agents'.format(self.agents_file)
        # Cache these variables
        file = open(fname, 'w')
        pickle.dump([self.P, self.L, self.E, self.X, self.N], file)
        file.close()

        print '{0}[{1}]: Initial state of agents pickled.'.format(self.city,self.scenario)

    def load_agents(self):
        """Load the agents from cache files.

        True  --- if successful
        False --- if unsuccessful"""

        fname = '{0}.agents'.format(self.agents_file)

        if os.path.isfile(fname):
            # Load cache
            file = open(fname, 'r')
            self.P, self.L, self.E, self.X, self.N = pickle.load(file)
            file.close()

            print '{0}[{1}]: Initial state of agents unpickled.'.format(self.city,self.scenario)
            return True
        else:
            print '{0}[{1}]: There are no agents to unpickle.'.format(self.city,self.scenario)
            return False

    def load_result_meta(self):
        '''Cache of meta data that takes a long time to calculate which is result dependent.'''

        try:
            self.X
        except AttributeError:
            self.load_agents()

        try:
            self.T
        except AttributeError:
            self.load_results()

        # Simulation time grouped by destin
        self.T_destin = {}

        for t,x in zip(self.T, self.X):
            try:
                self.T_destin[x].append(t)
            except KeyError:
                self.T_destin[x] = [t]

        # Flow per destin
        self.Q_destin = {} # Flow of agents per destin

        for d,w in zip(self.p.destins,self.p.destin_width):
            try:
                self.Q_destin[d] = [0]*(int(max(self.T_destin[d]))+1)
                for t in self.T_destin[d]:
                    self.Q_destin[d][int(t)] += 1/w
            except KeyError:
                pass


        print '{0}[{1}]: Result meta data loaded.'.format(self.city,self.scenario)

    def load_agent_meta(self):
        '''Cache of meta data that takes a long time to calculate.'''

        try:
            self.E
            self.X
            self.L
        except AttributeError:
            self.load_agents()

        # Calculate distance to exit for every agent
        fname = '{0}.DX'.format(self.agents_file)

        if os.path.isfile(fname):
            # Load cache
            file = open(fname, 'r')
            self.DX = pickle.load(file)
            file.close()

            print '{0}[{1}]: Distance to exits for all agents unpickled.'.format(self.city,self.scenario)
        else:

            try:
                self.p.path_length
            except AttributeError:
                self.p.init_path()

            self.DX = []
            for edge,destin,progress in zip(self.E,self.X,self.L):
                node = self.p.DAM.col[edge]
                dist = self.p.path_length[destin][node]-progress+self.p.DAM.data[edge]
                self.DX.append(dist)

            file = open(fname, 'w')
            pickle.dump(self.DX, file)
            file.close()

            print '{0}[{1}]: Distance to exits for all agents calculated and pickled.'.format(self.city,self.scenario)

        # Distance to exit grouped by destin
        self.DX_destin = {}

        for dx,x in zip(self.DX, self.X):
            try:
                self.DX_destin[x].append(dx)
            except KeyError:
                self.DX_destin[x] = [dx]

        # Numeric representation of destinations
        self.X_num = [self.p.destin_dict[d] for d in self.X]

        # Count the number of agents per destin
        self.n_destin = [0]*len(self.p.destins)
        for x in self.X_num:
            self.n_destin[x] += 1

        print '{0}[{1}]: Agent meta data loaded.'.format(self.city,self.scenario)

    def save_results(self):
        """Save the results to result files."""

        df_fname = '{0}.T'.format(self.scenario_file())
        file = open(df_fname, 'w')
        pickle.dump(self.T, file)
        file.close()

        df_fname = '{0}.D_agent'.format(self.scenario_file())
        file = open(df_fname, 'w')
        pickle.dump(self.D_agent, file)
        file.close()

        df_fname = '{0}.D_tstep'.format(self.scenario_file())
        file = open(df_fname, 'w')
        pickle.dump(self.D_tstep, file)
        file.close()

        df_fname = '{0}.D_edges'.format(self.scenario_file())
        file = open(df_fname, 'w')
        pickle.dump(self.D_edges, file)
        file.close()

        print '{0}[{1}]: Results saved.'.format(self.city,self.scenario)

    def load_results(self):
        """Load the results from result files.

        True  --- if successful
        False --- if unsuccessful"""

        success = True

        df_fname = '{0}.T'.format(self.scenario_file())
        if os.path.isfile(df_fname):
            # Load cache
            file = open(df_fname, 'r')
            self.T = pickle.load(file)
            file.close()
        else:
            print '{0}[{1}]: Time results could not be loaded.'.format(self.city,self.scenario)
            success = False

        df_fname = '{0}.D_agent'.format(self.scenario_file())
        if os.path.isfile(df_fname):
            # Load cache
            file = open(df_fname, 'r')
            self.D_agent = pickle.load(file)
            file.close()
        else:
            print '{0}[{1}]: Agent density profile of each agent could not be loaded.'.format(self.city,self.scenario)
            success = False

        df_fname = '{0}.D_tstep'.format(self.scenario_file())
        if os.path.isfile(df_fname):
            # Load cache
            file = open(df_fname, 'r')
            self.D_tstep = pickle.load(file)
            file.close()

        else:
            print '{0}[{1}]: Agent density profile per timestep could not be loaded.'.format(self.city,self.scenario)
            success = False

        df_fname = '{0}.D_edges'.format(self.scenario_file())
        if os.path.isfile(df_fname):
            # Load cache
            file = open(df_fname, 'r')
            self.D_edges = pickle.load(file)
            file.close()
        else:
            print '{0}[{1}]: Agent total presence density map could not be loaded.'.format(self.city,self.scenario)
            success = False

        if success:
            print '{0}[{1}]: Results successfully loaded.'.format(self.city,self.scenario)

        return success

    def init_agents(self, fresh=False):
        """Function to initialise agent properties."""

        # NOTE: Need to be able to raise error if no destination

        # Simply produce a list of agents so that they can be dealt with in this order
        self.S = range(self.n)

        # Current density array
        self.D = [0]*self.n

        # Current velocity array
        self.V = [0]*self.n

        # Density profile counter to record how often
        # each of the agents are in a dense environment
        self.D_agent = np.zeros((self.n,self.f.bins),dtype=np.int)
        # Record what the density profile looks like per timestep
        self.D_tstep = []
        # Cumulative agent presence density map for each edge
        self.D_edges = np.zeros((self.p.edges,self.f.bins),dtype=np.int)
        # No need to record velocity as well, we can determine that from the density array

        # Time at which destination reached
        # Save to result only
        self.T = [None]*self.n

        if not self.load_agents() or fresh==True:
            # Position of the agents
            # Save to cache only
            self.P = np.zeros((self.n,2))

            # Current location of agents along the link (min = 0, max = DAM.data[thisEdge])
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
            self.N = [0]*self.p.edges

            i = 0
            e = 0

            randomEdges = range(self.p.edges)

            while i < self.n:
                # Randomly determine an edge
                random.shuffle(randomEdges)
                for thisEdge in randomEdges:

                    # This is how many agents are supposed to be on this edge
                    x = self.prob_round(thisEdge)

                    # If the total number of agents for this link exceeds the total allowed to be created
                    if i + x > self.n:
                        x = self.n - i

                    for j in range(x):
                        # Determine the node that the agent is travelling to
                        ci = self.p.DAM.col[thisEdge]
                        try:
                            # Find the nearest destination, if unresolvable then throw KeyError
                            val, idx = min((val, idx) for (idx, val) in enumerate([self.p.path_length[destin][ci] for destin in self.p.destins]))

                            # Assign destination to the agent
                            self.X[i] = self.p.destins[idx]

                            # print 'Edge really manipulated' #tempchange
                            # thisEdge = 5780

                            # Agent is on this edge
                            self.E[i] = thisEdge

                            # Randomly assign a position of the agent on the edge
                            self.L[i] = random.uniform(0,self.p.DAM.data[thisEdge])

                            # Calculate the position co-ordinates
                            di = self.p.DAM.data[thisEdge]
                            ri = self.p.DAM.row[thisEdge]
                            Pr = self.L[i]/di
                            thisNode = self.p.nodes[ri,:]
                            thatNode = self.p.nodes[ci,:]
                            self.P[i,:] = thisNode + (thatNode - thisNode)*Pr

                            # Add to the number of agents on that link
                            self.N[thisEdge] += 1 # In the future, I will need to check capacity of the link before adding the agent

                            # Count the number of agents created
                            i += 1

                        except KeyError:
                            # Count the number of agents that had to be re-created
                            e += 1

                    # If we have the specified number of agents, stop
                    if i==self.n:
                        break

            print '{0}[{1}]: {2}% of agents ({3} of {4}) had to be re-created!'.format(self.city,self.scenario,e*100/float(self.n),e,self.n)

            self.save_agents()

        # If use of Journey Time is enabled, construct a journey time matrix.
        # It is currently not being used as it is not fully mature.
        if self.use_JT:
            # Construct a profile of journey time based on number of agents on various links
            self.JT=nx.DiGraph()
            for i in range(self.p.edges):
                ci=self.p.DAM.col[i]
                ri=self.p.DAM.row[i]
                di=self.p.DAM.data[i]
                density = self.density(i)
                velocity = self.f.velocity(density)
                self.JT.add_edge(ci,ri,{'weight':di/velocity})

    def run_sim(self,fresh=False,video=True,live_video=False,bitrate=4000,fps=20):
        """Function to animate the agents."""

        self.init_EM_EA_path()

        self.init_agents(fresh=fresh)

        # First set up the figure, the axis, and the plot element we want to animate
        self.fig = plt.figure(figsize=(16, 9), dpi=100)

        axs = plt.axes(xlim=(self.p.l, self.p.r), ylim=(self.p.b, self.p.t))

        axs.xaxis.set_major_formatter(FormatStrFormatter('%0.2f'))
        axs.yaxis.set_major_formatter(FormatStrFormatter('%0.2f'))

        plt.xlabel('Longitude')
        plt.ylabel('Latitude')

        # Load the background
        plt.imshow(self.p.image(fresh=True), zorder=0, extent=[self.p.l, self.p.r, self.p.b, self.p.t])

        # Load colourmap
        cm = plt.cm.get_cmap('Spectral_r')

        # Initialise the plots
        point = axs.scatter([0,0],[0,0],c=[0,0],marker='o',cmap=cm,alpha=0.25,norm=LogNorm(vmin=0.5, vmax=self.f.pMax))
        agents_text = axs.text(0.02, 0.94, '', transform=axs.transAxes,alpha=0.5,size='x-large')

        # Draw colorbar and label
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", "5%", pad="3%")

        cb=self.fig.colorbar(point,cax=cax,ticks=self.f.bin_mean, format='$%.2f$')
        cb.set_label("Agent Link Density $m^{-2}$", rotation=90)

        # Function to get current position of agents
        def sim_update_agents(t):
            # Shuffle the order of the agents around
            random.shuffle(self.S)
            removal = []
            # Record the density profile for this timestep in this list
            this_tstep = [0]*self.f.bins
            for i in self.S:
                thisEdge = self.E[i]
                self.D[i] = self.density(thisEdge)
                self.V[i] = self.f.velocity(self.D[i])
                self.L[i] += self.V[i] # Move the agent
                di = self.p.DAM.data[thisEdge]
                # Travelling from this node (ri)
                ri = self.p.DAM.row[thisEdge]
                # To this node (ci)
                ci = self.p.DAM.col[thisEdge]
                Pr = self.L[i]/di # Percentage progress along the link (1 = completed)
                # If the agent travels beyond the length of the link

                while Pr > 1:
                    # Determine the amount of time remaining from last run
                    residual_time = (self.L[i] - di)/self.V[i]

                    if ci == self.X[i]:
                        # Agent has reached its destination, record time
                        self.T[i] = t + 1 - residual_time

                        # Agent has reached the end of the link
                        self.L[i] = 1

                        # Subtract the agent from this edge
                        self.N[thisEdge] -= 1

                        # Remove agent from our list of agents
                        removal.append(i)

                        Pr = 1
                    else:
                        next_ri = ci
                        next_ci = self.p.path[self.X[i]][ci]
                        nextEdge = self.p.EM[next_ri][next_ci]
                        nextDensity = self.density(nextEdge,add=1)
                        # If there is enough space for the agent in the next link, only then proceed
                        if nextDensity <= self.pCap[self.scenario]:
                            # Assign new nodes
                            ri = next_ri
                            ci = next_ci

                            # Subtract agent away from previous edge
                            self.N[thisEdge] -= 1

                            # Determine the index of new edge
                            self.E[i] = thisEdge = nextEdge

                            # Add agent to the new edge
                            self.N[thisEdge] += 1

                            # Calculate the new density
                            self.D[i] = nextDensity

                            # If the next edge has vacancy then progress to the next edge
                            # if D[i]>pMax and random.random()<0.01: # 1% chance of rerouting
                                # print '{0} Congestion, rerouting...',D[i], thisEdge
                                # path=nx.single_source_dijkstra_path(JT,dest)

                            # Calculate new velocity
                            self.V[i] = self.f.velocity(self.D[i])

                            # Assign the new distance taking into account the remaining time
                            self.L[i] = self.V[i] * residual_time

                            # Calculate new progress
                            di = self.p.DAM.data[thisEdge]
                            Pr = self.L[i]/di
                        # If there is no space, then just wait at the end of the link
                        else:
                            self.L[i] = di
                            Pr = 1

                # Determine new position
                thisNode = self.p.nodes[ri,:]
                thatNode = self.p.nodes[ci,:]
                self.P[i,:] = thisNode + (thatNode - thisNode)*Pr

                # Update journey time based on velocity on this link
                if self.use_JT:
                    self.JT[ci][ri]['weight']=di/self.V[i]

                # Get the bin id in which to add the agent
                bin = self.f.which_bin(self.D[i])

                # This is the density profile for a given agent over all timestep
                self.D_agent[i,bin] = self.D_agent[i,bin] + 1

                # This is the density profile for all agents per time step
                this_tstep[bin] = this_tstep[bin] + 1

                # This is the density profile for all agents for all edges over all timestep
                self.D_edges[thisEdge,bin] = self.D_edges[thisEdge,bin] + 1

            # Append the agglomerate density profile to our list of density profiles per timestep
            self.D_tstep.append(this_tstep)

            # Remove agents that have reached destination
            for r in removal:
                self.S.remove(r)

        self.sim_complete = False
        self.last_agents_left = self.n

        # Initialization function: plot the background of each frame
        def sim_init():
            print '{0}[{1}]: Starting this simulation...'.format(self.city,self.scenario)

        # Animation function.  This is called sequentially
        def sim_animate(i):
            if not self.sim_complete:
                agents_left = len(self.S)

                # Update the agent location graph
                point.set_offsets(self.P[self.S,:])
                # Update the color array
                point.set_array(np.array(self.D))

                agents_text.set_text('T:{0}, A:{1}'.format(i,agents_left))
                # Print out the progress
                if i%10==0 or agents_left == 0:
                    exit_rate = (self.last_agents_left - agents_left)/10
                    print '{0}[{1}]: Time {2} Agents left {3} Exit per frame {4}'.format(self.city,self.scenario,i,agents_left,exit_rate)
                    self.last_agents_left = agents_left
                if agents_left == 0:
                    self.sim_complete = True
                    print '{0}[{1}]: End of this simulation.'.format(self.city,self.scenario)
                else:
                    # Update the position of the agents
                    sim_update_agents(i)

            # Since only the point and agent_text need updating, just return these
            return point,agents_text

        def data_gen():
            a = 0
            count_last_frames = 0

            # Generate frames until simulation has ended + additional second is added to the end of the video
            while not self.sim_complete and count_last_frames < fps:

                # If there are 0 agents, start counting additional frames
                if self.sim_complete:
                    count_last_frames = count_last_frames + 1

                yield a
                a = a + 1

        # Call the animator.  blit=True means only re-draw the parts that have changed.
        self.anim = animation.FuncAnimation(fig=self.fig, func=sim_animate, init_func=sim_init,
                               frames=data_gen(), interval=20, blit=False)

        # Save the animation as an mp4.  This requires ffmpeg or mencoder to be
        # installed.  The extra_args ensure that the x264 codec is used, so that
        # the video can be embedded in html5.  You may need to adjust this for
        # your system: for more information, see
        # http://matplotlib.sourceforge.net/api/animation_api.html

        if live_video:
            plt.show(block=True)

        if video:
            self.anim.save(self.video_file(), fps=fps, bitrate=bitrate,extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
            print '{0}[{1}]: Saving video to file {1}...'.format(self.city,self.scenario,self.video_file())

        # Resize and save time to file
        # Resizing to 2 decimal places as that is enough information which
        # is accurate enough to render seconds if every time step is a minute
        # which is more than we need.
        self.T=[round(t,2) for t in self.T]

        # Convert this to a numpy array as it will be easier to work with as opposed to a list!
        self.D_tstep = np.array(self.D_tstep,dtype=np.int)

        self.save_results()

    def density(self,edge,add=0):
        """Function to calculate link density for the input edge.
        If df[scenario] = 0, returns 0 density so that we can determine free flow evacuation time."""

        return self.df[self.scenario]*(self.N[edge]+add)/self.p.EA[edge]

    def prob_round(self,edge):
        """Function that rounds by using probability based on decimal places of the number.
        E.g. 6.7 has a 70% probability of being rounded to 7 and 30% probability of being rounded to 6."""

        x = self.p.pop_dist[edge]*self.n
        sign = np.sign(x)
        x = abs(x)
        is_up = random.random() < x-int(x)
        round_func = math.ceil if is_up else math.floor
        return int(sign * round_func(x))

    def et_stats(self):
        """Calculate the standard mean, standard deviation, median and ninetieth percentile."""

        mu, sigma = ss.norm.fit(self.T)
        median=ss.scoreatpercentile(self.T,50)
        ninetieth=ss.scoreatpercentile(self.T,90)
        return mu, sigma, median, ninetieth

    def et_figure(self,lognorm=False,count=100,xlim=None):
        """Produce histogram of the evacuation time."""

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

        plt.savefig('{0}-et.png'.format(self.scenario_file()), dpi=300)
        plt.close(fig)
        return fig

    # Plot population distribution map
    def highlight_agents(self,agent_list,agent_weight,weight_label,fname,colorbar_ticks=None,sample_size=1000):
        """Agent weight must be the same length as the number of agents."""
        fig = self.p.figure()

        pop_sample = random.sample(agent_list, sample_size)

        plt.scatter(self.P[pop_sample,0],self.P[pop_sample,1],c=agent_weight[pop_sample],alpha=0.5)
        cb = plt.colorbar(ticks=colorbar_ticks)
        cb.set_label(weight_label, rotation=90)

        plt.savefig(fname.format(self.scenario_file()), dpi=300)
        plt.close(fig)
        return fig

    def highlight_edges(self,edges,edge_weight,weight_label,fname):
        """Edge list must be the same length as the edge weight."""

        # Compile a list of node tuples from the edge numbers
        edge_list=zip(self.p.DAM.row[edges],self.p.DAM.col[edges])

        # Import the map layer
        fig = self.p.figure(theme='greyscale')

        # Draw the edges in the edge list in a new axis layer
        this_layer = nx.draw_networkx_edges(self.p.HAMG,pos=self.p.nodes,arrows=False,edgelist=edge_list,edge_color=edge_weight[edges],width=5,alpha=0.5)

        # Draw the colorbar
        cb = plt.colorbar(this_layer)
        cb.set_label(weight_label, rotation=90)

        plt.savefig(fname.format(self.scenario_file), dpi=300)
        plt.close(fig)
        return fig

    def range_D_edges(self,fr,to):
        """Retrieve edges that have density values in the given range."""

        return mlab.find(np.sum(self.D_edges[:,fr:to],1))

    def average_D_edges(self):
        """Average density weighted by units of time spent on that edge in total by the agents."""

        average = []
        for D_profile in self.D_edges:
            val = sum([self.f.bin_mean[i]*D_profile[i] for i in range(self.f.bins)])
            if val > 0:
                val = val/sum(D_profile)
            average.append(val)
        return np.array(average)

    def fraction_of_sum_total_time(self):
        """Fraction of sum of total time of all agents spent on this link.

        Note: The sum of all fractions should be approximately 1 where the error
        is accumulated rounding errors when placing the density quantity in bins."""
        return np.array(np.sum(self.D_edges,1)/sum(self.T))

    def range_D_agent(self,fr,to):
        """Retrieve agents that have density values in the given range."""

        agents=mlab.find(np.sum(self.D_agent[:,fr:to],1))
        return agents

    def average_D_agent(self):
        """Average density weighted by number of timesteps."""

        average = []
        for D_profile in self.D_agent:
            val = sum([self.f.bin_mean[i]*D_profile[i] for i in range(self.f.bins)])
            if val > 0:
                val = val/sum(D_profile)
            average.append(val)
        return np.array(average)

    def average_D_tstep(self):
        """Average density weighted by number of agents left in the simulation."""

        average = []
        for D_profile in self.D_tstep:
            val = sum([self.f.bin_mean[i]*D_profile[i] for i in range(self.f.bins)])
            if val > 0:
                val = val/sum(D_profile)
            average.append(val)
        return np.array(average)

    def video(self,fname=None):
        """Produce HTML inline video of the simulation."""

        if fname==None:
            fname = self.video_file()
        VIDEO_TAG = """<video width=800 controls autoplay>
         <source src="data:{0}">
         Your browser does not support the video tag.
        </video>"""
        video = open(fname, "rb").read()
        encoded_video = 'video/mp4;base64,' + video.encode("base64")
        return HTML(VIDEO_TAG.format(encoded_video))

# <codecell>

# copied directly from the proposed fix
def monkey_patch_init(self, fig, func, frames=None, init_func=None, fargs=None,
             save_count=None, **kwargs):
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

animation.FuncAnimation.__init__ = monkey_patch_init
