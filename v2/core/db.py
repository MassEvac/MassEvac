# MassEvac v2
import os
import sys
import math
import scipy
import random
import pickle
import psycopg2
import scipy.io
import shapely.wkb

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from matplotlib import mlab
from scipy.misc import imread

class Place:
    def __init__(self,city):
        self.city = self.name = city

        # Assumptions
        # --------------------------------------------------------------------------------
        # Standard with per lane
        self.wlane = 2.5

        # Widths of the 7 types of highways
        self.width = np.array([3,2,2,1.5,1,1,0.5])*self.wlane

        # 1 'motorway'
        # 1 'motorway_link'
        # 2 'trunk'
        # 2 'trunk_link'
        # 3 'primary'
        # 3 'primary_link'
        # 4 'secondary'
        # 4 'secondary_link'
        # 5 'tertiary'
        # 5 'tertiary_link'
        # 6 'residential'
        # 6 'byway'
        # 6 'road'
        # 6 'living_street'
        # 7 all others...

        self.highway = '../OSM/cache/highway/osm_gb/{0}'.format(self.city)
        self.folder = 'cities/osm_gb/{0}'.format(self.city)

        if not os.path.isdir(self.folder):
            os.mkdir(self.folder)

        self.DAM = scipy.io.loadmat(self.highway+'/sDAM.mat')['DAM'].tocoo()
        self.HAM = scipy.io.loadmat(self.highway+'/sHAM.mat')['HAM'].astype('uint8').tocoo()
        self.DAMG = nx.DiGraph(self.DAM,format='weighted_adjacency_matrix')
        self.HAMG = nx.DiGraph(self.HAM,format='weighted_adjacency_matrix')
        self.nodes = scipy.io.loadmat(self.highway+'/snodes.mat')['nodes']
        self.edges = self.DAM.getnnz()
        self.lon = self.nodes[:,0]
        self.lat = self.nodes[:,1]
        self.l,self.r=min(self.lon),max(self.lon)
        self.b,self.t=min(self.lat),max(self.lat)

        # Determine the centroidal node of the graph from which to trace all our routes from
        # (Assuming that all destinations are accessible from the centroid)

        # Get the list of nodes of the biggest connected component in the graph
        self.biggest_strconcom_nodes=nx.strongly_connected_components(self.HAMG)[0]

        self.centroid = self.nearest_node((self.l-self.r)/2+self.r,(self.t-self.b)/2+self.b,self.biggest_strconcom_nodes)

        self.mapname = self.folder + '/map-{0}.png'

    def __repr__(self):
        return self.city

    def init_EM(self,fresh=False):
        fname = self.folder + '/EM.place'
        if os.path.isfile(fname) and fresh==False:
            print '{0}: Loading EM from file...'.format(self.city)
            file = open(fname, 'r')
            self.EM = pickle.load(file)
            file.close()
        else:
            print '{0}: Processing EM dict...'.format(self.city)
            # Initialise matrix with edge number
            self.EM = {}

            for i in range(self.edges):
                try:
                    self.EM[self.DAM.row[i]].update({self.DAM.col[i]:i})
                except KeyError:
                    self.EM[self.DAM.row[i]] = {self.DAM.col[i]:i}

            # Dump the results to a file
            file = open(fname, 'w')
            pickle.dump(self.EM, file)
            file.close()

        # Returns the number of edges in the edges dictionary
        return sum([len(self.EM[d].values()) for d in self.EM])

    def init_EA(self,fd,fresh=False):
        """Function to initialise area of edges."""

        # Calculate the minimum area so that at least one agent can fit through at free flow velocity at all times
        # This is to correct a previous bug where agents were getting stuck on a path because it was too narrow!
        # I am assuming that it only affects fraction of edges
        EA_min = 1.0/fd.pFlat

        fname = self.folder + '/EA(min={0}).place'.format(EA_min)
        if os.path.isfile(fname) and fresh==False:
            print '{0}: Loading EA from file...'.format(self.city)
            file = open(fname, 'r')
            self.EA = pickle.load(file)
            file.close()
        else:
            print '{0}: Processing EA...'.format(self.city)
            # Initialise area of edges
            self.EA = [None] * self.edges

            # Assign the widths to the edges
            for i in range(self.edges):
                self.EA[i] = self.DAM.data[i]*self.width[self.HAM.data[i]-1]

                # If the edge area is less than the minimum area (enough to handle one agent)
                if self.EA[i] < EA_min:
                    self.EA[i] = EA_min

            # Dump the results to a file
            file = open(fname, 'w')
            pickle.dump(self.EA, file)
            file.close()

    def init_destins(self,fresh=False,cutoff=3.5,figure=False):
        # Name of file
        fname = self.folder + '/destins.place'
        if os.path.isfile(fname) and fresh==False:
            print '{0}: Loading destins from file...'.format(self.city)

            file = open(fname, 'r')
            self.destins = pickle.load(file)
            file.close()
        else:
            print '{0}: Processing destins ...'.format(self.city)

            # Detect all nodes with just one adjacent edge by assuming that the graph is undirected
            all_leaves=[n for n,d in self.HAMG.to_undirected().degree().items() if d==1]
            # Filter out the edges that are big
            big_leaves=[v for u,v,d in self.HAMG.in_edges(all_leaves,data=True) if d['weight']<cutoff]

            # Find path length from the centroidal node to all other nodes
            path_len=nx.single_source_dijkstra_path_length(self.HAMG,self.centroid)

            # If the path length to a given node is valid, the destination node is valid!
            self.removed_leaves = []
            for i in big_leaves:
                try:
                    path_len[i]
                except KeyError:
                    self.removed_leaves.append(i)

            # Remove the nodes that were invalidated from our list of destination nodes
            for r in self.removed_leaves:
                big_leaves.remove(r)

            self.destins = big_leaves
            self.all_leaves = all_leaves

            # Dump the results to a file
            file = open(fname, 'w')
            pickle.dump(self.destins, file)
            file.close()

        # Enumerates the exits with node numbers
        if figure:
            fig = self.figure()

            plt.scatter(self.nodes[self.destins,0],self.nodes[self.destins,1])
            plt.scatter(*self.nodes[self.centroid,:],color='r')

            for label, x, y in zip(self.destins, self.nodes[self.destins, 0], self.nodes[self.destins, 1]):
                plt.annotate(
                    label,
                    xy = (x, y), xytext = (20,20),
                    textcoords = 'offset points', ha = 'left', va = 'bottom',
                    bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
                    arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

            return fig

        # Create a dictionary map of destination node number to a numeric value
        self.destin_dict = dict([(d,i) for i,d in enumerate(self.destins)])
        # List of numeric destins
        self.destin_range = range(len(self.destins))
        # Dictionary of destin edges
        self.destin_edges = {}
        # List of destin width
        self.destin_width_dict = {}
        # List of destin width
        self.destin_width = []
        for destin in self.destins:
            self.destin_edges[destin]=list(mlab.find(self.DAM.col==destin))

            w = sum([self.width[self.HAM.data[e] - 1] for e in self.destin_edges[destin]])
            self.destin_width_dict[destin] = w
            self.destin_width.append(w)

    # Initialise shortest path dictionaries to various destination nodes
    def init_path(self,fresh=False):
        # Initialise path and path length dictionaries
        # Loading cache now seems to be twice as fast as precomputing
        self.path = {}
        self.path_length = {}
        self.destin_edges = {}

        # We are reversing so that we can determine the shortest path to a single sink rather than from a single source
        DAMGT = self.DAMG.reverse(copy=True)

        self.destins_folder = '{0}/destins'.format(self.folder)
        if not os.path.isdir(self.destins_folder):
            os.mkdir(self.destins_folder)

        for destin in self.destins:
            fname = '{0}/{1}.place'.format(self.destins_folder,destin)
            if os.path.isfile(fname) and fresh==False:
                print '{0}: Loading paths to destination node {1} from file...'.format(self.city,destin)


                file = open(fname, 'r')
                self.path[destin],self.path_length[destin],self.destin_edges[destin] = pickle.load(file)
                file.close()
            else:
                print '{0}: Processing paths to destination node {1}...'.format(self.city,destin)

                # Calculate a single SINK shortest path and path length
                self.path_length[destin],path=nx.single_source_dijkstra(DAMGT,destin)
                self.path[destin] = {}
                # Only store the next node to go to from any given node to reduce memory consumption
                for node in path:
                    try:
                        self.path[destin][node] = path[node][-2]
                    except IndexError:
                        pass

                # Determine the links connected to the node
                self.destin_edges[destin]=list(mlab.find(self.DAM.col==destin))

                # Dump the results to a file
                file = open(fname, 'w')
                pickle.dump([self.path[destin],self.path_length[destin],self.destin_edges[destin]], file)
                file.close()

        # Determine the list of all destination edges
        self.all_destin_edges = [x for v in self.destin_edges.values() for x in v]

        return 'Path to destinations initialised.'

    def init_pop(self,fresh=False):
        # Name of file
        fname = self.folder + '/pop_dist.place'
        if os.path.isfile(fname) and fresh==False:
            file = open(fname, 'r')
            self.pop_lon, self.pop_lat, self.pop, self.total_pop, self.pop_dist = pickle.load(file)
            file.close()
        else:
            # Get population distribution from the database
            pdb = db.Population(self.city)

            # List to record the total length of links per node on population grid
            total_length = np.zeros(len(pdb.pop))

            # List to record which node on population grid the link is nearest to
            which_pop_node = [None]*self.edges

            # Determine the nearest node on population grid for every link
            for i in range(self.edges):
                ri = self.DAM.row[i]
                ci = self.DAM.col[i]
                di = self.DAM.data[i]
                # Calculate the midpoint
                mx = (self.lon[ri] + self.lon[ci]) / 2
                my = (self.lat[ri] + self.lat[ci]) / 2
                # Determine the nearest population node
                pi = pdb.nearest_node(mx,my)
                which_pop_node[i] = pi
                # Sum the total length of the road segment
                total_length[pi] += di

            # Determine which population nodes have been referenced
            ref_nodes = np.unique(which_pop_node)

            # Prepare the output
            self.pop_lon = pdb.lon[ref_nodes]
            self.pop_lat = pdb.lat[ref_nodes]
            self.pop = pdb.pop[ref_nodes]

            self.total_pop = sum(self.pop)

            # List to record the percentage of total population per edge
            self.pop_dist = np.zeros(self.edges)

            # Now distribute the population proportional to the length of link
            for i in range(self.edges):
                di = self.DAM.data[i]
                pi = which_pop_node[i]
                self.pop_dist[i] = pdb.pop[pi]/self.total_pop * di/total_length[pi]

            # Dump the results to a file
            file = open(fname, 'w')
            pickle.dump([self.pop_lon, self.pop_lat, self.pop, self.total_pop, self.pop_dist], file)
            file.close()
        return 'Edge population distribution initiased.'

    def pop_figure(self):
        # Generate the figure
        fig = plt.figure()
        plt.scatter(self.pop_lon,self.pop_lat,c=self.pop)
        plt.colorbar()
        plt.close(fig)
        return fig

    def image(self,theme='default',fresh=False):
        # Name of file
        if not os.path.isfile(self.mapname.format(theme)) or fresh==True:
            fig = self.figure()
            plt.close(fig)
        return imread(self.mapname.format(theme))

    def figure(self,theme='default'):
        # Classify roads into big and small roads for the purpose of drawing the map
        edge_list = {}
        for i in range(1,8):
            edge_list[i]=[(u,v) for (u,v,d) in self.HAMG.edges(data=True) if d['weight'] == i]

        edge_dict = {'default':{
                                 1:{'alpha':1.00,'edge_color':'LightSkyBlue'},
                                 2:{'alpha':0.81,'edge_color':'DarkOliveGreen'},
                                 3:{'alpha':0.64,'edge_color':'IndianRed'},
                                 4:{'alpha':0.49,'edge_color':'GoldenRod'},
                                 5:{'alpha':0.36,'edge_color':'Gold'},
                                 6:{'alpha':0.25,'edge_color':'Gray'},
                                 7:{'alpha':0.16,'edge_color':'Pink'}
                                 },
                     'greyscale':{
                                 1:{'alpha':1.00,'edge_color':'Gray'},
                                 2:{'alpha':0.81,'edge_color':'Gray'},
                                 3:{'alpha':0.64,'edge_color':'Gray'},
                                 4:{'alpha':0.49,'edge_color':'Gray'},
                                 5:{'alpha':0.36,'edge_color':'Gray'},
                                 6:{'alpha':0.25,'edge_color':'Gray'},
                                 7:{'alpha':0.16,'edge_color':'Gray'}
                                 }
                     }

        # Generate the figure
        fig = plt.figure()
        ax = plt.axes(xlim=(self.l, self.r), ylim=(self.b, self.t))
        # Reversing so that the smaller roads are drawn first
        for i in reversed(edge_dict[theme].keys()):
            nx.draw_networkx_edges(self.HAMG,pos=self.nodes,arrows=False,edgelist=edge_list[i],**edge_dict[theme][i])

        # Draw the boundary of the city
        b=db.Boundary(self.city)
        x,y=b.shape.exterior.xy
        plt.plot(x,y,alpha=0.3)

        # Mark the destination nodes
        destin_nodes = self.nodes[self.destins]
        plt.scatter(destin_nodes[:,0],destin_nodes[:,1],s=100,c='g',alpha=0.5,marker='*')

        # Save the extent to a file
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        plt.savefig(self.mapname.format(theme), bbox_inches=extent, dpi=300)
        return fig
    
    def edge_width(self,edge):
        return self.width[self.HAM.data[edge]-1]

    def nearest_node(self,x,y,nodelist=None):
        if nodelist==None:
            nodelist = range(len(self.nodes))
            
        lon = self.lon[nodelist]
        lat = self.lat[nodelist]

        # Function to get the node number nearest to a prescribed lon and lat
        distance=np.sqrt(np.square(lon-x)+np.square(lat-y))
        nearest=min(range(len(distance)), key=distance.__getitem__)
        return nodelist[nearest]

    def where_is_node(self,node):
        """Shows the location of the node and returns the edges it is connected to."""

        fig=self.figure()
        plt.scatter(self.nodes[node,0],self.nodes[node,1],s=500,alpha=0.25,c='blue')
        plt.scatter(self.nodes[node,0],self.nodes[node,1],s=5,alpha=0.5,c='red')

        return self.DAMG.edges(node)

    def where_is_edge(self,edge):
        """Shows the location of the edge and returns the edges it is connected to."""

        ri = self.DAM.row[edge]
        ci = self.DAM.col[edge]

        mx = np.mean(self.nodes[[ri,ci],0])
        my = np.mean(self.nodes[[ri,ci],1])

        fig=self.figure()
        plt.scatter(mx,my,s=500,alpha=0.25,c='blue')
        nx.draw_networkx_edges(self.HAMG,pos=self.nodes,arrows=False,edgelist=[(ri, ci)],edge_color='red',width=5,alpha=0.5)

class Query:
    def __init__(self,SQL,data=None):
        self.connect()
        self.SQL = SQL
        self.data = data
        self.query()
                
    def connect(self):
        try:
            try:
                con = psycopg2.connect("dbname=osm_gb user=bk12369 host=localhost password=postgres")
            except:
                con = psycopg2.connect("dbname=osm_gb user=bharatkunwar host=localhost")
            self.cur = con.cursor()        
        except Exception, e:
            print 'Bharat, there is something wrong with the connection information you have supplied for the database!', e
                
    def query(self):
        if self.data == None:
            try:
                self.cur.execute(self.SQL)
                print self.SQL
            except Exception, e:
                print e.pgerror
        else:
            try:
                self.cur.execute(self.SQL,self.data)
                print self.cur.mogrify(self.SQL,self.data)
            except Exception, e:
                print e.pgerror
                
        self.result = self.cur.fetchall()
        
    def __len__(self):
        return len(self.result)

    def __repr__(self):
        for record in self.result:
            print record
        return 'End of record.'

# <codecell>

class Population:
    def __init__(self,city,fresh=None):
        self.city = city
        fname = 'cities/osm_gb/{0}/population_grumpv1.var'.format(self.city)
        if os.path.isfile(fname) and fresh == None:
            file = open(fname, 'r')
            self.lon,self.lat,self.pop=pickle.load(file)
            file.close()            
        else:
            SQL = """SELECT ST_RasterToWorldCoordX(p.rast, x) AS wx, ST_RasterToWorldCoordY(p.rast,y) AS wy, ST_Value(p.rast, x, y) as v
                    FROM population_grumpv1 AS p, 
                    (SELECT way FROM planet_osm_polygon WHERE name = %s AND boundary = %s ORDER BY ST_NPoints(way) DESC LIMIT 1) AS f
                    CROSS JOIN generate_series(1, 100) As x
                    CROSS JOIN generate_series(1, 100) As y
                    WHERE ST_Intersects(p.rast,f.way);"""
            
            data = (self.city,'administrative',)

            result = Query(SQL,data).result
            
            n = len(result)
            
            lon = [None]*n
            lat = [None]*n
            pop = [None]*n
            
            for i,r in enumerate(result):
                lon[i] = r[0]
                lat[i] = r[1]
                pop[i] = r[2]
            
            self.lon = np.array(lon)
            self.lat = np.array(lat)
            self.pop = np.array(pop)
            
            file = open(fname, 'w')
            pickle.dump([self.lon,self.lat,self.pop], file)
            file.close()
            
    def __repr__(self):
        return self.city + ' with ' + str(len(self.pop)) + ' data points.'
    
    def figure(self):
        fig = plt.figure()
        axs = plt.axes(xlim=(min(self.lon),max(self.lon)), ylim=(min(self.lat), max(self.lat)))        
        plt.scatter(self.lon,self.lat,c=self.pop)
        plt.colorbar()
        plt.close(fig)
        return fig
    
    def nearest_node(self,x,y):
        # Function to get the node number nearest to a prescribed node
        distance=np.sqrt(np.square(self.lon-x)+np.square(self.lat-y))
        nearest=min(range(len(distance)), key=distance.__getitem__)
        return nearest    

# <codecell>

class Boundary:
    def __init__(self,city,fresh=None):
        self.city = city
        fname = 'cities/osm_gb/{0}/boundary.var'.format(self.city)
        if os.path.isfile(fname) and fresh == None:
            file = open(fname, 'r')
            self.shape = pickle.load(file)
            file.close()
        else:
            SQL = 'SELECT way FROM planet_osm_polygon WHERE name = %s AND boundary = %s ORDER BY ST_NPoints(way) DESC LIMIT 1'
            
            data = (self.city,'administrative',)

            result = Query(SQL,data).result
             
            self.shape = shapely.wkb.loads(result[0][0], hex=True)
            
            file = open(fname, 'w')
            pickle.dump(self.shape, file)
            file.close()
            
    def __repr__(self):
        return self.city + ' boundary.'

# <codecell>

class Highway:
    def __init__(self,city,fresh=None):
        '''Initiate the Highway class.
        
        Parameters
        ----------
            
        city: string
            Name of the city to lookup highway graph for.
        fresh: boolean
            None: (default) Read processed highway graph from cache file
            True: Read and construct highway graph from the database (may take longer)
        
        Returns
        -------
        
        self.S: networkx DiGraph
            Simplified networkx highway graph.
        self.nodes: list of tuples
            List of tuples of (longitude, latitude)
        
        If fresh = True, also creates the following:
        
        self.F: networkx DiGraph
            Full networkx highway graph.
        self.result: list of tuples
            List of tuples of (way,osm_id,highway,oneway,width,lanes).
        '''
        
        self.city = city
        fname = 'cities/osm_gb/{0}/highway.gz'.format(self.city)
        if os.path.isfile(fname) and fresh == None:
            self.S, self.nodes = nx.read_gpickle(fname)
        else:
            SQL = """SELECT r.way, r.osm_id, r.highway, r.oneway, r.width, r.tags->'lanes' FROM 
                    planet_osm_line AS r,
                    (SELECT way FROM planet_osm_polygon WHERE name = %s AND boundary = %s 
                        ORDER BY ST_area(way) DESC LIMIT 1) AS s 
                    WHERE r.highway <> %s AND ST_Intersects(r.way, s.way);"""            

            data = (self.city,'administrative','',)
            
            self.result = Query(SQL,data).result
            
            # All nodes that appear on the edges 
            all_nodes = []
            
            # Compile a list of unique nodes
            for row in self.result:
                s=shapely.wkb.loads(row[0], hex=True)
                for c in s.coords[:]:
                    all_nodes.append(c)
                    
            # Unique nodes that appear on the edges after the duplicate nodes are removed
            self.nodes = list(set(all_nodes))
            
            # Call function to create a dictionary to enable node index lookup called self.node_dict
            node_dict = self.node_dict()
            
            # Count the number of times that a node is used
            node_count = {}
            for i,j in all_nodes:
                try:
                    node_count[node_dict[i][j]] = node_count[node_dict[i][j]] + 1
                except KeyError:
                    node_count[node_dict[i][j]] = 1
            
            # Find the nodes that are part of an intersection if they appear more than once
            junctions = []
            for n in node_count:
                if node_count[n] > 1:
                    junctions.append(n)

            # Create full networkx graph
            self.F=nx.DiGraph()
            
            # Create simplified networkx graph
            self.S=nx.DiGraph()

            for way,osm_id,highway,oneway,width,lanes in self.result:
                
                # Convert to shapely format
                s=shapely.wkb.loads(way, hex=True)
            
                # Create list of node indices for a path in the row
                node_indices = [node_dict[i][j] for i,j in s.coords]
                
                # Begin the FULL graph construction
                for this_node, that_node in zip(node_indices[:-1],node_indices[1:]):
                    
                    # Create list of edges (node pairs) for the row
                    foreward = [(this_node, that_node)]
                    backward = [(that_node, this_node)]
                    
                    # Call funtion to work out the distance for this edge using haversine formula
                    distance = self.haversine_distance(self.nodes[this_node],self.nodes[that_node])
                    
                    # Call funtion to determine the edges to add using the OSM oneway protocol
                    edges = self.oneway_edges(oneway,foreward,backward)
    
                    # Add edges to the FULL graph
                    self.F.add_edges_from(edges,osm_id=osm_id,highway=highway,oneway=oneway,width=width,lanes=lanes,distance=distance)
                
                # Now begin the SIMPLIFIED graph construction
                this_node = node_indices[0]
                distance = 0
                
                for that_node in node_indices[1:]:
                    # Call function do determine distance of the current edge to add to the sum of edges we are removing
                    distance = distance + self.haversine_distance(self.nodes[this_node],self.nodes[that_node])
                    
                    # If the that_node is a node at an intersection then complete the edge and create a new one
                    if that_node in junctions:
                        foreward = [(this_node, that_node)]
                        backward = [(that_node, this_node)]
                        
                        # Call funtion to determine the edges to add using the OSM oneway protocol
                        edges = self.oneway_edges(oneway,foreward,backward)
                        
                        # Add edges to the SIMPLIFIED graph
                        self.S.add_edges_from(edges,osm_id=osm_id,highway=highway,oneway=oneway,width=width,lanes=lanes,distance=distance)

                        # Start a new edge
                        this_node = that_node
        
                        # Reset distance to zero
                        distance = 0                
            
            print 'Number of edges BEFORE removing intermediate nodes ', self.F.number_of_edges()
            print 'Number of edges AFTER removing intermediate nodes ', self.S.number_of_edges()
            
            # Just save the simplified version as we only need that for our simulations
            nx.write_gpickle([self.S, self.nodes],fname)

    def __repr__(self):
        return self.city + ' highway.'

    def draw(self,graph='simple'):
        '''Draws the network layout of the graph.
        
        Parameters
        ----------
            
        graph: string
            A string that describes whether to draw the full graph or the simple graph
                simple: (Default) draw simplified graph
                full: draw full graph
        
        Returns
        -------
        
        fig: figure
            Figure of the network graph.'''

        if graph == 'full':
            G = self.F
        elif graph == 'simple':
            G = self.S

        # This is of highways that we know and classified
        # Any unaccounted highways will be assigned 7.
        color_render = {'motorway': 1,
                        'motorway_link': 1,
                        'trunk': 2,
                        'trunk_link': 2, 
                        'primary': 3,
                        'primary_link': 3,
                        'secondary': 4,
                        'secondary_link': 4,
                        'tertiary': 5,
                        'tertiary_link': 5,
                        'residential': 6,
                        'service': 6,
                        'services': 6,
                        'track': 6,
                        'unclassified': 6,
                        'road': 6,
                        'living_street': 6,
                        'pedestrian': 7,
                        'path': 7,
                        'raceway': 7,
                        'proposed': 7,
                        'steps': 7,
                        'footway': 7,
                        'bridleway': 7,
                        'bus_stop': 7,
                        'construction': 7,
                        'cycleway': 7,
                        }
        
        highways = self.count_features('highway',graph)
        
        for highway,count in highways:
            try:
                # Just checking to see if the key exists
                color_render[highway]
            except KeyError:
                color_render[highway] = 7
                print 'We are going to assume that ', highway , '  is of type 7.'    
        
        # Classify roads into big and small roads for the purpose of drawing the map
        edge_list = {}        
        for i in range(1,8):
            edge_list[i]=[(u,v) for (u,v,d) in G.edges(data=True) if color_render[d['highway']] == i]
        
        edge_dict = {1:{'alpha':1.00,'edge_color':'LightSkyBlue'},
                     2:{'alpha':0.81,'edge_color':'DarkOliveGreen'},
                     3:{'alpha':0.64,'edge_color':'IndianRed'},
                     4:{'alpha':0.49,'edge_color':'GoldenRod'},
                     5:{'alpha':0.36,'edge_color':'Gold'},
                     6:{'alpha':0.25,'edge_color':'Gray'},
                     7:{'alpha':0.16,'edge_color':'Pink'}}
        
        # Generate the figure
        fig = plt.figure()
        ax = plt.axes()#(xlim=(self.l, self.r), ylim=(self.b, self.t))
        # Reversing so that the smaller roads are drawn first
        for i in reversed(edge_dict.keys()):
            nx.draw_networkx_edges(G,pos=self.nodes,arrows=False,edgelist=edge_list[i],**edge_dict[i])
            
        # Draw the boundary of the city
        b=Boundary(self.city)
        x,y=b.shape.exterior.xy
        plt.plot(x,y,alpha=0.5)
        
        #extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        #plt.savefig(self.mapname, bbox_inches=extent, dpi=300)
        return fig

    def oneway_edges(self,oneway,foreward,backward):
        '''This is the encouraged OpenStreetMap protocol with most number of entries in the database.
        
        Parameters
        ----------
            
        oneway: string
            A string that describes if the street is one way or not.
            OpenStreetMap encourages using:
                oneway=yes (discouraged alternative: "true", "1")
                oneway=no (discouraged alternative: "false", "0")
                oneway=-1 (discouraged alternative: "reverse")
                oneway=reversible
            Source: http://wiki.openstreetmap.org/wiki/Key:oneway
        
        foreward: list of tuples
            A list of tuples of node indices in the foreward direction
            
        backward: list of tuples
            A list of tuples of node indices in the backward direction
        
        Returns
        -------
        
        edges: list of tuples
            A list of tuples of node indices appropriate to the OpenStreetMap protocol
            ignoring the use of discouraged alternatives and reversible clause as they occur
            very infrequently in the database and does not seem to be worth worrying about.'''
        
        if oneway == 'yes':
            edges = foreward
        elif oneway == -1:
            edges = backward
        else:
            edges = foreward+backward
            
        return edges
            
    def node_dict(self):
        '''Create dict of nodes for quick lookup of node index.
               
        Returns
        -------
        
        node_dict: dict
            A dictionary of longiture and latitude.     
            Do node_dict[lon][lat] to get the node number.
            
        
        Note
        ----
        
        This is a much faster way of looking up nodes compared to using a tuple list index.
        (At least a 100x faster!!!)'''
        
        node_dict = {}
        for i,node in enumerate(self.nodes):
            try:    
                node_dict[node[0]].update({node[1]:i})
            except KeyError:
                node_dict[node[0]] = {node[1]:i}
        
        return node_dict
            
    def count_features(self,feature,graph='simple'):
        '''Count and return the sorted number of edges with a given feature.
        
        Parameters
        ----------

        graph: string
            A string that describes whether to draw the full graph or the simple graph
                simple: (Default) draw simplified graph 
                full: draw full graph
        feature : string
            A string that describes an edge feature.
            
        Returns
        -------
        
        feature_count : list
            List of tuples (feature, number of times they appear in the graph).'''

        if graph == 'full':
            G = self.F
        elif graph == 'simple':
            G = self.S
        
        feature_list = []
        feature_count = []
        for i,j,d in G.edges(data=True):
            feature_list.append(d[feature])
        for f in set(feature_list):
            feature_count.extend([(f,feature_list.count(f))])
        return sorted(feature_count,key=lambda x: x[1])
            
    def haversine_distance(self,origin, destination):
        lon1, lat1 = origin
        lon2, lat2 = destination
        radius = 6371000 # metres
     
        dlat = math.radians(lat2-lat1)
        dlon = math.radians(lon2-lon1)
        a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
            * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        d = radius * c
     
        return d            
            
