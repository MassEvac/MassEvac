# MassEvac v4
import psycopg2
import os
import pickle
import math
import gzip
import shapely.wkb
import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from osgeo import gdal
from matplotlib import mlab
from scipy.misc import imread
from shapely.geometry import LineString,Point,mapping

''' Load the database configuration.
'''
try:
    with open('.dbconfig','r') as file:
        config=json.load(file)
except IOError:
    print   """ There is no .dbconfig file.

                Modify and paste the following into the working directory to create the file.

                with open('.dbconfig','w') as file:
                    json.dump({'dbname':'osm_gb','host':'localhost','user':'username','password':'password'},file,indent=True)
            """

def folder(place):
    ''' Returns the folder where settings for a place are saved.

        Input
        -----
            place: String
                Name of place
        Returns
        -------
            folder: String
                Name of the corresponding folder
    '''
    d = 'db/{0}/{1}'.format(config['dbname'],place)
    if not os.path.isdir(d):
        os.makedirs(d)
    return d

class Query:
    ''' Queries the PostgreSQL database.

        Inputs
        ------
            SQL: string
                The SQL query
            data: tuple (optional)
                Tuple containing variables in the SQL query

        Properties
        ----------
            self.result: list
                List containing the results of the SQL query
    '''

    def __init__(self,SQL,data=None):
        self.connect()
        self.SQL = SQL
        self.data = data
        self.query()
    
    def connect(self):
        try:
            con = psycopg2.connect(str.join(' ',[k+'='+config[k] for k in config]))
            self.cur = con.cursor()
        except Exception, e:
            print 'Oh dear! Check .dbconfig file or the query!'
            print e
    
    def query(self):
        if self.data == None:
            try:
                self.cur.execute(self.SQL)
                print self.SQL
            except Exception, e:
                print e
        else:
            try:
                self.cur.execute(self.SQL,self.data)
                print self.cur.mogrify(self.SQL,self.data)
            except Exception, e:
                print e
        
        try:
            self.result = self.cur.fetchall()
        except ProgrammingError:
            pass
    
    def __len__(self):
        return len(self.result)
    
    def __repr__(self):
        for record in self.result:
            print record
        return 'End of record.'

class Population:
    def __init__(self,place,fresh=False):
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
        self.place = str(place)
        self.fresh = fresh

        self.boundary = Boundary(place).shape
        
        fname = '{0}/population_grumpv1'.format(folder(self.place))
        if os.path.isfile(fname) and not self.fresh == True:
            print '{0}: Loading {1}'.format(self.place,fname)
            with open(fname, 'r') as file:
                self.x,self.y,self.lon,self.lat,self.pop=pickle.load(file)
        else:
            print '{0}: Processing {1}'.format(self.place,fname)
            SQL = """SELECT x, y, val, geom FROM
                        (SELECT (ST_PixelAsCentroids(ST_Clip(q.raster,q.clipper))).* FROM
                            (SELECT p.rast AS raster, b.way as clipper
                                FROM population_grumpv1 AS p,
                                    (SELECT ST_GeomFromText(%s,4326) AS way) AS b
                                WHERE ST_Intersects(p.rast,b.way)) AS q) AS foo;"""
            data = (self.boundary.wkt,)                    
            result = Query(SQL,data).result
            self.x,self.y,self.pop,geom = zip(*result)
            points=[shapely.wkb.loads(i,hex=True) for i in geom]
            self.lon,self.lat = zip(*[(i.x, i.y) for i in points])
            print '{0}: Writing {1}'.format(self.place,fname)
            with open(fname, 'w') as file:
                pickle.dump([self.x,self.y,self.lon,self.lat,self.pop], file)
    
    def __repr__(self):
        return self.place + ' with ' + str(len(self.pop)) + ' data points.'
    
    def fig(self):
        ''' Draw figure of raw poulation data from the database.
        '''
        axs = plt.axes(xlim=(min(self.lon),max(self.lon)), ylim=(min(self.lat), max(self.lat)))
        plt.scatter(self.lon,self.lat,self.pop,c=self.pop)
        plt.colorbar()
        return fig
    
    def nearest_node(self,x,y):
        ''' Function to get the node number nearest to a prescribed node.

            Inputs
            ------
                x: float
                    Longitude
                y: float
                    Latitude
            Outputs
            -------
                nearest: int
                    Integer with the index in the NumPy array 
                    nearest to the input latitude and longitude.
        '''
        distance=np.sqrt(np.square(self.lon-x)+np.square(self.lat-y))
        nearest=min(range(len(distance)), key=distance.__getitem__)
        return nearest

class Boundary:
    def __init__(self,place,fresh=False):
        ''' Queries the PostgreSQL database for boundary co-ordinates of input place.

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
                self.shape: Shapely object
                    Shapely object with the boundary information
        '''        
        self.bbox = None
        if type(place) == tuple:
            self.bbox = place

        self.place = str(place)
        self.fresh = fresh
        
        fname = '{0}/boundary'.format(folder(self.place))
        if os.path.isfile(fname) and not self.fresh == True:
            print '{0}: Loading {1}'.format(self.place,fname)
            with open(fname, 'r') as file:
                self.shape = pickle.load(file)
        else:
            print '{0}: Processing {1}'.format(self.place,fname)
            if self.bbox:
                SQL = 'SELECT ST_MakeEnvelope(%s,%s,%s,%s,4326)'
                data = self.bbox
            else:
                SQL = 'SELECT way FROM planet_osm_polygon WHERE name = %s AND boundary = %s ORDER BY ST_NPoints(way) DESC LIMIT 1'
                data = (self.place,'administrative',)
            
            result = Query(SQL,data).result
            
            self.shape = shapely.wkb.loads(result[0][0], hex=True)
            
            print '{0}: Writing {1}'.format(self.place,fname)
            with open(fname, 'w') as file:
                pickle.dump(self.shape, file)
    
    def __repr__(self):
        return self.place + ' boundary.'

class Highway:
    def __init__(self,place,graph='lite',fresh=False):
        ''' Loads Highway object from OpenStreetMap PostgreSQL database.
        
            Inputs
            ------    
                place: string or tuple (xmin, ymin, xmax, ymax)
                    Name of the polygon on OpenStreetMap being queried
                    Alternatively, input tuple with boundary box coordinates
                graph
                    'full': self.G = Full graph
                    'lite': self.G = Simplified graph
                fresh: boolean
                    False: (default) Read processed highway graph from cache file
                    True: Read and construct highway graph from the database (may take longer)
                
            Properties
            ----------    
                self.G: networkx DiGraph
                    Simplified networkx highway graph.
                self.G.edges: list of tuples
                    List of tuples containing ri, ci, di,etc.
                self.G.nodes: list of tuples
                    List of tuples of (longitude, latitude)
                self.boundary:
                    Boundary object
        '''
        self.place = str(place)
        self.boundary = Boundary(place).shape
        self.fresh = fresh
        self.mapname = folder(self.place)+'/highway.{0}.png'
        self.graph = graph
        self.Graph = {}        
        # Process highways if edge or node cache is not available or the 'fresh' boolean is True.
        if not (self.edges_file(mode='load') and self.nodes_file(mode='load') and self.hiclass_file(mode='load') and self.fresh == False):
            ''' This is how we classify different highway tags into different categories.'''        
            # Assumptions
            # -----------
            # Standard with per lane in metres
            self.assumed_width_per_lane = 2.5
            # Assumed number of lanes per hiclass
            self.assumed_lanes = np.array([3,2,2,1.5,1,1,0.5])
            # Widths of the 7 types of highways
            self.assumed_width = self.assumed_lanes*self.assumed_width_per_lane
            # The highway class processing takes some time, to generate such a short list.
            # Big time savings to be had from saving it to a file.
            # In the future, it may be worth having a giant list for all cities!
            # This is of highways that we know and classified
            self.hiclass = {'motorway': 0,
                            'motorway_link': 0,
                            'trunk': 1,
                            'trunk_link': 1,
                            'primary': 2,
                            'primary_link': 2,
                            'secondary': 3,
                            'secondary_link': 3,
                            'tertiary': 4,
                            'tertiary_link': 4,
                            'residential': 5,
                            'service': 5,
                            'services': 5,
                            'track': 5,
                            'unclassified': 5,
                            'road': 5,
                            'living_street': 5,
                            'pedestrian': 6,
                            'path': 6,
                            'raceway': 6,
                            'proposed': 6,
                            'steps': 6,
                            'footway': 6,
                            'bridleway': 6,
                            'bus_stop': 6,
                            'construction': 6,
                            'cycleway': 6,
                            }
            # 1 agent per metre squared is assumed to be the
            #  minimum area so that agents do not get stuck.
            # I am assuming that it only affects fraction of edges!
            # 1 agent / 1 agent per metre square = 1 metre square
            EA_min = 1.0 # metre squared
            print '{0}: Processing highway.'.format(place)
            SQL = """SELECT r.way, r.osm_id, r.highway, r.oneway, r.width, r.tags->'lanes' FROM
                    planet_osm_line AS r,
                    (SELECT ST_GeomFromText(%s,4326) AS way) AS s
                    WHERE r.highway <> %s AND ST_Intersects(r.way, s.way);"""
            data = (self.boundary.wkt,'',)
            self.result = Query(SQL,data).result
            # All nodes that appear on the edges
            all_nodes = []
            # Compile a list of unique nodes
            for row in self.result:
                s=shapely.wkb.loads(row[0], hex=True)
                for c in s.coords[:]:
                    all_nodes.append(c)
            # Unique nodes that appear on the edges after the duplicate nodes are removed
            nodes = list(set(all_nodes))
            # Reverse the key,value lookup
            node_lookup = dict(zip(nodes,range(len(nodes))))
            # Count the number of times that a node is used
            node_count = {}
            for coord in all_nodes:
                try:
                    node_count[node_lookup[coord]] = node_count[node_lookup[coord]] + 1
                except KeyError:
                    node_count[node_lookup[coord]] = 1
            # Identify the nodes that are part of an intersection if they appear more than once
            junctions = []
            for n in node_count:
                if node_count[n] > 1:
                    junctions.append(n)
            # Initialise full networkx graph
            self.Graph['full']=nx.DiGraph()
            # Initialise simplified networkx graph
            self.Graph['lite']=nx.DiGraph()
            for way,osm_id,highway,oneway,width,lanes in self.result:
                # Convert to shapely format
                s=shapely.wkb.loads(way, hex=True)
                # Create list of node indices for a path in the row
                node_indices = [node_lookup[coord] for coord in s.coords]
                # Begin the FULL graph construction
                for this_node, that_node in zip(node_indices[:-1],node_indices[1:]):
                    # Create list of edges (node pairs) for the row
                    foreward = [(this_node, that_node)]
                    backward = [(that_node, this_node)]
                    # Call funtion to work out the distance for this edge using haversine formula
                    distance = self.haversine_distance(nodes[this_node],nodes[that_node])
                    # Call funtion to determine the edges to add using the OSM oneway protocol
                    edges = self.oneway_edges(oneway,foreward,backward)
                    # Determine the hiclass and if not, assume it is 6
                    try:
                        hiclass = self.hiclass[highway]
                    except KeyError:
                        hiclass = self.hiclass[highway] = 6
                        print '{0}: Highway class 6 will be assumed for: {1}'.format(self.place,highway)
                    # Calculate assumed width
                    assumed_width = self.assumed_width[hiclass]
                    # Calculate area
                    area = distance*assumed_width
                    if area < EA_min:
                        area = EA_min
                    # Add edges to the FULL graph                    
                    self.Graph['full'].add_edges_from(edges,osm_id=osm_id,highway=highway,oneway=oneway,width=width,lanes=lanes,distance=distance,hiclass=hiclass,assumed_width=assumed_width,area=area)
                # Now begin the SIMPLIFIED graph construction
                this_node = node_indices[0]
                last_node = this_node
                distance = 0
                for that_node in node_indices[1:]:
                    # Call function to determine distance of the current edge to add to the sum of edges we are removing
                    distance = distance + self.haversine_distance(nodes[last_node],nodes[that_node])
                    last_node = that_node
                    # If the that_node is a node at an intersection then complete the edge and create a new one
                    if that_node in junctions or that_node == node_indices[-1]:
                        foreward = [(this_node, that_node)]
                        backward = [(that_node, this_node)]
                        # Call funtion to determine the edges to add using the OSM oneway protocol
                        edges = self.oneway_edges(oneway,foreward,backward)
                        # Determine the hiclass and if not, assume it is 6
                        hiclass = self.hiclass[highway]
                        # Calculate assumed width
                        assumed_width = self.assumed_width[hiclass]
                        # Calculate area
                        area = distance*assumed_width
                        if area < EA_min:
                            area = EA_min                        
                        # Add edges to the SIMPLIFIED graph
                        self.Graph['lite'].add_edges_from(edges,osm_id=osm_id,highway=highway,oneway=oneway,width=width,lanes=lanes,distance=distance,hiclass=hiclass,assumed_width=assumed_width,area=area)
                        # Start a new edge
                        this_node = that_node
                        # Reset distance to zero
                        distance = 0
            print 'Number of edges BEFORE removing intermediate nodes ', self.Graph['full'].number_of_edges()
            print 'Number of edges AFTER removing intermediate nodes ', self.Graph['lite'].number_of_edges()
            # Save the edge and node list
            # We save the edge list as a 'list' to preserve
            # the order as some elements rely on it.
            # Saving the 'DiGraph' or a 'dict' messes up the order.
            for k in self.Graph:
                for n in self.Graph[k].nodes_iter():
                    self.Graph[k].node[n] = nodes[n]
                self.G = self.Graph[k]
                self.init_pop_dist()
            self.hiclass_file(mode='cache')
            self.edges_file(mode='cache')
            self.nodes_file(mode='cache')
            self.G = self.Graph[self.graph]            
        lon,lat=np.array(zip(*[v for u,v in self.G.nodes(data=True)]))
        self.l,self.r=min(lon),max(lon)
        self.b,self.t=min(lat),max(lat)

    def hiclass_file(self,mode):
        ''' Cache or load hiclass.'''
        fname = '{0}/highway.class'.format(folder(self.place))        
        if mode == 'cache':
            print '{0}: Writing {1}'.format(self.place,fname)
            with open(fname, 'w') as file:
                pickle.dump(self.hiclass, file)        
        elif mode == 'load':
            if os.path.isfile(fname):        
                print '{0}: Loading {1}'.format(self.place,fname)
                with open(fname, 'r') as file:
                    self.hiclass = pickle.load(file)
                return True
            else:
                return False

    def edges_file(self,mode):
        ''' Cache or load edges.'''
        # If self.graph = full:
        #   Read or write both full and lite graph
        # If self.graph = lite:
        #   Read or write only the lite graph
        for graph in ['lite','full']:
            # Skip if self.graph == 'lite'
            if self.graph == 'lite' and graph == 'full':
                continue
            fname = '{0}/highway.{1}.edges.gz'.format(folder(self.place),graph)
            if mode == 'cache':
                print '{0}: Writing {1}'.format(self.place,fname)
                with gzip.open(fname, 'w') as file:
                    pickle.dump(self.Graph[graph].edges(data=True), file)
            elif mode == 'load':
                try:
                    print '{0}: Loading {1}'.format(self.place,fname)
                    with gzip.open(fname, 'r') as file:
                        self.G = self.Graph[graph] = nx.DiGraph(pickle.load(file))
                    if self.graph == graph:
                        return True
                except IOError:
                    return False

    def nodes_file(self,mode):
        ''' Cache or load nodes.'''
        # If self.graph = full:
        #   Read or write both full and lite graph
        # If self.graph = lite:
        #   Read or write only the lite graph
        for graph in ['lite','full']:
            # Skip if self.graph == 'lite'
            if self.graph == 'lite' and graph == 'full':
                continue                  
            fname = '{0}/highway.{1}.nodes.gz'.format(folder(self.place),graph)
            if mode == 'cache':
                print '{0}: Writing {1}'.format(self.place,fname)
                with gzip.open(fname, 'w') as file:
                    pickle.dump(self.Graph[graph].nodes(data=True), file)              
            elif mode == 'load':
                try:
                    print '{0}: Loading {1}'.format(self.place,fname)
                    with gzip.open(fname, 'r') as file:
                        nodes = dict(pickle.load(file))
                        for n in self.Graph[graph].nodes_iter():
                            self.Graph[graph].node[n] = nodes[n]
                    if self.graph == graph:
                        return True
                except IOError:
                    return False

    def nearest_destin_from_edge(self,edge_index):
        ''' Determines the nearest destin index and the distance from input edge index.

            Input
            -----
                edge_index: int
                    Edge index
            Output
            ------
                nearest_destin: int
                    Nearest destin node index
                distance_destin: float
                    Distance to the nearest destin 
        '''
        # Initiate the route file if routes are not present
        try:
            self.route_length
        except AttributeError:                
            self.init_route()
        # Extract edge information 
        u,v,d = self.edges[edge_index]
        # Determine nearest catchment areas
        nearest_destin = None
        distance_destin = None
        for this_destin in self.destins:
            try:
                # Take the mean of distance from u and v
                this_dist = max(self.route_length[this_destin][u],self.route_length[this_destin][v])
                print this_destin, self.route_length[this_destin][u], self.route_length[this_destin][v]
                # If distance_destin is greater than this distance then replace
                # Give distance_destin a number if not done yet                                    
                if distance_destin > this_dist or distance_destin == None:
                    nearest_destin = this_destin
                    distance_destin = this_dist
            except KeyError:
                pass
        return nearest_destin, distance_destin

    def geojson_edges(self, fname, properties={}):
        ''' Produces a geojson file with feature tag.

            Inputs
            ------
                properties: List
                    List of dicts where the index corresponds to edge index
                fname:  String
                    Path to the file where the geojson file will be dumped
            Output
            ------
                geojson file
        '''
        features = []
        for u,v,d in self.G.edges_iter(data=True):
            try:
                p = properties[(u,v)]
            except KeyError:
                p = {}
            # Generate properties
            p["u"] = u
            p["v"] = v
            # Determine highway class and corresponding assumed width          
            p.update(d)
            l = LineString([self.G.node[u],self.G.node[v]])
            feature = {
                "type": "Feature",
                "properties": p,
                "geometry": mapping(l)
            }
            features.append(feature)
        out = {
            "type": "FeatureCollection",
            "features": features
        }
        with open (fname,'w') as file:
            json.dump(out,file,indent=True)     

    def geojson_nodes(self, fname, properties):
        ''' Produces a geojson file with feature tag.

            Inputs
            ------
                properties: List
                    List of dicts where the index corresponds to node index
                fname:  LineString
                    Path to the file where the geojson file will be dumped
            Output
            ------
                geojson file
        '''
        features = []
        for n in properties:
            p = properties[n]
            p['index'] = n
            feature = {
                "type": "Feature",
                "properties": p,
                "geometry": mapping(Point(self.G.node[n]))
            }
            features.append(feature)
        out = {
            "type": "FeatureCollection",
            "features": features
        }
        with open (fname,'w') as file:
            json.dump(out,file,indent=True)   

    def __repr__(self):
        return self.place + ' highway.'
    
    def init_destins(self):
        ''' Returns a list of destin nodes.
        
            Returns
            -------        
                destins: list
                    A list of node numbers that refer to the destination.
        '''
        fname = '{0}/highway.destins'.format(folder(self.place))
        if os.path.isfile(fname) and not self.fresh == True:
            print '{0}: Loading {1}'.format(self.place,fname)
            with open(fname, 'r') as file:
                self.destins = pickle.load(file)
        else:
            print '{0}: Processing {1}'.format(self.place,fname)
            SQL = """SELECT l.way,l.osm_id,l.highway,l.oneway FROM planet_osm_line AS l,
                     (SELECT ST_GeomFromText(%s,4326) AS way) AS p
                     WHERE ST_intersects(ST_ExteriorRing(p.way),l.way)
                     AND l.highway IN ('motorway','motorway_link','trunk','trunk_link','primary','primary_link')"""
            data = (self.boundary.wkt,)
            r = Query(SQL,data).result
            node_lookup = dict(zip(self.G.node.values(),self.G.node.keys()))
            self.destins = []
            # Determine the centroidal node of the graph from which to trace all our routes from
            # (Assuming that all destinations are accessible from the centroid)
            # For which one would use the largest component, omitted here
            # Get the list of nodes of the biggest connected component in the graph
            largest_component_nodes=sorted(nx.strongly_connected_components(self.G),key=len,reverse=True)[0]
            centroid = self.nearest_node((self.l-self.r)/2+self.r,(self.t-self.b)/2+self.b,largest_component_nodes)
            # Find path length from the centroidal node to all other nodes
            route_len_from_centroid=nx.single_source_dijkstra_path_length(self.G,centroid)
            for way,osm_id,highway,oneway_tag in r:
                s = shapely.wkb.loads(way, hex=True)
                # Determine the longitude and latitude of the first and last node
                this_coord = s.coords[0]
                that_coord = s.coords[-1]
                destin_coord = {1: [that_coord],
                                0: [this_coord, that_coord],
                               -1: [this_coord]}
                oneway = self.oneway(oneway_tag)
                # And if the last node of a path is outside the place boundary
                for d in destin_coord[oneway]:
                    if not self.boundary.intersects(shapely.geometry.Point(d)):
                        destin_node = node_lookup[d]
                        # If the destin node can be accessed from the centroidal node, accept, otherwise reject
                        try:
                            route_len_from_centroid[destin_node]
                            self.destins.append(destin_node)
                        except KeyError:
                            pass
            print '{0}: Writing {1}'.format(self.place,fname)
            with open(fname, 'w') as file:
                pickle.dump(self.destins, file)

    def init_route(self,save=True):
        ''' Initialise route and route length dictionaries
            
            Properties
            ----------
                self.route: dict
                    Call self.route[destin][this_node] to retrieve the next_node to that destin
                self.route_length: dict
                    Call self.route_length[destin][this_node] to retrieve the distance to the destin
            Notes
            -----
                Loading cache now seems to be twice as fast as precomputing
        '''
        self.route = {}
        self.route_length = {}
        try:
            self.destins
        except AttributeError:
            self.init_destins()
        # We are reversing so that we can determine the shortest path to a single sink rather than from a single source
        GT = self.G.reverse(copy=True)
        self.route_folder = '{0}/highway.{1}.route'.format(folder(self.place),self.graph)
        if not os.path.isdir(self.route_folder):
            os.mkdir(self.route_folder)
        for destin in self.destins:
            fname = '{0}/{1}'.format(self.route_folder,destin)
            if os.path.isfile(fname) and not self.fresh == True:
                print '{0}: Loading {1}'.format(self.place,fname)
                with open(fname, 'r') as file:
                    self.route[destin],self.route_length[destin] = pickle.load(file)
            else:
                print '{0}: Processing {1}'.format(self.place,fname)
                # Calculate a single SINK shortest path and path length
                # Determine shortest paths by reversing the direction of the paths (from the transpose of the graph)
                self.route_length[destin],path=nx.single_source_dijkstra(GT,destin,weight='distance')
                self.route[destin] = {}
                # Only store the next node to go to from any given node to reduce memory consumption
                for node in path:
                    try:
                        self.route[destin][node] = path[node][-2]
                    except IndexError:
                        pass
                if save:        
                    # Dump the results to a file
                    print '{0}: Writing {1}'.format(self.place,fname)
                    with open(fname, 'w') as file:
                        pickle.dump([self.route[destin],self.route_length[destin]], file)
        # Determine the list of all destination edges
        for u,v in self.G.edges_iter():
            # Iterate over all destins and find the maximum distance between node u and v
            destin_distance = {}
            for x in self.destins:
                try:
                    destin_distance[x] = max(self.route_length[x][u],self.route_length[x][v])
                except KeyError:
                    pass
            # Assign the nearest destin to the edge
            try:                    
                self.G[u][v]['nearest_destin'] = min(destin_distance,key=destin_distance.__getitem__)
                self.G[u][v]['destin_distance'] = destin_distance
            except ValueError:
                pass
        return 'Route to destinations initialised.'
    
    def nearest_node(self,x,y,nodelist=None):
        ''' Function to get the node number nearest to a prescribed node.

            Inputs
            ------
                x: float
                    Longitude
                y: float
                    Latitude
                nodelist: list
                    List with indices of nodes to narrow the search within
            Outputs
            -------
                nearest: int
                    Node index integer nearest to input latitude and longitude.
        '''
        all_nodes,coords=zip(*[(n,c) for n,c in self.G.nodes(data=True)])
        if nodelist==None:
            nodelist = all_nodes    
        lon,lat = np.array(zip(*coords))
        # Function to get the node number nearest to a prescribed lon and lat
        distance=np.sqrt(np.square(lon-x)+np.square(lat-y))
        nearest=min(range(len(distance)), key=distance.__getitem__)
        return nodelist[nearest]
    
    def init_pop_dist(self):
        ''' Function to initialise population.
        '''
        print '{0}: Processing population distribution.'.format(self.place)
        # Get population distribution from the database
        pdb = Population(self.place,fresh=self.fresh)
        # List to record the total length of links per node on population grid
        total_area = np.zeros(len(pdb.pop))
        # List to record which node on population grid the link is nearest to
        which_pop_node = {}
        count = 0
        # Determine the nearest node on population grid for every link
        for u,v,d in self.G.edges_iter(data=True):
            midpoint = sum(np.array([self.G.node[u],self.G.node[v]]))/2
            # Determine the nearest population node
            pi = pdb.nearest_node(*midpoint)
            which_pop_node[(u,v)] = pi
            # Sum the total length of the road segment
            total_area[pi] += d['area']
            count = count + 1
            if count%100 == 0:
                print count, ' edges processed...\r',
        # Prepare the output
        pop_indices = set(which_pop_node.values())
        # This is not the total population but rather the 
        total_accounted_pop = sum(pdb.pop[k] for k in pop_indices)
        print 'Total accounted population vs total population according to GRUMPv1 = ', total_accounted_pop, sum(pdb.pop)
        # Now distribute the population proportional to the length of link
        for u,v,d in self.G.edges_iter(data=True):
            pi = which_pop_node[(u,v)]
            self.G[u][v]['pop_dist'] = pdb.pop[pi]/total_accounted_pop * d['area']/total_area[pi]
        sanity_check = 0
        for u,v,d in self.G.edges_iter(data=True):
            sanity_check+=d['pop_dist']
        print 'Sanity check value should be close to 1 = ', sanity_check
        print 'Run self.fig_pop_dist() to view the population distribution.'
    
    def fig_pop_dist(self):
        ''' Generate the figure for processed population.
        '''
        edgelist,edgewidth = zip(*[((u,v),d['pop_dist'])for u,v,d in self.G.edges(data=True)])
        thickest_line_width = 2
        edgewidth=np.array(edgewidth)/max(edgewidth)*thickest_line_width
        nx.draw_networkx_edges(self.G,pos=self.G.node,arrows=False,edgelist=edgelist,edge_color=edgewidth,width=edgewidth,alpha=1.0)
    
    def fig_destins(self):
        ''' Returns highway map figure with the exits nodes numbered.
        '''
        fig = self.fig_highway()
        for label in self.destins:
            x,y = self.G.node[label]
            plt.annotate(
                label,
                xy = (x, y), xytext = (20,20),
                textcoords = 'offset points', ha = 'left', va = 'bottom',
                bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
                arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
        return fig
    
    def img_highway(self,theme='default'):
        ''' Returns the image of the highway in array-like format for faster reading.
        
            Returns
            -------
                image: NumPy array
                    Image of the highway
        '''
        fname = self.mapname.format(theme)
        if not os.path.isfile(fname) or self.fresh == True:
            fig = self.fig_highway()
            plt.close(fig)
        print '{0}: Loading {1}'.format(self.place,fname)
        return imread(fname)
    
    def fig_highway(self,theme='default'):
        ''' Draws the network layout of the graph.
        
            Parameters
            ----------
                theme: string
                    Type of colour scheme, 'default' or 'greyscale'        
            Returns
            -------    
                fig: figure
                    Figure of the network graph.
        '''
        fname = self.mapname.format(theme)
        print '{0}: Processing {1}'.format(self.place,fname)
        edge_dict = {'default':{
                                 0:{'alpha':1.00,'edge_color':'LightSkyBlue'},
                                 1:{'alpha':1.00,'edge_color':'DarkOliveGreen'},
                                 2:{'alpha':1.00,'edge_color':'IndianRed'},
                                 3:{'alpha':0.8,'edge_color':'GoldenRod'},
                                 4:{'alpha':0.6,'edge_color':'Gold'},
                                 5:{'alpha':0.4,'edge_color':'Gray'},
                                 6:{'alpha':0.2,'edge_color':'Pink'}
                                 },
                     'greyscale':{
                                 0:{'alpha':1.00,'edge_color':'Gray'},
                                 1:{'alpha':1.00,'edge_color':'Gray'},
                                 2:{'alpha':1.00,'edge_color':'Gray'},
                                 3:{'alpha':0.8,'edge_color':'Gray'},
                                 4:{'alpha':0.6,'edge_color':'Gray'},
                                 5:{'alpha':0.4,'edge_color':'Gray'},
                                 6:{'alpha':0.2,'edge_color':'Gray'}
                                 }
                     }
        # Classify roads into big and small roads for the purpose of drawing the map
        edge_list = {}
        for i in range(len(edge_dict[theme])):
            edge_list[i]=[(u,v) for (u,v,d) in self.G.edges(data=True) if d['hiclass'] == i]
        # Generate the figure
        fig = plt.figure()

        ax = plt.axes(xlim=(self.l, self.r), ylim=(self.b, self.t),aspect='equal')
        # Reversing so that the smaller roads are drawn first
        for i in reversed(edge_dict[theme].keys()):
            nx.draw_networkx_edges(self.G,pos=self.G.node,arrows=False,edgelist=edge_list[i],**edge_dict[theme][i])
        # Draw the boundary of the place
        x,y=self.boundary.exterior.xy
        plt.plot(x,y,alpha=0.5)
        try:
            self.destins
        except AttributeError:
            self.init_destins()
        # Mark the destination nodes if they are available
        for d in self.destins:
            x,y = self.G.node[d]
            plt.scatter(x,y,s=200,c='g',alpha=0.5,marker='o')
        print '{0}: Writing {1}'.format(self.place,fname)
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        plt.savefig(fname, bbox_inches=extent, dpi=300)
        return fig
    
    def oneway(self,oneway_tag):
        ''' This is the adopted OpenStreetMap parsing protocol for derived from entries in the database.
        
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
            Returns
            -------
                1
                    if oneway
                0
                    if both ways
                -1
                    if oneway in opposite direction
        '''
        
        if oneway_tag == 'yes':
            return 1
        elif oneway_tag == '-1':
            return -1
        else:
            return 0
    
    def oneway_edges(self,oneway_tag,foreward,backward):
        ''' This is the adopted OpenStreetMap parsing protocol for derived from entries in the database.
        
            Inputs
            ------
                foreward: list of tuples
                    A list of tuples of node indices in the foreward direction
                backward: list of tuples
                    A list of tuples of node indices in the backward direction
            Returns
            -------
                edges: list of tuples
                    A list of tuples of node indices appropriate to the OpenStreetMap protocol
                    ignoring the use of discouraged alternatives and reversible clause as they occur
                    very infrequently in the database and does not seem to be worth worrying about.
        '''
        edges = {1: foreward,
                 0: foreward+backward,
                -1: backward}
        return edges[self.oneway(oneway_tag)]
    
    def count_features(self,feature):
        ''' Count and return the sorted number of edges with a given feature.
        
            Parameters
            ----------
                feature : string
                    A string that describes an edge feature.
            Returns
            -------
                feature_count : list
                    List of tuples (feature, number of times they appear in the graph).
        '''
        feature_list = []
        feature_count = []
        for u,v in self.G.edges():
            feature_list.append(self.G[u][v][feature])
        for f in set(feature_list):
            feature_count.extend([(f,feature_list.count(f))])
        return sorted(feature_count,key=lambda x: x[1])
    
    def haversine_distance(self,origin, destination):
        ''' Compute the haversine formula to calculate distance between two points on earth.
        '''
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
    
    def where_is_node(self,node):
        """Shows the location of the node and returns the edges it is connected to."""
        fig=self.fig_highway()
        x,y = self.G.node[node]
        plt.scatter(x,y,s=500,alpha=0.25,c='blue')
        plt.scatter(x,y,s=5,alpha=0.5,c='red')
    
    def where_is_edge(self,u,v):
        """Shows the location of the edge and returns the edges it is connected to."""
        fig=self.fig_highway()
        xu,yu = self.G.node[u]
        xv,yv = self.G.node[v]
        plt.scatter([xu,xv],[yu,yv],s=500,alpha=0.25,c='blue')
        nx.draw_networkx_edges(self.G,pos=self.G.node,arrows=False,edgelist=[(ri, ci)],edge_color='red',width=5,alpha=0.5)

class Flood:
    def __init__(self,filename):
        ''' Initiate the flood map by loading the file and establish the extents.

            Inputs
            ------
                filename: String
                    Location of geo-raster file which consists of binary flood map in WGS-84 4326 projection
            Properties
            ----------
                self.Map: NumPy array
                    NumPy 2D array containing the flood map
                self.ox: float
                    X origin
                self.oy: float
                    Y origin
                self.dx: float
                    X pixel resolution
                self.dy: float
                    Y pixel resolution
                self.rx: float
                    X rotation
                self.ry: float
                    Y rotation
                self.lx: int
                    Number of pixels in w-e direction
                self.ly: int
                    Number of pixels in n-s direction
                self.minx: float
                    Bottom-right y coordinate
                self.miny: float
                    Bottom-right y coordinate
                self.maxx: float
                    Top-left x coordinate
                self.maxy: float
                    Top-left y coordinate
        '''
        self.filename = filename
        # Read the GDAL file
        self.ds = gdal.Open(filename)
        # gt[0] /* top left x */
        # gt[1] /* w-e pixel resolution */
        # gt[2] /* rotation, 0 if image is "north up" */
        # gt[3] /* top left y */
        # gt[4] /* rotation, 0 if image is "north up" */
        # gt[5] /* n-s pixel resolution */ 
        self.ox, self.dx, self.rx, self.oy, self.ry, self.dy =  self.ds.GetGeoTransform()
        # Number of pixels in w-e and n-s directions
        self.lx = self.ds.RasterXSize
        self.ly = self.ds.RasterYSize
        # Establish the extents
        self.minx = self.ox
        self.miny = self.oy + self.ly*self.dy #  + self.lx*self.ry because rotation should be 0
        self.maxx = self.ox + self.lx*self.dx # + self.ly*self.rx because rotation should be 0
        self.maxy = self.oy
        # Read the flood level
        self.Map = self.ds.ReadAsArray()

    def __repr__(self):
        return self.filename

    def fig(self):
        plt.imshow(self.Map,extent=[self.minx,self.maxx,self.miny,self.maxy],vmin=0)        
        plt.colorbar()

    def coordinateIndex(self,c):
        ''' Returns index of x and y coordinate if we know the origin and pixel resolution.

        Parameters
        ----------
            c: tuple
                Longitude/Latitude
        Returns
        -------
            (int xi, int yi): tuple
                Indices of x and y coordinate
        '''
        xi = int((c[0]-self.ox)/self.dx)
        yi = int((c[1]-self.oy)/self.dy)
        return xi,yi

    def lineToCoordList(self,a,b):
        ''' Returns a list intermediate coordinates for an input line coordinates.

        Inputs
        ------
            a: tuple(int x0, int y0)
                Point a coordinates
            b: tuple(int x1, int y1)
                Point b coordinates
        Outputs
        -------
            [(int x0, int y0),...,(int xn, int yn),...,(int x1, int y1)]: list
                List of intermediate pixels
        '''
        x = np.array((a[0],b[0]))
        y = np.array((a[1],b[1]))   
        dx = np.abs(np.diff(x))
        dy = np.abs(np.diff(y))
        if a == b:
            result = [a]    
        else:
            if dy > dx:
                slope, intercept = np.polyfit(y,x,1)
                smaller = min(y)
                larger = max(y)
                result = [(int(round(slope*i+intercept)),i) for i in range(smaller,larger+1)]           
            else:
                slope, intercept = np.polyfit(x,y,1)
                smaller = min(x)
                larger = max(x)
                result = [(i,int(round(slope*i+intercept))) for i in range(smaller,larger+1)]
        return result

    def floodStatus(self,m,n):
        ''' Returns number of flooded and safe pixels.
        
        Inputs
        ------
            m: tuple(float m0, float y0)
                Point m lon/lat
            n: tuple(float x1, float y1)
                Point n lon/lat
        Returns
        -------
            (flooded, safe): tuple
                Number of flooded and safe pixels
        '''
        # Convert lon/lat to raster coordinates
        a = self.coordinateIndex(m)
        b = self.coordinateIndex(n)
        # Retrieve coordinates of intermediate raster pixels
        coordList = self.lineToCoordList(a,b)
        # Cross reference to check if intermediate cells are flooded
        yes = 0
        no = 0
        error = 0
        for i,j in coordList:
            try:
                if i < 0 or j < 0:
                    raise IndexError
                else:
                    if self.Map[j,i]>0:
                        yes+=1
                    else:
                        no+=1
            except IndexError:
                error+=1
        return yes,no,error

    def isFlooded(self,a,b):
        ''' Returns True if a link is flooded.

        Inputs 
        ------
            m: tuple(float x0, float y0)
                Point m lon/lat
            n: tuple(float x1, float y1)
                Point n lon/lat
        Outputs
        -------
            True: boolean
                If the link is flooded
            False: boolean
                If the link is not flooded          
        '''
        yes,no,error = self.floodStatus(a,b)
        if yes>0:
            return True
        else:
            return False