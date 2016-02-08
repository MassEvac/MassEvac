# Calculates network metrics of subgraphs ready for analysis

import sys
sys.path.append('core')
import abm
import db
reload(abm)
reload(db)

import networkx as nx
import gzip
import json
import time
import os
import socket
import numpy as np
import matplotlib.pyplot as plt
from functools import partial    
from multiprocessing import Pool, cpu_count

# ---------------------------------------------------------
# Metric calculations

# Closeness centrality [1] of a node u is the reciprocal of the sum
# of the shortest path distances from u to all n-1 other nodes.
# Since the sum of distances depends on the number of nodes in the graph,
# closeness is normalized by the sum of minimum possible distances n-1.
def spatial_distance(h, destin):
    return nx.single_source_dijkstra_path_length(h.SG[destin].reverse(copy=True),source=destin,weight='distance')

def topological_distance(h, destin):
    return nx.single_source_dijkstra_path_length(h.SG[destin].reverse(copy=True),source=destin)

def closeness_centrality(h, destin):
    return nx.closeness_centrality(h.SG[destin],distance='distance')

# Calculate the degree centrality of highway
def degree_centrality(h, destin):
    return nx.degree_centrality(h.SG[destin])

# Calculate the in degree centrality of highway
def in_degree_centrality(h, destin):
    return nx.in_degree_centrality(h.SG[destin])

# Calculate the out degree centrality of highway
def out_degree_centrality(h, destin):
    return nx.out_degree_centrality(h.SG[destin])

# The load centrality of a node is the fraction of all shortest paths that pass through that node.
def load_centrality(h, destin):
    return nx.load_centrality(h.SG[destin],weight='distance')

# Betweenness centrality of a node v is the sum of the fraction of all-pairs shortest paths that pass through v
def betweenness_centrality(h, destin):
    return nx.betweenness_centrality(h.SG[destin],normalized=True,weight='distance')

# Betweenness centrality of a node v is the sum of the fraction of all-pairs shortest paths that pass through v
def betweenness_centrality_exit(h, destin):
    return nx.betweenness_centrality_subset(h.SG[destin],normalized=True,sources=h.SG[destin].nodes(),targets=[destin],weight='distance')

# Betweenness centrality of an edge e is the sum of the 
# fraction of all-pairs shortest paths that pass through e:
def edge_betweenness_centrality(h, destin):
    return nx.edge_betweenness_centrality(h.SG[destin],weight='distance')
    # JSON writer doesn't understand tuple as key so need to convert it to string
    # To convert back:
    #   from ast import literal_eval
    #   literal_eval('(1,2)')

# Betweenness centrality of a node v is the sum of the fraction of all-pairs shortest paths that pass through v
def edge_betweenness_centrality_exit(h, destin):
    return nx.edge_betweenness_centrality_subset(h.SG[destin],normalized=True,sources=h.SG[destin].nodes(),targets=[destin],weight='distance')

# Eigenvector centrality computes the centrality for a node based on the centrality of its neighbors.
def eigenvector_centrality(h, destin):
    try:
        return nx.eigenvector_centrality_numpy(h.SG[destin],weight='weight')
    except Exception as e:
        print 'Error',destin,':',e
        return {}

# Betweenness centrality of a node v is the sum of the fraction of all-pairs shortest paths that pass through v
def betweenness_centrality(h, destin):
    return nx.betweenness_centrality(h.SG[destin],weight='distance')

# ---------------------------------------------------------
# Yet to be done

# Transitivity is the fraction of all possible triangles present in G.
def square_clustering(h, destin):
    return nx.square_clustering(h.SG[destin])

# Worth trying
# - Stream ordering - Horton and Strahler: no preavailable algorithms

# The following did not work for directed graph
# - triangles(G, nodes=None)
# - clustering(G, nodes=None, weight=None)
# - square_clustering(G, nodes=None)

# The following refused to work easily
# - closeness vitality just took too long
# - eccentricity(G[, v, sp])    Return the eccentricity of nodes in G.
# -- center(G[, e])    Return the center of the graph G.
# -- diameter(G[, e])    Return the diameter of the graph G.
# -- periphery(G[, e])    Return the periphery of the graph G.
# -- radius(G[, e])    Return the radius of the graph G.

# Calculate number of nodes on a highway
def summary(h, destin):
    r = OrderedDict()
    r['num_edges'] = h.SG[destin].number_of_edges()    
    r['num_nodes'] = h.SG[destin].number_of_nodes()

    # Spatial distance shortest path
    sdsp = h.results[destin]['spatial_distance']

    # Shortest path from all the nodes in the catchment area to the exit
    # min,max,mean shortest path length
    r['min_sdsp'] = np.min(sdsp.values())
    r['max_sdsp'] = np.max(sdsp.values())
    r['mean_sdsp'] = np.mean(sdsp.values())
    # 10%,50% and 90% shortest path lengths
    r['10pc_sdsp'] = np.percentile(sdsp.values(),10)
    r['50pc_sdsp'] = np.percentile(sdsp.values(),50)
    r['90pc_sdsp'] = np.percentile(sdsp.values(),90)

    nodes_mean_sdsp = [key for key,value in sdsp.iteritems() if value <= r['mean_sdsp']]
    nodes_10pc_sdsp = [key for key,value in sdsp.iteritems() if value <= r['10pc_sdsp']]
    nodes_50pc_sdsp = [key for key,value in sdsp.iteritems() if value <= r['50pc_sdsp']]
    nodes_90pc_sdsp = [key for key,value in sdsp.iteritems() if value <= r['90pc_sdsp']]

    r['num_nodes_mean_sdsp'] = len(nodes_mean_sdsp)
    r['num_nodes_10pc_sdsp'] = len(nodes_10pc_sdsp)
    r['num_nodes_50pc_sdsp'] = len(nodes_50pc_sdsp)
    r['num_nodes_90pc_sdsp'] = len(nodes_90pc_sdsp)

    r['frac_nodes_mean_sdsp'] = len(nodes_mean_sdsp)/float(r['num_nodes'])
    r['frac_nodes_10pc_sdsp'] = len(nodes_10pc_sdsp)/float(r['num_nodes'])
    r['frac_nodes_50pc_sdsp'] = len(nodes_50pc_sdsp)/float(r['num_nodes'])
    r['frac_nodes_90pc_sdsp'] = len(nodes_90pc_sdsp)/float(r['num_nodes'])

    # number of nodes less than 1000,500,250,125 metres away
    # could be a good indicator of how likely the bottlenecks are
    nodes_2000m = [key for key,value in sdsp.iteritems() if value <= 2000]
    nodes_1000m = [key for key,value in sdsp.iteritems() if value <= 1000]
    nodes_500m = [key for key,value in sdsp.iteritems() if value <= 500]
    nodes_250m = [key for key,value in sdsp.iteritems() if value <= 250]

    r['num_nodes_2000m'] = len(nodes_2000m)
    r['num_nodes_1000m'] = len(nodes_1000m)
    r['num_nodes_500m'] = len(nodes_500m)
    r['num_nodes_250m'] = len(nodes_250m)

    r['frac_nodes_2000m'] = len(nodes_2000m)/float(r['num_nodes'])
    r['frac_nodes_1000m'] = len(nodes_1000m)/float(r['num_nodes'])
    r['frac_nodes_500m'] = len(nodes_500m)/float(r['num_nodes'])
    r['frac_nodes_250m'] = len(nodes_250m)/float(r['num_nodes'])

    # Topological distance
    tdsp = h.results[destin]['topological_distance']

    # Shortest path from all the nodes in the catchment area to the exit
    # min,max,mean shortest path length
    r['min_tdsp'] = np.min(tdsp.values())
    r['max_tdsp'] = np.max(tdsp.values())
    r['mean_tdsp'] = np.mean(tdsp.values())
    # 10%,50% and 90% shortest path topological lengths
    r['10pc_tdsp'] = np.percentile(tdsp.values(),10)
    r['50pc_tdsp'] = np.percentile(tdsp.values(),50)
    r['90pc_tdsp'] = np.percentile(tdsp.values(),90)

    nodes_mean_tdsp = [key for key,value in tdsp.iteritems() if value <= r['mean_tdsp']]
    nodes_10pc_tdsp = [key for key,value in tdsp.iteritems() if value <= r['10pc_tdsp']]
    nodes_50pc_tdsp = [key for key,value in tdsp.iteritems() if value <= r['50pc_tdsp']]
    nodes_90pc_tdsp = [key for key,value in tdsp.iteritems() if value <= r['90pc_tdsp']]

    r['num_nodes_mean_tdsp'] = len(nodes_mean_tdsp)
    r['num_nodes_10pc_tdsp'] = len(nodes_10pc_tdsp)
    r['num_nodes_50pc_tdsp'] = len(nodes_50pc_tdsp)
    r['num_nodes_90pc_tdsp'] = len(nodes_90pc_tdsp)

    r['frac_nodes_mean_tdsp'] = len(nodes_mean_tdsp)/float(r['num_nodes'])
    r['frac_nodes_10pc_tdsp'] = len(nodes_10pc_tdsp)/float(r['num_nodes'])
    r['frac_nodes_50pc_tdsp'] = len(nodes_50pc_tdsp)/float(r['num_nodes'])
    r['frac_nodes_90pc_tdsp'] = len(nodes_90pc_tdsp)/float(r['num_nodes'])

    # number of nodes in this topological distance from the exit
    nodes_8_tdsp = [key for key,value in tdsp.iteritems() if value <= 8]
    nodes_4_tdsp = [key for key,value in tdsp.iteritems() if value <= 4]
    nodes_2_tdsp = [key for key,value in tdsp.iteritems() if value <= 2]
    nodes_1_tdsp = [key for key,value in tdsp.iteritems() if value <= 1]

    r['num_nodes_8_tdsp'] = len(nodes_8_tdsp)
    r['num_nodes_4_tdsp'] = len(nodes_4_tdsp)
    r['num_nodes_2_tdsp'] = len(nodes_2_tdsp)
    r['num_nodes_1_tdsp'] = len(nodes_1_tdsp)

    r['frac_nodes_8_tdsp'] = len(nodes_8_tdsp)/float(r['num_nodes'])
    r['frac_nodes_4_tdsp'] = len(nodes_4_tdsp)/float(r['num_nodes'])
    r['frac_nodes_2_tdsp'] = len(nodes_2_tdsp)/float(r['num_nodes'])
    r['frac_nodes_1_tdsp'] = len(nodes_1_tdsp)/float(r['num_nodes'])

    # Total length and area
    r['sum_edge_length'] = h.SG[destin].size(weight='distance')
    r['sum_edge_area'] = h.SG[destin].size(weight='area')

    # Average edge length,width,area
    r['mean_edge_length'] = r['sum_edge_length']/r['num_edges']
    r['mean_edge_area'] = r['sum_edge_area']/r['num_edges']
    r['mean_edge_width'] = r['sum_edge_area']/r['sum_edge_length']

    # Components
    # ----------

    r['num_strgconcom'] = nx.number_strongly_connected_components(h.SG[destin])
    r['num_weakconcom'] = nx.number_weakly_connected_components(h.SG[destin])
    r['num_attractcom'] = nx.number_attracting_components(h.SG[destin])

    # Degree
    # ------

    r['mean_degree'] = 2*r['num_edges']/float(r['num_nodes'])

    # Number of nodes with no incoming edges - based on leaf node
    r['num_nodes_0_in_deg'] = len([n for n,d in h.SG[destin].in_degree().items() if d==0])
    r['num_nodes_1_in_deg'] = len([n for n,d in h.SG[destin].in_degree().items() if d==1])
    r['num_nodes_2_in_deg'] = len([n for n,d in h.SG[destin].in_degree().items() if d==2])
    r['num_nodes_3+_in_deg'] = len([n for n,d in h.SG[destin].in_degree().items() if d>=3])
    r['num_nodes_0_out_deg'] = len([n for n,d in h.SG[destin].out_degree().items() if d==0])
    r['num_nodes_1_out_deg'] = len([n for n,d in h.SG[destin].out_degree().items() if d==1])
    r['num_nodes_2_out_deg'] = len([n for n,d in h.SG[destin].out_degree().items() if d==2])
    r['num_nodes_3+_out_deg'] = len([n for n,d in h.SG[destin].out_degree().items() if d>=3])

    r['frac_nodes_0_in_deg'] = r['num_nodes_0_in_deg']/float(r['num_nodes'])
    r['frac_nodes_1_in_deg'] = r['num_nodes_1_in_deg']/float(r['num_nodes'])
    r['frac_nodes_2_in_deg'] = r['num_nodes_2_in_deg']/float(r['num_nodes'])
    r['frac_nodes_3+_in_deg'] = r['num_nodes_3+_in_deg']/float(r['num_nodes'])
    r['frac_nodes_0_out_deg'] = r['num_nodes_0_out_deg']/float(r['num_nodes'])
    r['frac_nodes_1_out_deg'] = r['num_nodes_1_out_deg']/float(r['num_nodes'])
    r['frac_nodes_2_out_deg'] = r['num_nodes_2_out_deg']/float(r['num_nodes'])
    r['frac_nodes_3+_out_deg'] = r['num_nodes_3+_out_deg']/float(r['num_nodes'])

    # Centralities
    # ------------


    def _means(key,all):
        return [
                ('mean_{}'.format(key),np.mean(all.values())),
                ('mean_{}_mean_sdsp'.format(key),np.mean([v for k,v in all.iteritems() if k in nodes_mean_sdsp])),
                ('mean_{}_10pc_sdsp'.format(key),np.mean([v for k,v in all.iteritems() if k in nodes_10pc_sdsp])),
                ('mean_{}_50pc_sdsp'.format(key),np.mean([v for k,v in all.iteritems() if k in nodes_50pc_sdsp])),
                ('mean_{}_90pc_sdsp'.format(key),np.mean([v for k,v in all.iteritems() if k in nodes_90pc_sdsp])),
                ('mean_{}_2000m'.format(key),np.mean([v for k,v in all.iteritems() if k in nodes_2000m])),
                ('mean_{}_1000m'.format(key),np.mean([v for k,v in all.iteritems() if k in nodes_1000m])),
                ('mean_{}_500m'.format(key),np.mean([v for k,v in all.iteritems() if k in nodes_500m])),
                ('mean_{}_250m'.format(key),np.mean([v for k,v in all.iteritems() if k in nodes_250m])),
                ('mean_{}_mean_tdsp'.format(key),np.mean([v for k,v in all.iteritems() if k in nodes_mean_tdsp])),
                ('mean_{}_10pc_tdsp'.format(key),np.mean([v for k,v in all.iteritems() if k in nodes_10pc_tdsp])),
                ('mean_{}_50pc_tdsp'.format(key),np.mean([v for k,v in all.iteritems() if k in nodes_50pc_tdsp])),
                ('mean_{}_90pc_tdsp'.format(key),np.mean([v for k,v in all.iteritems() if k in nodes_90pc_tdsp])),
                ('mean_{}_8_tdsp'.format(key),np.mean([v for k,v in all.iteritems() if k in nodes_8_tdsp])),
                ('mean_{}_4_tdsp'.format(key),np.mean([v for k,v in all.iteritems() if k in nodes_4_tdsp])),
                ('mean_{}_2_tdsp'.format(key),np.mean([v for k,v in all.iteritems() if k in nodes_2_tdsp])),
                ('mean_{}_1_tdsp'.format(key),np.mean([v for k,v in all.iteritems() if k in nodes_1_tdsp])),
            ]

    # Degree Centrality
    r.update(_means('deg_cen',h.results[destin]['degree_centrality']))
    
    # In degree centrality
    r.update(_means('in_deg_cen',h.results[destin]['in_degree_centrality']))

    # Out degree centrality
    r.update(_means('out_deg_cen',h.results[destin]['out_degree_centrality']))

    # Betweenness centrality
    r.update(_means('bet_cen',h.results[destin]['betweenness_centrality']))
    
    # Betweenness centrality subset for exit
    r.update(_means('bet_cen_exit',h.results[destin]['betweenness_centrality_exit']))

    # Edge betweenness centrality
    r['mean_edge_bet_cen'] = np.mean(h.results[destin]['edge_betweenness_centrality'].values())
    
    # Edge betweenness centrality subset for exit
    r['mean_edge_bet_cen_exit'] = np.mean(h.results[destin]['edge_betweenness_centrality_exit'].values())

    # Load centrality
    r.update(_means('load_cen_exit',h.results[destin]['load_centrality']))

    # Eigenvector centrality
    r.update(_means('eivec_cen_exit',h.results[destin]['eigenvector_centrality']))

    # Closeness centrality
    r.update(_means('close_cen_exit',h.results[destin]['closeness_centrality']))

    # Density - 0 for no edges, 1 for fully connected
    r['density'] = nx.density(h.SG[destin])

    # Clustering
    # ----------
    # Transitivity is the fraction of all possible triangles present in G.
    r['transitivity'] = nx.transitivity(h.SG[destin])
    
    # Mean square_clustering
    r.update(_means('sq_clust',h.results[destin]['square_clustering']))
    
    return r

# Drawing
# -------
'''
m = h.results[destin]['betweenness_centrality']
SG = h.SG[destin]
nodelist = [int(v) for v in m.keys()]
values = m.values()
nodes = nx.draw_networkx_nodes(SG,pos=h.nodes,nodelist=nodelist,node_color=values,node_size=5,linewidths=0)
edges = nx.draw_networkx_edges(SG,pos=h.nodes,nodelist=nodelist,node_color=values,linewidths=0,arrows=False)
plt.colorbar(nodes)
plt.draw()
'''

# ---------------------------------------------------------
# Configuration

sim = 'bristol25'
places = abm.Places('bristol25').names
folder_structure = 'analysis/{}/metrics/{}/{}'
file_structure = '{}/{}.json.gz'

from collections import OrderedDict

# For the ubuntu machine
# if socket.gethostname() == 'IT050339':
metric_functions = OrderedDict({
    'degree_centrality': degree_centrality,
    'in_degree_centrality': in_degree_centrality,
    'out_degree_centrality': out_degree_centrality,
    'betweenness_centrality': betweenness_centrality,
    'betweenness_centrality_exit': betweenness_centrality_exit,
    'edge_betweenness_centrality': edge_betweenness_centrality,
    'edge_betweenness_centrality_exit': edge_betweenness_centrality_exit,
    'load_centrality': load_centrality,
    'eigenvector_centrality': eigenvector_centrality,
    'closeness_centrality': closeness_centrality,
    'square_clustering': square_clustering,
    'spatial_distance': spatial_distance,    
    'topological_distance': topological_distance,
})

metric_functions.update({'summary': summary})

# ---------------------------------------------------------

def load_json(metric,place,destin):
    print metric
    folder = folder_structure.format(sim, metric, place)
    print folder
    fname = file_structure.format(folder, destin)
    with gzip.open(fname, 'r') as infile:
        intermediate = json.load(infile)
        # Convert key to number or tuple before returning
        if metric == 'summary':
            return intermediate
        else:
            return {eval(key):value for key,value in intermediate.iteritems()}

def stringify_key(dictionary):
    # Convert key of a dictionary to string
    return OrderedDict([(str(key),value) for key,value in dictionary.iteritems()])

# ---------------------------------------------------------
# Job to make parallel

def job(place):
    # Run the following to debug multiprocessing pool
    # for a single place with the place uncommented
    # place = 'Solihull'
    h = db.Highway(place)
    h.init_destins()
    h.init_route()

    # Precalculate width and area to ease the process
    for u,v,d in h.G.edges(data=True):
        # assumed width
        h.G[u][v]['awidth'] = h.width[h.hiclass[d['highway']]]
        # edge area
        h.G[u][v]['area'] = d['distance'] * h.G[u][v]['awidth']

    # Load subgraph
    h.init_SG(subgraph='nearest')
    
    status = {}
    h.results = {}
    for destin in h.destins: 
        # Initialise results to use to generate summary
        h.results[destin] = {}
        for metric in metric_functions:
            try:
                status[metric]
            except KeyError:                
                status[metric] = []
            print 'Starting',place, metric, destin
            start_time = time.time()
            folder = folder_structure.format(sim, metric, place)
            fname = file_structure.format(folder, destin)
            if not os.path.isfile(fname) or metric == 'summary':
                if not os.path.isdir(folder):
                    os.makedirs(folder)
                h.results[destin][metric] = metric_functions[metric](h, destin)
                with gzip.open(fname, 'w') as outfile:
                    json.dump(stringify_key(h.results[destin][metric]), outfile, indent=True)
                status[metric].append((destin,'processed'))
            else:
                h.results[destin][metric] = load_json(metric,place,destin)
                status[metric].append((destin,'loaded from file'))
            elapsed_time = time.time() - start_time  
            print 'Finished',place, metric, destin, int(elapsed_time),'seconds elapsed'

    for metric in metric_functions:
        for status_row in status[metric]:
            print place,metric,status_row

# ---------------------------------------------------------
# Parallelisation starts here

if __name__ == '__main__':
    pool = Pool(processes=cpu_count()) 
    pool.map(job, places)    