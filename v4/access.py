import sys
sys.path.append('core')

import db
reload(db)

import gzip

# For the non-flooded case:
# - Remove all small roads from normal.G
# - Loop n times
#   - Pick a random location
#   - Determine the time it takes to reach hospital from that location
#   - Look at the distribution of time across n

# Initialise the normal highway
# For the actual thing, use .G. and use fresh=True
normal = db.Highway('Carlisle',graph='full')

# Remove all small roads, i.e. hiclass > 5
small_roads=[(u,v) for u,v,d in normal.G.edges(data=True) if d['hiclass']>5]
print normal.G.number_of_edges(),normal.G.number_of_nodes()
normal.G.remove_edges_from(small_roads)
print normal.G.number_of_edges(),normal.G.number_of_nodes()

# Determine the location of hospital and route to the hospital
normal.fresh = True
hospital = normal.nearest_node(-2.955547,54.8955621)
normal.destins = [hospital]

# Display the location of the hospital on the map
# db.plt.ion()
# db.plt.figure()
# normal.fig_destins()
# db.plt.xlabel('lon')
# db.plt.ylabel('lat')

# Initialise the route for the normal case
normal.init_route()

# For the flooded case:
# - Remove all small roads from flooded.G
# - Remove all blocked roads from flooded.G
# - Loop n times
#   - Pick a random location
#   - Determine the time it takes to reach hospital from that location
#   - Look at the distribution of time across n

# Links to retain despite flooding because we know they wont get flooded
# override_osm_id = []
# override_osm_id.append(70899124L) # Nelson road bridge

# Make the list of flooded roads
from zipfile import ZipFile
zip_file = 'flood/Archive.zip'
z = ZipFile(zip_file)
flood_files = z.namelist()
flood_files.reverse()

# Get bounding box of the flood map[0] and create a rectangle
from shapely.geometry.geo import box
flood = db.Flood('/vsizip/{}/{}'.format(zip_file,flood_files[0]))
bbox = box(flood.minx,flood.miny,flood.maxx,flood.maxy)

# Get a list of all the nodes that intersect with the rectangle
from shapely.geometry import Point
nodes_in_bbox = [k for k,v in normal.G.nodes(data=True) if bbox.intersects(Point(v))]

# Output a file with list of nodes in the bbox with their latlon
with open('flood/nodes_in_bbox.csv','w') as file:
    to_write = ''
    for k in nodes_in_bbox:
        to_write += '{},{},{}\n'.format(k,normal.G.node[k][0],normal.G.node[k][1])
    file.write(to_write)

# Only do a selection
# flood_files = []
# for i in range(10,101,10):
#     for j in range(10,201,10):
#         flood_files.append('res{}_{}_binary.tif'.format(i,j))

# Import CSV file
import csv
import copy

# Deep copy the normal highway
# flooded = copy.deepcopy(normal)

def access(case):
    # If the results already exist then read the file and skip this iteration
    csv_file = 'flood/results/{}.csv.gz'.format(case)
    try:
        with gzip.open(csv_file,'r') as file:
            diff=dict([(int(k),float(v)) for k,v in csv.reader(file)])
    except IOError:
        # Load the raster file for the flood case
        flood = db.Flood('/vsizip/{}/{}'.format(zip_file,case))

        # Deep copy the normal highway
        flooded = copy.deepcopy(normal)
        
        # Now remove all the flooded edges and find route to the hospital after considering flooded roads
        # flooded_roads=[(u,v) for u,v,d in flooded.G.edges_iter(data=True) if flood.isFlooded(flooded.G.node[u],flooded.G.node[v]) is True and d['osm_id'] not in override_osm_id]
        flooded_roads=[(u,v,d) for u,v,d in flooded.G.edges_iter(data=True) if flood.isFlooded(flooded.G.node[u],flooded.G.node[v]) is True]
        before = flooded.G.number_of_edges(),flooded.G.number_of_nodes()
        flooded.G.remove_edges_from(flooded_roads)
        after = flooded.G.number_of_edges(),flooded.G.number_of_nodes()
        print case, before, after

        # Show the flooded map
        # db.plt.figure()
        # flooded.fig_destins()
        # db.plt.show()

        # Initialise route for the flooded case
        # Do not save as the route is only valid for this configuration of flood
        flooded.init_route(save=False)

        # Re-add the edges
        # flooded.G.add_edges_from(flooded_roads)

        # Plot figures of distances to exit in flooded and non flooded case
        # db.plt.figure()
        # db.plt.hist(normal.route_length[hospital].values(),bins=50,histtype='step')
        # db.plt.hist(flooded.route_length[hospital].values(),bins=50,histtype='step')
        # db.plt.xlabel('distance (m)')
        # db.plt.ylabel('number of nodes')

        diff = {}
        # Iterate through the list of nodes within the boundary
        #   assuming that the hospital can be accessed through roads
        #   outside of the flood area
        for k in nodes_in_bbox:
            try:
                diff[k] = flooded.route_length[hospital][k] - normal.route_length[hospital][k]
            except KeyError:
                diff[k] = -99.99 # Arbitrary value so that we can clearly see the inaccessible nodes in the histogram

        # Plot figures of distances to hospitals from nodes on a map            
        # db.plt.figure()
        # flooded.fig_highway()
        # nx.draw_networkx_nodes(flooded.F, flooded.G.node, nodelist=diff[case].keys(), node_color=diff[case].values(), alpha=0.5,node_shape='.',node_size=40,linewidths=0)
        # db.plt.xlabel('lon')
        # db.plt.ylabel('lat')

        with gzip.open(csv_file,'w') as file:
            for k in nodes_in_bbox:
                file.write('{},{}\n'.format(k,diff[k]))
    # Either way, return diff as the output
    return diff

# Plot figures of difference histograms side by side
# db.plt.figure()
# for case in flood_files:
#     db.plt.hist(diff[case].values(),bins=50,histtype='step',cumulative=True,label=case)
# db.plt.xlabel('difference (m)')
# db.plt.ylabel('number of nodes')
# db.plt.legend(loc=3)

# Meeting
# Distance
# - Look at the change in distances for each nodes
# - Colour the nodes according to distance to the hospital
#   - lon, lat, distance data
# Flood distance
#   - Plot lon, lat, difference and send this data
# Plot the difference from no flood --> lots of flood
# Population weighted nodes

import os
import time
import logging

log_file = 'logs/{0}.log'.format(time.ctime())

# get logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# create a file handler
handler = logging.FileHandler(log_file)
handler.setLevel(logging.INFO)

# create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(handler)

# Save each exit to a different file
def job(case):
    '''
    Multiprocessing runs this subroutine for each place in parallel

    Parameters
    ----------

    case : string
        A string that describes case.

    Returns
    -------

    The number of successfully completed scenarios in this run.
    '''

    try:
        return access(case)
    except Exception as e:
        logger.critical(case)
        logger.exception(e)
        print 'Exception caught in run.py'
        print e
        return False

import multiprocessing
import signal

def init_process():
    '''
    Multiprocessing calls this before starting.
    '''
    signal.signal(signal.SIGINT, signal.SIG_IGN)    
    print 'Starting', multiprocessing.current_process().name    

# Start the multiprocessing unit
if __name__ == '__main__':
    pool_size = multiprocessing.cpu_count()

    try:
        pool = multiprocessing.Pool(processes=pool_size,
                                    initializer=init_process,
                                    )
        pool_outputs = pool.map(job, flood_files)
        pool.close() # no more tasks
        pool.join()  # wrap up current task
        print 'Pool closed and joined normally.'
    except KeyboardInterrupt:
        print 'KeyboardInterrupt caught in parent...'
        pool.terminate()
        pool.join()  # wrap up current task
        print 'Pool terminated and joined due to an exception'

print '----------------------------------------------------------------------'
print 'Summary'
print '----------------------------------------------------------------------'
complete = 0
incomplete = 0
for case, diff in zip(flood_files, pool_outputs):
    if type(diff) == bool:
        print 'INCOMPLETE CASE: {}'.format(case)
        incomplete += 1
    elif type(diff) == dict:
        complete += 1
print 'INCOMPLETE CASES: {}'.format(incomplete)
print 'COMPLETE CASES: {}'.format(complete)

# Create a dict of values corresponding to each
f = dict(zip(flood_files,pool_outputs))
all_nodes = {}
for i in range(1,101,1):
    for j in range(1,201,1):
        for n in nodes_in_bbox:
            v = f['res{}_{}_binary.tif'.format(i,j)][n]
            try:
                all_nodes[n].append(v)
            except KeyError:
                all_nodes[n] = [v]

# Save the outputs to a file
with gzip.open('flood/all_nodes.csv.gz', 'wb') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    for k in all_nodes:
        writer.writerow([k]+all_nodes[k])