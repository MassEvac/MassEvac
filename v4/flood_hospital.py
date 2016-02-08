import sys,os,gzip
sys.path.append('core')
import db
reload(db)

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
override_osm_id = []
override_osm_id.append(42049842) # cycleway
override_osm_id.append(4785033) # primary
override_osm_id.append(42467180) # primary
override_osm_id.append(4260330) # primary
override_osm_id.append(223872960) # cycleway
override_osm_id.append(112145858) # primary
override_osm_id.append(206315230) # primary
override_osm_id.append(206315241) # cycleway
override_osm_id.append(70899124) # Nelson road bridge

# Make the list of flooded roads
from zipfile import ZipFile
input_file = '30cm'
zip_file = 'flood/resni_nj_binary_{}.zip'.format(input_file)
z = ZipFile(zip_file)
flood_files = z.namelist()
flood_files.reverse()

# Make folder for results
output_folder = 'flood/results/{}'.format(input_file)
if not os.path.isdir(output_folder):
    os.makedirs(output_folder)

# Get bounding box of the flood map[0] and create a rectangle
from shapely.geometry.geo import box
flood = db.Flood('/vsizip/{}/{}'.format(zip_file,flood_files[0]))
bbox = box(flood.minx,flood.miny,flood.maxx,flood.maxy)

# See if the node is reachable or not
def reachable(node):
    try:
        normal.route[hospital][node]
        return True
    except KeyError:
        return False

# Get a list of all the nodes that intersect with the rectangle
from shapely.geometry import Point
nodes_in_bbox = [k for k,v in normal.G.nodes(data=True) if bbox.intersects(Point(v)) and reachable(k)]

# Initialise node population to 0
all_node_pop = {}
for n in normal.G.nodes():
    all_node_pop[n] = 0

# From each edge, distribute its population to connecting nodes.
for u,v,d in normal.G.edges(data=True):
    all_node_pop[u] += d['pop_dist']/2
    all_node_pop[v] += d['pop_dist']/2

# Output a file with list of nodes in the bbox with their lon/lat/dist/time
# https://www.gov.uk/government/news/speed-limit-exemption-for-life-saving-services
ambulance_speed = 30 # km/h  to calculate ambulance time to hospital
dist_hospital = {}
with open('flood/nodes_in_bbox.csv','w') as file:
    fieldnames = ['node_id','longitude', 'latitude','dist_hospital','time_hospital','pop_dist']
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    for k in nodes_in_bbox:
        d = dist_hospital[k] = normal.route_length[hospital][k] # in metres
        t = d/ambulance_speed/1000*60 # in mins
        writer.writerow({'node_id':k,'longitude':normal.G.node[k][0],'latitude':normal.G.node[k][1],'dist_hospital':d,'time_hospital':t,'pop_dist':all_node_pop[k]})

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
    csv_file = '{}/{}.csv.gz'.format(output_folder,case)
    try:
        with gzip.open(csv_file,'r') as file:
            diff=dict([(int(k),float(v)) for k,v in csv.reader(file)])
    except IOError:
        # Load the raster file for the flood case
        flood = db.Flood('/vsizip/{}/{}'.format(zip_file,case))

        # Deep copy the normal highway
        flooded = copy.deepcopy(normal)
        
        # Now remove all the flooded edges and find route to the hospital after considering flooded roads
        flooded_roads=[(u,v) for u,v,d in flooded.G.edges_iter(data=True) if flood.isFlooded(flooded.G.node[u],flooded.G.node[v]) is True and d['osm_id'] not in override_osm_id]
        # flooded_roads=[(u,v) for u,v,d in flooded.G.edges_iter(data=True) if flood.isFlooded(flooded.G.node[u],flooded.G.node[v]) is True]
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
                diff[k] = -99 # Arbitrary value so that we can clearly see the inaccessible nodes in the histogram

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
    for j in range(1,101,1):
        for n in nodes_in_bbox:
            v = f['res{}_{}_binary_30cm.tif'.format(i,j)][n]
            try:
                all_nodes[n].append(v)
            except KeyError:
                all_nodes[n] = [v]

# Save the outputs to a file
with gzip.open('flood/all_nodes_{}.csv.gz'.format(input_file), 'wb') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    for k in nodes_in_bbox:
        writer.writerow([k]+all_nodes[k])

# Load output areas
from shapely.geometry import shape
with fiona.open('flood/OA_unclipped.shp') as source:
    OA_shapes = [shape(OA['geometry']) for OA in source]

# Determine OA_id for each nodes_in_bbox
OA_nodes_in_bbox = {}
for k in nodes_in_bbox:
    node = Point(normal.G.node[k])
    for OA_id,OA_shape in enumerate(OA_shapes):
        if OA_shape.intersects(node):
            try:
                OA_nodes_in_bbox[OA_id].append(k)
            except KeyError:
                OA_nodes_in_bbox[OA_id] = [k]


# Now for each OA_id, determine what proportion of nodes
# are accessible per scenario within 8 and 30 mins
OA_accessible_08min = {}
OA_accessible_30min = {}
ambulance_speed = 30 # km/h  to calculate ambulance time to hospital
for scenario in range(10000):
    print scenario
    for OA_id in OA_nodes_in_bbox:
        flooded_dist_hospital = np.array([all_nodes[k][scenario]+dist_hospital[k] for k in OA_nodes_in_bbox[OA_id]])/ambulance_speed/1000*60
        access_08min = sum(flooded_dist_hospital < 8)/float(len(flooded_dist_hospital))
        access_30min = sum(flooded_dist_hospital < 30)/float(len(flooded_dist_hospital))
        try:
            OA_accessible_08min[OA_id].append(access_08min)
            OA_accessible_30min[OA_id].append(access_30min)
        except KeyError:
            OA_accessible_08min[OA_id] = [access_08min]
            OA_accessible_30min[OA_id] = [access_30min]

OA_ref = ["E00097300","E00097298","E00097297","E00097274","E00097273","E00097272","E00097271","E00097270","E00097269","E00097268","E00097267","E00097266","E00097265","E00097264","E00097263","E00097262","E00097261","E00097260","E00097259","E00097258","E00097257","E00097256","E00097252","E00097250","E00097243","E00097242","E00097241","E00097240","E00097239","E00097238","E00097237","E00097236","E00097235","E00097234","E00097233","E00097232","E00097231","E00097230","E00097229","E00097228","E00097227","E00097226","E00097225","E00097224","E00097223","E00097222","E00097212","E00097211","E00097209","E00097208","E00097207","E00097206","E00097205","E00097204","E00097203","E00097198","E00097164","E00097163","E00097162","E00097161","E00097150","E00097136","E00097134","E00097133","E00097132","E00097131","E00097130","E00097129","E00097128","E00097127","E00097126","E00097125","E00097124","E00097123","E00097122","E00097121","E00097120","E00097119","E00097118","E00097117","E00097116","E00097115","E00097096","E00097095","E00097094","E00097093","E00097079","E00097078","E00097075","E00097074","E00097073","E00097072","E00097071","E00097070","E00097069","E00097068","E00097067","E00097066","E00097065","E00097064","E00097063","E00097062","E00097061","E00097060","E00097059","E00097058","E00097057","E00097056","E00097055","E00097034","E00097033","E00097032","E00097031","E00097030","E00097029","E00097028","E00097027","E00097026","E00097025","E00097024","E00097023","E00097022","E00097021","E00097020","E00097019","E00097018","E00097017","E00097016","E00097015","E00097014","E00097011","E00097007","E00097000","E00096988","E00096987","E00096986","E00096985","E00096984","E00096983","E00096981","E00096975"]

# Save the outputs to a file
with gzip.open('flood/OA_accessible_08min_{}.csv.gz'.format(input_file), 'wb') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    for OA_id in OA_accessible_08min:
        writer.writerow([OA_id]+OA_accessible_08min[OA_id])

with gzip.open('flood/OA_accessible_30min_{}.csv.gz'.format(input_file), 'wb') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    for OA_id in OA_accessible_30min:
        writer.writerow([OA_id]+OA_accessible_30min[OA_id])

# Write min and max to file
min_dist = {}
max_dist = {}
max_accessible_dist = {}
for k in nodes_in_bbox:
    a = np.array(all_nodes[k])
    max_accessible_dist[k] = a.max()
    np.place(a,a<0,np.inf)
    max_dist[k] = a.max()   
    min_dist[k] = a.min()
    if min_dist[k] == np.inf:
        min_dist[k] = -99
    if max_dist[k] == np.inf:
        max_dist[k] = -99       

with open('flood/nodes_stats_{}.csv'.format(input_file),'w') as file:
    fieldnames = ['node_id','dist_min', 'dist_max','dist_max_accessible']
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    for k in nodes_in_bbox:
        d = normal.route_length[hospital][k] # in metres
        writer.writerow({'node_id':k,'dist_min':min_dist[k],'dist_max':max_dist[k],'dist_max_accessible':max_accessible_dist[k]})

# Output the results to aggregate CSV files
for i in range(1,101,1):
    with gzip.open('flood/results_agg/{}/{}.csv.gz'.format(input_file,i),'w') as file:
        fieldnames = ['node_id']
        for j in range(1,101,1):
            fieldnames.append('{}_{}'.format(i,j))
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for k in nodes_in_bbox:
            row = {'node_id':k}
            for j in range(1,101,1):
                row['{}_{}'.format(i,j)] = f['res{}_{}_binary_30cm.tif'.format(i,j)][k]
            writer.writerow(row)


fieldnames = ['node_id']
for i in range(1,101,1):
    for j in range(1,101,1):
        fieldnames.append('{}_{}'.format(i,j))

with gzip.open('flood/results_{}.csv.gz'.format(input_file),'w') as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    for k in nodes_in_bbox:
        row = {'node_id':k}
        for i in range(1,101,1):
            for j in range(1,101,1):
                row['{}_{}'.format(i,j)] = f['res{}_{}_binary_30cm.tif'.format(i,j)][k]
        writer.writerow(row)