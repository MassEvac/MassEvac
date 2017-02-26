import sys
sys.path.append('core/')
import db
import os
import csv
import gzip
import copy
import time
import fiona
import pandas
import random
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from zipfile import ZipFile
from shapely.geometry import Point, shape, box

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
print 'no of edges & nodes',normal.G.number_of_edges(),normal.G.number_of_nodes()
normal.G.remove_edges_from(small_roads)
print 'after removing small roads'
print 'no of edges & nodes',normal.G.number_of_edges(),normal.G.number_of_nodes()

# Determine the location of hospital and route to the hospital
hospital = normal.nearest_node(-2.955547,54.8955621)
normal.destins = [hospital]

# Initialise the route for the normal case
normal.fresh = True # This needs to be true otherwise it will load it from existing file
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

#  ---------------
# Load foood files
# Make the list of flooded roads
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
nodes_in_bbox = [k for k,v in normal.G.nodes(data=True) if bbox.intersects(Point(v['pos'])) and reachable(k)]

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
        writer.writerow({'node_id':k,'longitude':normal.G.node[k]['pos'][0],'latitude':normal.G.node[k]['pos'][1],'dist_hospital':d,'time_hospital':t,'pop_dist':all_node_pop[k]})

# Meeting
# Distance
# - Look at the change in distances for each nodes
# - Colour the nodes according to distance to the hospital
#   - lon, lat, distance data
# Flood distance
#   - Plot lon, lat, difference and send this data
# Plot the difference from no flood --> lots of flood
# Population weighted nodes

"""Start describing multiprocessing job"""
def job(case):
    # If the results already exist then read the file and skip this iteration
    csv_file = '{}/{}.csv.gz'.format(output_folder,case)
    try:
        with gzip.open(csv_file,'r') as file:
            diff = dict([(int(k),float(v)) for k,v in csv.reader(file)])
    except IOError:
        # Load the raster file for the flood case
        flood = db.Flood('/vsizip/{}/{}'.format(zip_file,case))

        # Deep copy the normal highway
        flooded = copy.deepcopy(normal)
        
        # Now remove all the flooded edges and find route to the hospital after considering flooded roads
        flooded_roads=[(u,v) for u,v,d in flooded.G.edges_iter(data=True) if flood.isFlooded(flooded.G.node[u]['pos'],flooded.G.node[v]['pos']) is True and d['osm_id'] not in override_osm_id]
        # flooded_roads=[(u,v) for u,v,d in flooded.G.edges_iter(data=True) if flood.isFlooded(flooded.G.node[u]['pos'],flooded.G.node[v]['pos']) is True]
        before = flooded.G.number_of_edges(),flooded.G.number_of_nodes()
        flooded.G.remove_edges_from(flooded_roads)
        after = flooded.G.number_of_edges(),flooded.G.number_of_nodes()
        print case, before, after

        # Initialise route for the flooded case
        # Do not save as the route is only valid for this configuration of flood
        flooded.init_route(save=False)

        diff = {}
        # Iterate through the list of nodes within the boundary
        #   assuming that the hospital can be accessed through roads
        #   outside of the flood area
        for k in nodes_in_bbox:
            try:
                diff[k] = flooded.route_length[hospital][k] - normal.route_length[hospital][k]
            except KeyError:
                diff[k] = -99 # Arbitrary value so that we can clearly see the inaccessible nodes in the histogram

        with gzip.open(csv_file,'w') as file:
            for k in nodes_in_bbox:
                file.write('{},{}\n'.format(k,diff[k]))
    
    # Either way, return diff as the output
    return diff

def init_process():
    '''
    Multiprocessing calls this before starting.
    '''
    print 'Starting', multiprocessing.current_process().name    

"""Read all distances to hospital or compile them"""
diff2h_file = 'flood/diff2h.hdf'
try:
    # Read distance to hospital
    diff2h = pandas.read_hdf(diff2h_file,input_file)
except IOError:            
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

    print '----------------------------------------------------------'
    print 'Summary'
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
    print '----------------------------------------------------------'

    # Dict with -99 values
    all_nodes_99 = {}
    # Create a dict of values corresponding to each
    f = dict(zip(flood_files,pool_outputs))
    for i in range(1,101,1):
        for j in range(1,101,1):
            for n in nodes_in_bbox:
                v = f['res{}_{}_binary_{}.tif'.format(i,j,input_file)][n]
                try:
                    all_nodes_99[n].append(v)
                except KeyError:
                    all_nodes_99[n] = [v]

    diff2h = pandas.DataFrame(all_nodes_99)
    diff2h[diff2h<0] = np.nan
    diff2h.to_hdf(diff2h_file,input_file,mode='a',complib='blosc',fletcher32=True)


"""Load shape files"""
# Load output areas
with fiona.open('flood/OA_unclipped.shp') as source:
    OA_ref,OA_shapes = zip(*[(OA['properties']['OA11CD'],shape(OA['geometry'])) for OA in source])

# Determine OA_id for each nodes_in_bbox
OA_nodes = {}
for k in nodes_in_bbox:
    node = Point(normal.G.node[k]['pos'])
    for id,OA_shape in enumerate(OA_shapes):
        if OA_shape.intersects(node):
            try:
                OA_nodes[id].append(k)
            except KeyError:
                OA_nodes[id] = [k]

"""Now assess all the output areas"""
ambulance_speed = 30 # km/h  to calculate ambulance time to hospital
normal_dist2h = pandas.Series(dist_hospital)
normal_time2h = normal_dist2h/ambulance_speed/1000*60  # in mins
dist2h = diff2h + normal_dist2h
time2h = dist2h/ambulance_speed/1000*60 # in mins

# Now for each OA_id, determine what proportion of nodes
# are accessible per scenario within 8 and 30 mins

settings = {'08min':8,'30min':30}
access = {} # 1 means OA has 100% access. # 0 = OA has 0% access
baseline = {} # where there is no flood
for setting,threshold in settings.iteritems():
    access[setting] = pandas.DataFrame()
    baseline[setting] = pandas.Series()
    for id,nodes in OA_nodes.iteritems():
        baseline[setting].loc[id] = (normal_time2h.loc[nodes] < threshold).sum()/float(len(nodes))        
        access[setting][id] = (time2h[nodes] < threshold).sum(axis=1)/float(len(nodes))
    baseline[setting].to_csv('flood/baseline_{}.csv'.format(setting))
    access[setting].to_csv('flood/access_{}.csv'.format(setting))

""" Write min and max of diff and dist to file"""
stats = pandas.DataFrame()
stats['dist_min'] = dist2h.min()
stats['dist_max'] = dist2h.max()
stats['diff_min'] = diff2h.min()
stats['diff_max'] = diff2h.max()
stats.index = nodes_in_bbox
stats.index.name='node_id'
stats.fillna(-99).to_csv('flood/nodes_stats_{}.csv'.format(input_file))

# Quick overview stats
for setting in settings:
    target = access[setting] 

    # Mean and std of access score per output area
    target.mean().to_csv('flood/mean_target_{}.csv'.format(setting))
    target.std().to_csv('flood/std_target_{}.csv'.format(setting))

"""NOW INCORPORATING INFLOWS"""
inflows = pandas.read_csv('flood/inflows.csv')[['Eden','Petterill','Caldew']]

"""Separate training and testing set"""

try:
    train,test = pandas.DataFrame.from_csv('flood/classifier_indices.csv').transpose().values
except IOError:
    train = np.array(random.sample(range(10000),5000))
    test = np.array([i for i in range(10000) if i not in train])
    pandas.DataFrame({'train':train,'test':test}).to_csv('flood/classifier_indices.csv')

"""Inputs"""
xtrain = inflows.ix[train]
xtest = inflows.ix[test]

xtrain.to_csv('flood/train_inflows.csv')
xtest.to_csv('flood/test_inflows.csv')

"""
TRAIN THE CLASSIFIER
                    Train inflows                          
                            |
                            v
  Test inflows -------> Classifier <-- Train access
                            |
                            |           Test access
                            |               |
                            v               v
                    Predicted access---> Evaluation
"""

"""Try Gaussian"""
from sklearn.naive_bayes import GaussianNB,BernoulliNB,MultinomialNB
from sklearn import metrics
from sklearn import linear_model
from sklearn import tree
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

classifiers = {
    'DecisionTree':tree.DecisionTreeClassifier(),
    'LinearRegression':linear_model.LinearRegression(),
    'GaussianNB':GaussianNB(),
    'KNeighbors':KNeighborsClassifier(3),
    'RandomForest': RandomForestClassifier(),
    'MultinomialNB': MultinomialNB(),
}


markers = {'30min':':','08min':'-'}
fpr = {}
tpr = {}
th = {}


# if access = 1.0, all nodes have access
# if access < 0.95 criteria, less than 95% of nodes have access - need to flag a warning if this is the case or lower
criterias = np.linspace(0.75,0.95,3)

results = pandas.DataFrame(columns = ['setting','criteria','classifier','AUC','SEE'])
row = 0

for criteria in criterias: # check if access < criteria
    for setting in settings:
        target = access[setting] 

        ytrain = np.round(target.loc[train]<criteria)
        ytest =  np.round(target.loc[test]<criteria)

        for clf_name,clf in classifiers.iteritems():
            ypred = pandas.DataFrame()
            yprob = pandas.DataFrame()

            for i in target.columns:
                clf.fit(xtrain, ytrain[i])
                ypred[i] = clf.predict(xtest)

                if hasattr(clf, "predict_proba"):
                    yprob[i] = clf.predict_proba(xtest)[:,-1]
                else:
                    yprob[i] = clf.decision_function(xtest)     
                    yprob[i] = (yprob[i] - yprob[i].min()) / (yprob[i].max() - yprob[i].min())

            actual = ytest.values.flatten()
            pred = ypred.values.flatten()
            prob = yprob.values.flatten()

            fpr[row], tpr[row], th[row] = metrics.roc_curve(actual, prob)

            valid = ~np.isnan(prob)
            SEE = np.sqrt(sum((prob[valid]-actual[valid])**2)/sum(valid))
            AUC = metrics.auc(fpr[row],tpr[row])

            results.loc[row] = [settings[setting],criteria,clf_name,AUC,SEE]
            row +=1

"""Save results to tex"""
results.sort_values(by=['setting','criteria']).set_index(['setting','criteria','classifier']).unstack('criteria').round(3).to_latex('flood/results.tex')

"""Summary of reference node"""
ref_node = 1
t1=dist2h[ref_node].dropna().describe()
t2=diff2h[ref_node].dropna().describe()
t3=diff2h[ref_node].loc[diff2h[ref_node]>0].dropna().describe()
summary_ref_node=pandas.DataFrame({'dist2h':t1,'diff2h':t2,'diff2h > 0':t3},columns=['dist2h','diff2h','diff2h > 0']).round().applymap(int)
"""Summary of all nodes"""
t4=dist2h.mean().dropna().describe()
t5=diff2h.mean().dropna().describe()
t6=diff2h[diff2h>0].mean().dropna().describe()
summary_all=pandas.DataFrame({'dist2h_all':t4,'diff2h_all':t5,'diff2h_all > 0':t6},columns=['dist2h_all','diff2h_all','diff2h_all > 0']).round().applymap(int)

"""Summary of node count"""
OA_node_count = pandas.Series({i:len(nodes) for i,nodes in OA_nodes.iteritems()})
OA_node_count.describe()

"""Baseline Flooded scenario description"""
description = {}
for setting in settings:
    description[(setting,'baseline')] = baseline[setting].describe()
    description[(setting,'mu')] = access[setting].mean().describe()
    description[(setting,'sigma')] = access[setting].std().describe()
description = pandas.DataFrame(description).T
description.round(3).to_latex('flood/OA_mean.tex')