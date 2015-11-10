import gzip
import csv
import numpy as np

# Read as list
all_nodes = {}
with gzip.open('flood/all_nodes.csv.gz', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
		k = int(row[0])
		all_nodes[k] = [float(v) for v in row[1:]]

import matplotlib.pyplot as plt

plt.ion()
plt.close('all')
plt.figure()
plt.imshow(np.reshape(all_nodes.values()[24],(100,200)),interpolation='none')
plt.colorbar()

plt.ion()
plt.close('all')
plt.figure()
plt.imshow(np.reshape(all_nodes[21841],(100,200)),interpolation='none')
plt.colorbar()

# Read list of nodes in the bounding box
with open('flood/nodes_in_bbox.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    nodes_in_bbox = [int(k) for k,lon,lat in reader]

# Write min and max to file
min_dist = {}
max_dist = {}
max_accessible_dist = {}
for k in nodes_in_bbox:
	a = np.array(all_nodes[k])
	max_accessible_dist = a.max()
	np.place(a,a<0,np.inf)
	max_dist[k] = a.max()	
	min_dist[k] = a.min()
	if min_dist[k] == np.inf:
		min_dist[k] = -99.99
	if max_dist[k] == np.inf:
		max_dist[k] = -99.99

with open('flood/max_accessible.csv','wb') as csvfile:
	writer = csv.writer(csvfile, delimiter=',')
	for k in nodes_in_bbox:
		writer.writerow([k,max_accessible_dist[k]])

with open('flood/min.csv','wb') as csvfile:
	writer = csv.writer(csvfile, delimiter=',')
	for k in nodes_in_bbox:
		writer.writerow([k,min_dist[k]])

with open('flood/max.csv','wb') as csvfile:
	writer = csv.writer(csvfile, delimiter=',')
	for k in nodes_in_bbox:
		writer.writerow([k,max_dist[k]])

# Calculate distance to hospital
normal_dist_to_hospital = [normal.route_length[hospital][k] for k in nodes_in_bbox]
with open('flood/normal_dist_to_hospital.csv','wb') as csvfile:
	writer = csv.writer(csvfile, delimiter=',')
	for k,v in zip(nodes_in_bbox,normal_dist_to_hospital):
		writer.writerow([k,v])

# Calculate ambulance time to hospital
ambulance_speed = 30 # km/h
normal_time_to_hospital = [d/ambulance_speed/1000*60 for d in normal_dist_to_hospital] # in mins
with open('flood/normal_time_to_hospital.csv','wb') as csvfile:
	writer = csv.writer(csvfile, delimiter=',')
	for k,v in zip(nodes_in_bbox,normal_time_to_hospital):
		writer.writerow([k,v])