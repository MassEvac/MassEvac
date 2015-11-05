import gzip
import csv
import numpy as np
all_nodes = {}
with gzip.open('all_nodes.csv.gz', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
		a = all_nodes[node]
		np.place(a,a<0,np.inf)    	
		all_nodes[int(row[0])] = np.reshape(a,(100,200))

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

min_dist = {}
max_dist = {}
for node in all_nodes:
	min_dist[node] = a.min()
	max_dist[node] = a.max()
	if min_dist[node] == np.inf:
		min_dist[node] = -99.99
	if max_dist[node] == np.inf:
		max_dist[node] = -99.99

with open('flood/nodes_in_bbox.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
		a = all_nodes[node]
		np.place(a,a<0,np.inf)    	
		all_nodes[int(row[0])] = np.reshape(a,(100,200))

with open('min.csv','wb') as csvfile:
	writer = csv.writer(csvfile, delimiter=',')
	for k in min_dist:
		writer.writerow([k,min_dist[k]])

with open('max.csv','wb') as csvfile:
	writer = csv.writer(csvfile, delimiter=',')
	for k in max_dist:
		writer.writerow([k,max_dist[k]])