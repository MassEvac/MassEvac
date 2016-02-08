import gzip
import csv
import numpy as np
import matplotlib.pyplot as plt

# Read as list
all_nodes = {}
input_file = '30cm'
with gzip.open('flood/all_nodes_{}.csv.gz'.format(input_file), 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
		k = int(row[0])
		all_nodes[k] = [float(v) for v in row[1:]]

# Read list of nodes in the bounding box
with open('flood/nodes_in_bbox.csv', 'rb') as csvfile:
    reader = csv.DictReader(csvfile, delimiter=',')
    nodes_in_bbox = [int(row['node_id']) for row in reader]

plt.close('all')
plt.ion()

plt.figure()
plt.imshow(np.reshape(all_nodes.values()[24],(100,100)),interpolation='none')
plt.colorbar()

plt.figure()
plt.imshow(np.reshape(all_nodes[26753],(100,100)),interpolation='none')
plt.colorbar()