# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# Here I am going to investigate the correlations from Matlab outputs and discuss the results.

# <codecell>

import os,scipy,pickle
import scipy.io
folder = '../../OSM/urban'

labels = sorted([each[:-4] for each in os.listdir(folder) if each.endswith('.mat')])

file = open("{0}/cities.txt".format(folder), "r")
my_cities = file.read().split('\n')
file.close()

metrics = {}

nans = {}

rowsum = zeros((len(my_cities),1))

for label in labels:
    metrics[label] = scipy.io.loadmat('{0}/{1}.mat'.format(folder,label))['col']
    nans[label] = sum(np.isnan(metrics[label]))
    rowsum = rowsum + metrics[label];

# <codecell>

file = open('et_stats.pickle','r')
mu,sigma,median,ninetieth=pickle.load(file)
file.close()

# <codecell>

for i,j in enumerate(ninetieth[0]):
    if j == None:
        ninetieth[0][i] = NaN

scipy.io.savemat('ninetieth_0.mat',{'ninetieth_0':ninetieth[0]})

# <codecell>

for i,j in enumerate(ninetieth[1]):
    if j == None:
        ninetieth[1][i] = NaN

scipy.io.savemat('ninetieth_1.mat',{'ninetieth_1':ninetieth[1]})

# <codecell>

et_ratio = []
for i,j in zip(ninetieth[0],ninetieth[1]):
    if i and j:
        et_ratio.append(j/i)
    else:
        et_ratio.append(None)
zip(my_cities,et_ratio)

# <markdowncell>

# This is our list of unusable columns from the processing done over in matlab

# <codecell>

labels

# <codecell>

nans

# <markdowncell>

# Filter out the 'nan' rows so that the data is usable.

# <codecell>

usable = find(~np.isnan(rowsum))
unusable = find(np.isnan(rowsum))

# <codecell>

find([city=='Telford' for city in my_cities])

# <codecell>

[my_cities[i] for i in unusable]

# <codecell>


# <markdowncell>

# The following shows the histogram distribution of data in our metrics:

# <codecell>

def distribution_fig(label):
    fig,axes = subplots(1, 2) # one row, two columns
    data = metrics[label][usable]
    n, bins, patches = axes[0].hist(data, bins=no_bins, normed=False, histtype='stepfilled')
    n, bins, patches = axes[1].hist(data, bins=np.logspace(log10(min(data)),log10(max(data)),10), normed=False, histtype='stepfilled')    
    axes[0].set_xlabel('{1}'.format(col,label))    
    axes[0].set_ylabel('Number of cities')    
    axes[0].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    axes[1].set_xlabel('log({1})'.format(col,label))
    axes[1].set_ylabel('Number of cities')
    axes[1].set_xscale('log')
    plt.close(fig)
    return fig

# <markdowncell>

# Nodes in the network before and after

# <codecell>

no_bins = 15
labels_here = ['nodes_before','nodes_after']
for label in labels_here:
    display(distribution_fig(label))

# <markdowncell>

# Edges in the network before and after

# <codecell>

no_bins = 15
labels_here = ['edges_before','edges_after']
for label in labels_here:
    display(distribution_fig(label))

# <markdowncell>

# Area of the network before and after

# <codecell>

no_bins = 15
labels_here = ['area_before','area_after']
for label in labels_here:
    display(distribution_fig(label))

# <markdowncell>

# Total length of the network before and after

# <codecell>

no_bins = 15
labels_here = ['length_before','length_after']
for label in labels_here:
    display(distribution_fig(label))

# <markdowncell>

# Average width of the roads before and after

# <codecell>

no_bins = 15
labels_here = ['width_before','width_after']
for label in labels_here:
    display(distribution_fig(label))

# <markdowncell>

# Mean degree of the network before and after simplification

# <codecell>

no_bins = 15
labels_here = [ 'mean_degree_after', 'mean_degree_before']
for label in labels_here:
    display(distribution_fig(label))

# <markdowncell>

# Number of leaf nodes and edges after simplification

# <codecell>

no_bins = 15
labels_here = ['no_leaf_nodes','no_leaf_edges']
for label in labels_here:
    display(distribution_fig(label))

# <markdowncell>

# Fraction of nodes and edges after simplification that are leaf nodes

# <codecell>

no_bins = 15
labels_here = ['connected_component_before','fraction_leaf_edges']
for label in labels_here:
    display(distribution_fig(label))

# <codecell>

no_bins = 15
labels_here = ['fraction_leaf_nodes','fraction_leaf_edges']
for label in labels_here:
    display(distribution_fig(label))

# <markdowncell>

# Network average clustering coefficient before and after simplification

# <codecell>

no_bins = 15
labels_here = ['network_average_clustering_coefficient_before_unweighted','network_average_clustering_coefficient_after_unweighted']
for label in labels_here:
    display(distribution_fig(label))

# <codecell>


# <codecell>

list(my_cities)[]

# <codecell>

find(metrics['network_average_clustering_coefficient_before_unweighted'][usable]==0)

# <markdowncell>

# Network average clustering coefficient before and after simplification where bigger roads are weighted more heavily

# <codecell>

no_bins = 15
labels_here = ['network_average_clustering_coefficient_after_unweighted','network_average_clustering_coefficient_before_unweighted']
for label in labels_here:
    display(distribution_fig(label))

# <codecell>

#['area_after',
# 'area_before',
 'connected_components_after',
 'connected_components_before',
# 'edges_after',
# 'edges_before',
 'edges_per_connected_components_after',
 'edges_per_connected_components_before',
#  'fraction_leaf_edges',
#  'fraction_leaf_nodes',
# 'length_after',
# 'length_before',
#  'mean_degree_after',
#  'mean_degree_before',
 'network_average_clustering_coefficient_after_unweighted',
 'network_average_clustering_coefficient_after_weighted',
 'network_average_clustering_coefficient_before_unweighted',
 'network_average_clustering_coefficient_before_weighted',
# 'no_leaf_edges',
# 'no_leaf_nodes',
# 'nodes_after',
# 'nodes_before',
 'nodes_per_connected_components_after',
 'nodes_per_connected_components_before',
# 'width_after',
# 'width_before']

