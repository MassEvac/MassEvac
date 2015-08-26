
import abm
import numpy as np
import networkx as nx

s = abm.Sim('bristol25','City of Bristol')
s.h.init_destins()
s.h.init_route()

s.scenario = 'ia'

s.h.init_EM()
s.h.init_EA()

# #-----------------------------------------

# # Enlist subgraph nodes from edge list
# SG_nodes = {}

# for u,v,d in s.h.G.edges_iter(data=True):
# 	nearest_x = np.nan
# 	nearest_du = np.inf
# 	for x in s.h.destins:
# 		try:
# 			dv = s.h.route_length[x][v]

# 			# u is further away than v
# 			du = dv + d['distance']

# 			if du < nearest_du:
# 				nearest_x = x
# 				nearest_du = du
# 				nearest_dv = dv

# 		except KeyError:
# 			pass

# 	if not np.isnan(nearest_x):
# 		s.h.G[u][v]['nearest_x'] = nearest_x
# 		s.h.G[u][v]['nearest_du'] = nearest_du
# 		s.h.G[u][v]['nearest_dv'] = nearest_dv

# 		try:
# 			SG_nodes[nearest_x].extend([u,v])
# 		except KeyError:
# 			SG_nodes[nearest_x] = [nearest_x,u,v]


# # come up with a way of saving and retrieving this list of 'SG_nodes' at this point
# SG = {}
# for x in s.h.destins:
# 	# remove duplicates
# 	SG_nodes[x] = np.unique(SG_nodes[x])
# 	# create subgraphs
# 	SG[x] = s.h.G.subgraph(SG_nodes[x])


# this = SG[6186]

# nx.draw_networkx_edges(this,pos=s.h.nodes)

# for x in s.h.destins:
# 	r = {}
# 	t = {}

# 	r['nodes'] = SG[x].number_of_nodes()
# 	r['edges'] = SG[x].number_of_edges()

# 	r['length'] = SG[x].size(weight='distance')

# 	for u,v,d in SG[x].edges(data=True):
# 		width = s.h.width[s.h.hiclass[d['highway']]]
# 		# assumed width
# 		SG[x][u][v]['awidth'] = width
# 		# edge area
# 		SG[x][u][v]['area'] = d['distance'] * width

# 	r['area'] = SG[x].size(weight='area')

# 	# strongly connected components
# 	r['strconcom'] = nx.number_strongly_connected_components(SG[x])

# 	# mean degree
# 	r['meandeg'] = 2*r['edges']/r['nodes']

# 	# density - 0 for no edges, 1 for fully connected
# 	r['density'] = nx.density(SG[x])

# 	# number of nodes less than 1000,500,250,125 metres away
# 	# could be a good indicator of how likely the bottlenecks are
# 	t['n1000'] = []
# 	t['n500'] = []
# 	t['n250'] = []
# 	t['n125'] = []

# 	for n in SG[x].nodes_iter():
# 	    l = s.h.route_length[x][n]
# 	    if l < 1000:
# 	        t['n1000'].append(n)
#             if l < 500:
#                 t['n500'].append(n)
#                 if l < 250:
#                     t['n250'].append(n)
#                     if l < 125:
#                         t['n125'].append(n)

# 	r['n1000'] = len(t['n1000'])
# 	r['n500'] = len(t['n500'])
# 	r['n250'] = len(t['n250'])
# 	r['n125'] = len(t['n125'])

# #-----------------------------------------

# reading N from tstep folder
import pickle
import os

# Copy and paste from here
tstep = 0

# Number of bins for that particular destin
destin_bins = {}

# Weighted agent density = sum_agent_density_product/sum_agents_per_metre
all_weighted_density = {}

# #-----------------------------------------

# try:
# 	while True:
# 		print tstep
# 		with open(s.tstep_file(tstep,'N'),'r') as f:
# 			N = pickle.load(f)

# 		for destin in s.h.destins:
# 			this = SG[destin]

# 			# Number of agents per metre length
# 			sum_agents_per_metre = {}
# 			# Number of agents per metre length * Agent density applicable to that stretch
# 			sum_agents_per_metre_density_product = {}

# 			for u,v,d in this.edges_iter(data=True):
# 				i = s.h.EM[u][v] # Edge index

# 				n = N[i]

# 				if u != v:
# 					agents_per_metre = n/d['distance']
# 					density = n/d['awidth']/d['distance']
# 					agents_per_metre_density_product = agents_per_metre * density

# 					for distance_from_destin in range(int(d['nearest_dv']),int(d['nearest_du'])+1):
# 						try:
# 							sum_agents_per_metre[distance_from_destin] = sum_agents_per_metre[distance_from_destin] + agents_per_metre
# 							sum_agents_per_metre_density_product[distance_from_destin] = sum_agents_per_metre_density_product[distance_from_destin] + agents_per_metre_density_product
# 						except KeyError:
# 							sum_agents_per_metre[distance_from_destin] = agents_per_metre
# 							sum_agents_per_metre_density_product[distance_from_destin] = agents_per_metre_density_product

# 			if tstep == 0:
# 				destin_bins[destin] = max(sum_agents_per_metre.keys())+1

# 			if sum(sum_agents_per_metre.values()) > 0:
# 				weighted_density = [0]*destin_bins[destin]

# 				for distance_from_destin in sum_agents_per_metre.keys():
# 					weighted_density[distance_from_destin] = sum_agents_per_metre_density_product[distance_from_destin]/sum_agents_per_metre[distance_from_destin]

# 				if tstep == 0:
# 					all_weighted_density[destin] = [weighted_density]
# 				else:
# 					all_weighted_density[destin].append(weighted_density)

# 		tstep = tstep + 1

# except IOError:
# 	pass



# #saving module
# directory = 'analysis/all_weighted_density'
# os.makedirs(directory)

# for k in s.h.destins:
#     print 'destin,#tstep,#destin_bins'
#     print k,len(all_weighted_density[k]),len(all_weighted_density[k])
#     with open('{0}/{1}'.format(directory,k),'w') as f:
#     	pickle.dump(np.array(all_weighted_density[k]),f)

# #-----------------------------------------

#loading module
directory = 'analysis/all_weighted_density'

for k in s.h.destins:
	try:
		with open('{0}/{1}'.format(directory,k),'r') as f:
			all_weighted_density[k] = pickle.load(f)
			print 'destin,#tstep,#destin_bins'
			print k,len(all_weighted_density[k]),len(all_weighted_density[k][0])
	except IOError:
		pass



# # destin,#tstep,#destin_bins
# # 47968 140 5113
# # destin,#tstep,#destin_bins
# # 39778 318 6629
# # destin,#tstep,#destin_bins
# # 16547 1961 6168
# # destin,#tstep,#destin_bins
# # 32741 168 8084
# # destin,#tstep,#destin_bins
# # 6186 31 2538
# # destin,#tstep,#destin_bins
# # 52523 276 4645
# # destin,#tstep,#destin_bins
# # 49325 536 6077
# # destin,#tstep,#destin_bins
# # 40174 874 8177
# # destin,#tstep,#destin_bins
# # 17232 364 5615
# # destin,#tstep,#destin_bins
# # 6037 141 11353
# # destin,#tstep,#destin_bins
# # 15318 361 4870
# # destin,#tstep,#destin_bins
# # 8888 7 702
# # destin,#tstep,#destin_bins
# # 61116 813 6865
# # destin,#tstep,#destin_bins
# # 1021 297 7389
# # destin,#tstep,#destin_bins
# # 19678 601 6473


# save figures

from matplotlib import pyplot as plt
import os

s.load_results()
s.load_agents()

def genfig(key):
	# Draw the image
	directory = 'analysis/all_weighted_density_fig'
	try:
		os.makedirs(directory)
	except OSError:
		pass

	plt.figure()

	plt.imshow(all_weighted_density[key],interpolation='none')

	# Draw the trajectory
	for k in s.tracked_agent.keys():
		if x == s.X[k]:
			agent_index = k

	yax = s.tracked_agent[agent_index]	
	xax = range(len(yax))

	# plot(xax,yax,linewidth=5,c='w',alpha=0.75)

	plt.xlim(len(all_weighted_density[key]),0)
	# plt.ylim(0,len(all_weighted_density[key][0]))	

	# Label and save the figure
	plt.xlabel('$T/60$')
	plt.ylabel('$D$')
	
	cbar = plt.colorbar()
	
	cbar.set_label('$k$')

	plt.savefig('{0}/{1}.png'.format(directory,key))

x = 61116
genfig(x)