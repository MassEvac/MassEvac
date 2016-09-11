# Produces a figure with agent trajectory
# Used for the PRE paper

import sys
sys.path.append('core')
import abm
import db
reload(abm)
reload(db)

from matplotlib import pyplot as plt
import numpy as np
import pickle
import os

s = abm.Sim('bristol25', 'City of Bristol')
s.h.init_destins()
s.h.init_route()

s.scenario = 'k5-original'

s.h.init_SG(subgraph='nearest')

# -----------------------------------------

# reading N from tstep folder

# Copy and paste from here
tstep = 0

# Number of bins for that particular destin
destin_bins = {}

# Weighted agent density = sum_agent_density_product/sum_agents_per_metre
all_weighted_density = {}

# Loading module
directory = 'analysis/bristol25/all_weighted_density'

for k in s.h.destins:
    try:
        with open('{0}/{1}'.format(directory, k), 'r') as f:
            all_weighted_density[k] = pickle.load(f)
            print 'destin,#tstep,#destin_bins'
            print 'Loading', k, len(all_weighted_density[k]), len(all_weighted_density[k][0])
    except IOError:
        pass

# Process all_weighted_density
if not all_weighted_density:
    while os.path.isfile(s.tstep_file(tstep, 'N')):
        print tstep
        with open(s.tstep_file(tstep,'N'),'r') as f:
            N = pickle.load(f)

        for destin in s.h.destins:
            this = s.h.SG[destin]

            # Number of agents per metre length
            sum_agents_per_metre = {}
            # Number of agents per metre length * Agent density applicable to that stretch
            sum_agents_per_metre_density_product = {}

            for u,v,d in this.edges_iter(data=True):
                i = s.h.EM[u][v] # Edge index

                n = N[i]

                if u != v:
                    agents_per_metre = n/d['distance']
                    width = s.h.width[s.h.hiclass[d['highway']]]
                    density = n/width/d['distance']
                    agents_per_metre_density_product = agents_per_metre * density

                    for distance_from_destin in range(int(d['nearest_dv']),int(d['nearest_du'])+1):
                        try:
                            sum_agents_per_metre[distance_from_destin] = sum_agents_per_metre[distance_from_destin] + agents_per_metre
                            sum_agents_per_metre_density_product[distance_from_destin] = sum_agents_per_metre_density_product[distance_from_destin] + agents_per_metre_density_product
                        except KeyError:
                            sum_agents_per_metre[distance_from_destin] = agents_per_metre
                            sum_agents_per_metre_density_product[distance_from_destin] = agents_per_metre_density_product

            if tstep == 0:
                destin_bins[destin] = max(sum_agents_per_metre.keys())+1

            if sum(sum_agents_per_metre.values()) > 0:
                weighted_density = [0]*destin_bins[destin]

                for distance_from_destin in sum_agents_per_metre.keys():
                    weighted_density[distance_from_destin] = sum_agents_per_metre_density_product[distance_from_destin]/sum_agents_per_metre[distance_from_destin]

                if tstep == 0:
                    all_weighted_density[destin] = [weighted_density]
                else:
                    all_weighted_density[destin].append(weighted_density)

        tstep = tstep + 1

    if all_weighted_density:
        #saving module
        if not os.path.isdir(directory):
            os.makedirs(directory)

        for k in s.h.destins:
            print 'destin,#tstep,#destin_bins'
            print 'Saving',k,len(all_weighted_density[k]),len(all_weighted_density[k])
            with open('{0}/{1}'.format(directory,k),'w') as f:
                pickle.dump(np.array(all_weighted_density[k]),f)
    else:
        print 'ERROR - FOLDER EMPTY: {}'.format(s.tstep_file('','N'))

# Should receive the following output if all_weighted_density is properly loaded

# destin,#tstep,#destin_bins
# 47968 140 5113
# destin,#tstep,#destin_bins
# 39778 318 6629
# destin,#tstep,#destin_bins
# 16547 1961 6168
# destin,#tstep,#destin_bins
# 32741 168 8084
# destin,#tstep,#destin_bins
# 6186 31 2538
# destin,#tstep,#destin_bins
# 52523 276 4645
# destin,#tstep,#destin_bins
# 49325 536 6077
# destin,#tstep,#destin_bins
# 40174 874 8177
# destin,#tstep,#destin_bins
# 17232 364 5615
# destin,#tstep,#destin_bins
# 6037 141 11353
# destin,#tstep,#destin_bins
# 15318 361 4870
# destin,#tstep,#destin_bins
# 8888 7 702
# destin,#tstep,#destin_bins
# 61116 813 6865
# destin,#tstep,#destin_bins
# 1021 297 7389
# destin,#tstep,#destin_bins
# 19678 601 6473

# -----------------------------------------

s.track_list=[0]

s.load_results()
s.load_agents()

def tracked_agent_fig(key):
    # Draw the image
    fig_directory = 'figs/bristol25'
    if not os.path.isdir(fig_directory):
        os.makedirs(fig_directory)

    plt.figure()

    plt.imshow(all_weighted_density[key].T,interpolation='none',cmap='Reds',aspect=0.1)

    # Draw the trajectory
    for k in s.tracked_agent.keys():
        if x == s.X[k]:
            agent_index = k

    yax = s.tracked_agent[agent_index]  
    xax = range(len(yax))

    plt.plot(xax,yax,':',label='Randomly picked agent trajectory',linewidth=5,c='lime')

    plt.xlim(0,len(all_weighted_density[key]))
    plt.ylim(0,len(all_weighted_density[key][0]))   

    # Label and save the figure
    fontsize = 20
    plt.xlabel('$\mathrm{T \ [s]}$',fontsize=fontsize)
    plt.ylabel('$\mathrm{D \ [m]}$',fontsize=fontsize)

    xticks = range(0,len(all_weighted_density[key]),200)
    xticklabs = np.array(xticks)*60

    plt.xticks(xticks,xticklabs)

    print len(xax)

    cbar = plt.colorbar()
    
    cbar.set_label('$\mathrm{k \ [ped/m^2]}$',fontsize=fontsize)

    # plt.gca().tight_layout()

    plt.legend()

    print fig_directory
    plt.savefig('{}/{}.pdf'.format(fig_directory,key))

x = 61116
tracked_agent_fig(x)