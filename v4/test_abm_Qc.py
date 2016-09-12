import sys
sys.path.append('core')
import abm, db
reload(abm)
reload(db)
import matplotlib.pyplot as plt

# self = abm.Sim('flood','Carlisle')
self = abm.Sim('test_abm_Qc',(-2.88612967859241, 54.91133135582125, -2.9609706894330743, 54.88327220183169))
self.n = 500000
self.use_buffer = False

self.scenarios=['ia']
# self.run(agent_progress_bar=False)

self.h.init_route()
self.init_scenario('ia')
self.init_agents()

# Load tracks file
self.load_events()

# Initialise these to zero
for u,v in self.h.G.edges_iter():
    self.h.G[u][v]['agg_time'] = []

# Aggregate simulation time
for event in self.events:
    time,u = event[0]
    for time,v in event[1:]:
        self.h.G[u][v]['agg_time'].append(time)
        u = v

# ------------------------------------------        

# Sanity check
total = 0
for v in self.h.destins:
    for u in self.h.G.predecessors(v):
        try:
            total += len(self.h.G[u][v]['agg_time'])
            print u,v,total
        except KeyError:
            pass
print total # This should be equal to the population

# ------------------------------------------

# self.run(agent_progress_bar=False)

# ------------------------------------------

# Detect all the leaf nodes in the route file
# - Leaf nodes do not have any preceding nodes

# ------------------------------------------

# Initialise these to zero
for u,v in self.h.G.edges_iter():
    self.h.G[u][v]['agg_distance'] = []

# Compute aggregate distance and population
for u,v in self.h.G.edges_iter():
    try:
        this_population = self.h.G[u][v]['queue_length']
        if this_population > 0:
            destin = self.h.G[u][v]['nearest_destin']
            this_distance = self.h.G[u][v]['distance']
            this_agg_distance = [float(i+1)/this_population*this_distance for i in range(this_population)]
            while True:
                self.h.G[u][v]['agg_distance'] += this_agg_distance
                # If v is the destin, break the while loop
                if v == destin:
                    break
                else:
                    u,v = v,self.h.route[destin][v]
                    this_agg_distance = [self.h.G[u][v]['distance']+i for i in this_agg_distance]
    except KeyError:
        pass

# ------------------------------------------

# Sanity check
total = 0
for v in self.h.destins:
    for u in self.h.G.predecessors(v):
        try:
            total += len(self.h.G[u][v]['agg_distance'])
            print u,v,total
        except KeyError:
            pass
print total # This should be equal to the population

# ------------------------------------------

# Compute Qc for every edge
import numpy as np
q_max = max(abm.fd.q)/60
for u,v,d in self.h.G.edges_iter(data=True):
    d['Qc'] = 0
    d['T90p'] = 0
    d['T90s'] = 0
    d['T90f'] = 0
    d['D90'] = 0
    d['kc'] = 0
    d['A/m'] = 0
    population = len(d['agg_distance'])
    if population > 0:
        d['A/m'] = population/d['assumed_width']
        d['D90'] = np.percentile(d['agg_distance'],90)
        d['T90f'] = d['D90']/abm.fd.v_ff/60        
        d['T90s'] = np.percentile(d['agg_time'],90)        
        d['Qc'] = population/d['assumed_width']/d['T90f']
        d['kc'] = population/d['assumed_width']/d['D90']
        multiplier = 1
        # If the characteristic density is greater than density for minimum velocity,
        # 
        multiplier = d['kc']/abm.fd.k_vmin
        if multiplier < 1:
            multiplier = 1
        d['T90p'] = multiplier*d['D90']/abm.fd.velocity(d['kc']) # Predicted T90 
        # if d['Qc'] > q_max:
        #     print u,v,population,d['D90'],d['Qc'],d['nearest_destin'],d['destin_distance'][d['nearest_destin']]

# ------------------------------------------

def polyfit(x, y, degree):
    results = {}
    coeffs = np.polyfit(x, y, degree)
     # Polynomial Coefficients
    results['polynomial'] = coeffs.tolist()
    # r-squared
    p = np.poly1d(coeffs)
    # fit values, and mean
    yhat = p(x)                         # or [p(z) for z in x]
    ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
    results['determination'] = ssreg / sstot
    return results

# ------------------------------------------
# Start Drawing

plt.ion()

# ------------------------------------------

import networkx as nx
edgelist,data = zip(*[((u,v),d['T90s'])for u,v,d in self.h.G.edges(data=True)])
thickest_line_width = 2
edgewidth=np.array(data)/max(data)*thickest_line_width

plt.figure()
edges = nx.draw_networkx_edges(self.h.G,pos=self.h.G.node,arrows=False,edgelist=edgelist,edge_color=data,width=edgewidth,alpha=1.0)
plt.colorbar(edges)
plt.title('T90 simulated')
plt.savefig('figs/T90s-graph.pdf')

# ------------------------------------------
import networkx as nx
edgelist,data = zip(*[((u,v),d['T90p'])for u,v,d in self.h.G.edges(data=True)])
thickest_line_width = 2
edgewidth=np.array(data)/max(data)*thickest_line_width

plt.figure()
edges = nx.draw_networkx_edges(self.h.G,pos=self.h.G.node,arrows=False,edgelist=edgelist,edge_color=data,width=edgewidth,alpha=1.0)
plt.colorbar(edges)
plt.title('T90 predicted')
plt.savefig('figs/T90p-graph.pdf')

# ------------------------------------------
# Agent per metre graph

import networkx as nx
edgelist,data = zip(*[((u,v),d['A/m'])for u,v,d in self.h.G.edges(data=True)])
thickest_line_width = 2
edgewidth=np.array(data)/max(data)*thickest_line_width

plt.figure()
edges = nx.draw_networkx_edges(self.h.G,pos=self.h.G.node,arrows=False,edgelist=edgelist,edge_color=data,width=edgewidth,alpha=1.0)
plt.colorbar(edges)
plt.title('Agents per metre')
plt.savefig('figs/agents-per-metre-graph.pdf')

# ------------------------------------------
# Create a list of destin edges

destin_edges = []
for v in self.h.destins:
    for u in self.h.G.predecessors(v):
        destin_edges += [(u,v)]

# Plot predicted vs simulated T90 at every edge
kc_all,s_all,p_all = zip(*[(d['kc'],d['T90s'],d['T90p']) for u,v,d in self.h.G.edges(data=True)])
# Plot predicted vs simulated T90 at exits
kc,s,p = zip(*[(self.h.G[u][v]['kc'],self.h.G[u][v]['T90s'],self.h.G[u][v]['T90p']) for u,v in destin_edges])
print polyfit(p_all,s_all,1)
print polyfit(p_all,s_all,2)

# Plot predicted vs reality
plt.figure()
plt.xlabel('Predicted')
plt.ylabel('Simulated')
plt.scatter(p_all,s_all,c=kc_all,marker='.',linewidth=0)
plt.scatter(p,s,c=kc)
plt.colorbar()
plt.plot([0,350],[0,700])
plt.plot([0,350],[0,350])
plt.plot([0,700],[0,350],c='r')
plt.title('Predicted vs Simulated T90 at every edge')
plt.savefig('figs/T90p-T90s.pdf')

# ------------------------------------------
# T90s vs agents/metre

# Plot predicted vs simulated T90 at every edge
kc_all,s_all,am_all = zip(*[(d['kc'],d['T90s'],d['A/m']) for u,v,d in self.h.G.edges(data=True)])
kc,s,am = zip(*[(self.h.G[u][v]['kc'],self.h.G[u][v]['T90s'],self.h.G[u][v]['A/m']) for u,v in destin_edges])
print polyfit(p_all,am_all,1)
print polyfit(p_all,am_all,2)




# Plot predicted vs reality
plt.figure()
plt.xlabel('A/m')
plt.ylabel('Simulated')
plt.scatter(am_all,s_all,c=kc_all,marker='.',linewidth=0)
plt.scatter(am,s,c=kc)
plt.colorbar()
plt.title('A/m vs Simulated T90 at every edge')
plt.savefig('figs/agents-per-metre-vs-T90s.pdf')

# ------------------------------------------
# Normalised T90s-vs-T90p

# Plot predicted vs simulated T90 at every edge
kc_all,s_all_norm,p_all_norm = zip(*[(d['kc'],d['T90s']/d['T90f'],d['T90p']/d['T90f']) for u,v,d in self.h.G.edges(data=True) if d['T90f'] > 0])
# Plot predicted vs simulated T90 at exits
kc,s_norm,p_norm = zip(*[(self.h.G[u][v]['kc'],self.h.G[u][v]['T90s']/self.h.G[u][v]['T90f'],self.h.G[u][v]['T90p']/self.h.G[u][v]['T90f']) for u,v in destin_edges])
print polyfit(p_all_norm,s_all_norm,1)
print polyfit(p_all_norm,s_all_norm,2)

print ss.pearsonr(p_all_norm,s_all_norm)

# Plot predicted vs reality
plt.figure()
plt.xlabel('Predicted T90/T90f')
plt.ylabel('Simulated T90/T90f')
plt.scatter(p_all_norm,s_all_norm,c=kc_all,marker='.',linewidth=0)
plt.scatter(p_norm,s_norm,c=kc)
plt.colorbar()
plt.plot([0,max(p_norm)],[0,max(s_norm)])
plt.plot([0,max(p_norm)],[0,max(s_norm)])
plt.plot([0,max(p_norm)],[0,max(s_norm)],c='r')
plt.title('Predicted vs Simulated T90 at every edge normalised by T90f')
plt.savefig('figs/T90p-T90s-norm-T90f.pdf')

# ------------------------------------------
# Normalised T90s-vs-T90p in LOG scale

# Plot predicted vs simulated T90 at every edge
kc_all,s_all_norm,p_all_norm = zip(*[(d['kc'],d['T90s']/d['T90f'],d['T90p']/d['T90f']) for u,v,d in self.h.G.edges(data=True) if d['T90f'] > 0])
# Plot predicted vs simulated T90 at exits
kc,s_norm,p_norm = zip(*[(self.h.G[u][v]['kc'],self.h.G[u][v]['T90s']/self.h.G[u][v]['T90f'],self.h.G[u][v]['T90p']/self.h.G[u][v]['T90f']) for u,v in destin_edges])
print polyfit(p_all_norm,s_all_norm,1)
print polyfit(p_all_norm,s_all_norm,2)

print ss.pearsonr(np.log(p_all_norm),np.log(s_all_norm))

# Plot predicted vs reality
plt.figure()
plt.xlabel('Predicted T90/T90f')
plt.ylabel('Simulated T90/T90f')
plt.scatter(p_all_norm,s_all_norm,c=kc_all,marker='.',linewidth=0)
plt.scatter(p_norm,s_norm,c=kc)
plt.colorbar()
plt.plot([0,max(p_norm)],[0,max(s_norm)])
plt.plot([0,max(p_norm)],[0,max(s_norm)])
plt.plot([0,max(p_norm)],[0,max(s_norm)],c='r')
plt.xscale('log')
plt.yscale('log')
plt.title('Predicted vs Simulated T90 at every edge normalised by T90f')
plt.savefig('figs/T90p-T90s-norm-T90f-log.pdf')

# ------------------------------------------
# Correct downstream T90 if upstream T90 is greater

# Initialise T90 upstream
# If the edge downstream has a greater T90p then make sure that it 
# This is supposed to correct for under prediction (when Ts is above the diagonal)
for u,v,d in self.h.G.edges(data=True):   
    d['T90p_upstream'] = d['T90p']

for u,v,initial_d in self.h.G.edges(data=True):
    try:
        # Follow the route downstream
        destin = initial_d['nearest_destin']
        T90p_upstream = initial_d['T90p_upstream']
        while v != destin:
            u = v
            v = self.h.route[destin][u]
            d = self.h.G[u][v]
            if T90p_upstream > d['T90p_upstream']:
                velocity = abm.fd.velocity(d['queue_length']/d['area'])
                T90p_upstream += d['distance']/velocity
                d['T90p_upstream'] = T90p_upstream
            if v == 350:
                print d['T90p_upstream']
    except KeyError:
        pass

# ------------------------------------------

# Draw
import networkx as nx
edgelist,data = zip(*[((u,v),d['T90p_upstream'])for u,v,d in self.h.G.edges(data=True)])
thickest_line_width = 2
edgewidth=np.array(data)/max(data)*thickest_line_width

plt.figure()
edges = nx.draw_networkx_edges(self.h.G,pos=self.h.G.node,arrows=False,edgelist=edgelist,edge_color=data,width=edgewidth,alpha=1.0)
plt.colorbar(edges)
plt.title('T90 predicted upstream correction')
plt.savefig('figs/T90p-upstream.pdf')

# ------------------------------------------

# Predicted vs simulated T90 at every edge
kc_all,s_all,p_upstream_all = zip(*[(d['kc'],d['T90s'],d['T90p_upstream']) for u,v,d in self.h.G.edges(data=True) if d['T90s'] > 0])
# Predicted vs simulated T90 at exits
kc,s,p_upstream = zip(*[(self.h.G[u][v]['kc'],self.h.G[u][v]['T90s'],self.h.G[u][v]['T90p_upstream']) for u,v in destin_edges])
print polyfit(p_upstream_all,s_all,1)
print polyfit(p_upstream_all,s_all,2)
print 'correlation', np.corrcoef(s_all,p_upstream_all)

# Plot predicted vs simulated taking upstream time into account
plt.figure()
plt.scatter(p_upstream_all,s_all,c=kc_all,marker='.',linewidth=0)
plt.scatter(p,s,c=kc)
plt.colorbar()
plt.plot([0,350],[0,700])
plt.plot([0,350],[0,350])
plt.plot([0,700],[0,350])
plt.xlabel('Predicted')
plt.ylabel('Simulated')
plt.title('Predicted vs Simulated T90 taking upstream time into acount')
plt.savefig('figs/T90p-T90s-upstream.pdf')

# ------------------------------------------

# Predicted/simulated time vs kc
kc_nz,p_over_s_nz = zip(*[(kcn,pn/sn) for kcn,pn,sn in zip(kc_all,p_all,s_all) if kcn > 0])
# Fit function
pv=np.polyfit(kc_nz,p_over_s_nz,1)
# Regular figures
nbins = 12
# def CI_fig(CI,nbins = 1000):
CI = 67
upper = 50 + CI/2
lower = 50 - CI/2   
# Digitize
all_x = np.array(kc_nz)
all_y = np.array(p_over_s_nz)
H,xedges=np.histogram(all_x,bins=nbins)
digitized = np.digitize(all_x, xedges)
x_range = np.array(range(len(xedges)))+1
x_len = np.array([len(all_x[digitized == i]) for i in x_range if np.any(all_y[digitized == i])])
x_mean = np.array([all_x[digitized == i].mean() for i in x_range if np.any(all_y[digitized == i])])
y_lower = np.array([np.percentile(all_y[digitized == i],lower) for i in x_range if np.any(all_y[digitized == i])])
y_median = np.array([np.percentile(all_y[digitized == i],50) for i in x_range if np.any(all_y[digitized == i])])
y_upper = np.array([np.percentile(all_y[digitized == i],upper) for i in x_range if np.any(all_y[digitized == i])])

# Plot kc vs p/s
plt.figure()
plt.scatter(kc_nz,p_over_s_nz)
plt.xlabel('$k_c$',fontsize=20)
plt.ylabel('$T_p/T_s$',fontsize=20)
plt.scatter(kc_nz,np.polyval(pv,kc_nz),c='r')
plt.plot(x_mean,y_median,label='$Tp/Ts$')
plt.fill_between(x_mean,y_upper,y_lower,facecolor='gray',alpha=0.5)
plt.xlabel('$k_c$',fontsize=20)
plt.ylabel('$T_p/T_s$',fontsize=20)
plt.savefig('figs/kc-T90pT90s-upstream.pdf')

# ------------------------------------------