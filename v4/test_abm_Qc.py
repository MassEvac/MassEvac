import sys
sys.path.append('core')
import abm, db
reload(abm)
reload(db)

# self = abm.Sim('flood','Carlisle')
self = abm.Sim('test_abm_Qc',(-2.88612967859241, 54.91133135582125, -2.9609706894330743, 54.88327220183169))
self.n = 500000
self.use_buffer = False

self.scenarios=['ia']
self.run(agent_progress_bar=False)

self.h.init_route()
self.init_scenario('ia')
self.init_agents()

# Load tracks file
self.tracks_file(mode='load')

# Initialise these to zero
for u,v in self.h.G.edges_iter():
    self.h.G[u][v]['agg_time'] = []

# Aggregate simulation time
for t in self.tracks:
    time,u = t.events[0]
    for time,v in t.events[1:]:
        self.h.G[u][v]['agg_time'] += [time]
        u = v

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


# self.run(agent_progress_bar=False)

# Detect all the leaf nodes in the route file
# - Leaf nodes do not have any preceding nodes

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
    population = len(d['agg_distance'])
    if population > 0:
        d['D90'] = np.percentile(d['agg_distance'],90)
        d['T90f'] = d['D90']/abm.fd.vFf/60        
        d['T90s'] = np.percentile(d['agg_time'],90)        
        d['Qc'] = population/d['assumed_width']/d['T90f']
        d['kc'] = population/d['assumed_width']/d['D90']        
        multiplier = d['kc']/abm.fd.kFlat
        if multiplier < 1:
            multiplier = 1
        d['T90p'] = multiplier*d['D90']/abm.fd.velocity(d['kc']) # Predicted T90 
        if d['Qc'] > q_max:
            print u,v,population,d['D90'],d['Qc'],d['nearest_destin'],d['destin_distance'][d['nearest_destin']]

# Draw
import networkx as nx
edgelist,data = zip(*[((u,v),d['T90s'])for u,v,d in self.h.G.edges(data=True)])
thickest_line_width = 2
edgewidth=np.array(data)/max(data)*thickest_line_width
plt.figure()
edges = nx.draw_networkx_edges(self.h.G,pos=self.h.G.node,arrows=False,edgelist=edgelist,edge_color=data,width=edgewidth,alpha=1.0)
plt.colorbar(edges)

# Draw
import networkx as nx
edgelist,data = zip(*[((u,v),d['T90p'])for u,v,d in self.h.G.edges(data=True)])
thickest_line_width = 2
edgewidth=np.array(data)/max(data)*thickest_line_width
plt.figure()
edges = nx.draw_networkx_edges(self.h.G,pos=self.h.G.node,arrows=False,edgelist=edgelist,edge_color=data,width=edgewidth,alpha=1.0)
plt.colorbar(edges)

# Create a list of destin edges
destin_edges = []
for v in self.h.destins:
    for u in self.h.G.predecessors(v):
        destin_edges += [(u,v)]

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
                T90p_upstream += d['distance']/abm.fd.vFf/60
                d['T90p_upstream'] = T90p_upstream
            if v == 350:
                print d['T90p_upstream']
    except KeyError:
        pass

# Draw
import networkx as nx
edgelist,data = zip(*[((u,v),d['T90p_upstream'])for u,v,d in self.h.G.edges(data=True)])
thickest_line_width = 2
edgewidth=np.array(data)/max(data)*thickest_line_width
plt.figure()
edges = nx.draw_networkx_edges(self.h.G,pos=self.h.G.node,arrows=False,edgelist=edgelist,edge_color=data,width=edgewidth,alpha=1.0)
plt.colorbar(edges)

def polyfit(x, y, degree):
    results = {}
    coeffs = numpy.polyfit(x, y, degree)
     # Polynomial Coefficients
    results['polynomial'] = coeffs.tolist()
    # r-squared
    p = numpy.poly1d(coeffs)
    # fit values, and mean
    yhat = p(x)                         # or [p(z) for z in x]
    ybar = numpy.sum(y)/len(y)          # or sum(y)/len(y)
    ssreg = numpy.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = numpy.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
    results['determination'] = ssreg / sstot
    return results
 
# Plot predicted vs reality
plt.figure()

# Plot predicted vs simulated T90 at every edge
kc_all,s_all,p_all = zip(*[(d['kc'],d['T90s'],d['T90p']) for u,v,d in self.h.G.edges(data=True)])
plt.scatter(p_all,s_all,c=kc_all,marker='.',linewidth=0)

# Plot predicted vs simulated T90 at exits
kc,s,p = zip(*[(self.h.G[u][v]['kc'],self.h.G[u][v]['T90s'],self.h.G[u][v]['T90p']) for u,v in destin_edges])
plt.scatter(p,s,c=kc)
plt.colorbar()

plt.plot([0,350],[0,700])
plt.plot([0,350],[0,350])
plt.plot([0,700],[0,350],c='r')

print polyfit(p_all,s_all,1)
print polyfit(p_all,s_all,2)

# Plot predicted vs simulated taking upstream time into account
plt.figure()

# Less than optimum density
# Plot predicted vs simulated T90 at every edge
kc_all,s_all,p_upstream_all = zip(*[(d['kc'],d['T90s'],d['T90p_upstream']) for u,v,d in self.h.G.edges(data=True)])
plt.scatter(p_upstream_all,s_all,c=kc_all,marker='.',linewidth=0)

# Plot predicted vs simulated T90 at exits
kc,s,p_upstream = zip(*[(self.h.G[u][v]['kc'],self.h.G[u][v]['T90s'],self.h.G[u][v]['T90p_upstream']) for u,v in destin_edges])
plt.scatter(p,s,c=kc,marker='x')
plt.colorbar()

plt.plot([0,350],[0,700])
plt.plot([0,350],[0,350])
plt.plot([0,700],[0,350],c='g')

print polyfit(p_upstream_all,s_all,1)
print polyfit(p_upstream_all,s_all,2)



# Plot predicted vs simulated taking upstream time into account
plt.figure()

# Less than optimum density
# Plot predicted vs simulated T90 at every edge
kc_all,s_all,p_upstream_all = zip(*[(d['kc'],d['T90s']*len(d['agg_distance'])/300000,d['T90p_upstream']*len(d['agg_distance'])/300000) for u,v,d in self.h.G.edges(data=True) if d['T90f']>0])
plt.scatter(p_upstream_all,s_all,c=kc_all,marker='.',linewidth=0)

print polyfit(p_upstream_all,s_all,1)
print polyfit(p_upstream_all,s_all,2)

# Plot predicted vs simulated T90 at exits
kc,s,p_upstream = zip(*[(self.h.G[u][v]['kc'],self.h.G[u][v]['T90s'],self.h.G[u][v]['T90p_upstream']) for u,v in destin_edges])
plt.scatter(p,s,c=kc,marker='x')
plt.colorbar()

plt.plot([0,350],[0,700])
plt.plot([0,350],[0,350])
plt.plot([0,700],[0,350],c='g')

