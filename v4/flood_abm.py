import sys
sys.path.append('core')
import abm, db
reload(abm)
reload(db)

ec = [
        {
          "name": "Greystone Community Centre, Close Street",
          "label": "Community",
          "color": "r",
          "approx_coord": (-2.924540999999976,54.8896232)
        },
        {
          "name": "Trinity Church, Wigton Road",
          "label": "Church",
          "color": "b",
          "approx_coord": (-2.963238999999931,54.8870799)
        },
        {
          "name": "Sea Cadets, Nicholson Street",
          "label": "Cadets",
          "color": "y",
          "approx_coord": (-2.927017100000057,54.8834087)
        },        
        {
          "name": "Richard Rose Morton Academy, Wigton Road",
          "label": "Academy",
          "color": "g",
          "approx_coord": (-2.97123, 54.88005)
        },
    ]

# self = abm.Sim('flood','Carlisle')
p = db.Population((-2.960970689433074, 54.88321937312177, -2.8861928154524805, 54.91133135582124))

destin_type={'invdistprob':True,'nearest':False}

combinations = {
    ('invdistprob','ia'):'k5-idp',
    ('invdistprob','ff'):'ff-idp',
    ('nearest','ia'):'k5',
    ('nearest','ff'):'ff',
}

sim = abm.Sim('flood','Carlisle')
sim.n = int(round(sum(p.pop)))
sim.scenarios = combinations.values()

sim.h.destins = []
for k,v in enumerate(ec):
    ec[k]['nearest_node'] = sim.h.nearest_node(*v['approx_coord'])
    sim.h.destins.append(ec[k]['nearest_node'])

nodes_not_in_bbox = set(sim.h.G.nodes()).difference(nodes_in_bbox)

for u,v,d in sim.h.G.edges_iter(nodes_not_in_bbox,data=True):
    d['pop_dist'] = 0

# sim.fresh=True
sim.run_old(rerun=True)

# Events for each agent
E = {}
# Time for each agent
T = {}
# Exit for each agent
X = {}
# Viewer for each dt,scenario
V = {}

for k,v in combinations.iteritems():
    sim.init_scenario(v)
    sim.load_events()
    E[k] = sim.events
    X[k]=[e[-1][0] for e in E[k]]
    T[k]=[e[-1][1] for e in E[k]]
    V[k]=abm.Viewer(sim,bbox=bbox.buffer(0.02).bounds,video=False,percent=1)

plt.close('all')
for dt,scenario in combinations:
    for which in ['speed','time']:
        V[(dt,scenario)].path(which=which)
        plt.plot(*bbox.exterior.xy,c='g')   
        plt.axis('equal')
        plt.savefig('flood/figs/abm-{}-{}-{}.pdf'.format(dt,which,scenario),bbox_inches='tight')

plt.close('all')
sim.h.fig_highway(show_destins=False)
fig = plt.gcf()
fig.set_size_inches(12,8,forward=True)
for v in ec:
    plt.scatter(*sim.h.G.node[v['nearest_node']],s=200,c=v['color'],alpha=0.5,marker='o',label=v['label'])
plt.axis('equal')
plt.plot(*bbox.exterior.xy,c='g')
# refer to any viewer to get the boundary coordinates
this = V[(dt,scenario)]
gca().set_xlim(this.l,this.r)
gca().set_ylim(this.b,this.t)
plt.legend(scatterpoints=1)
plt.xlabel('Longitude',fontsize=15)
plt.ylabel('Latitude',fontsize=15)
plt.savefig('flood/figs/abm-evacuation-points.pdf',bbox_inches='tight')


"""Figure of population distribution"""
# The factor normalises population distribution inside the bounding box
factor = np.sum([d['pop_dist'] for u,v,d in sim.h.G.edges(data=True) if d['pop_dist'] > 0])
sim.h.fig_pop_dist(pop=sim.n/factor)
fig = plt.gcf()
fig.set_size_inches(10,14,forward=True)
plt.subplot(211)
for v in ec:
    plt.scatter(*sim.h.G.node[v['nearest_node']],s=200,c=v['color'],alpha=0.5,marker='o',label=v['label'])
plt.axis('equal')
plt.plot(*bbox.exterior.xy,c='g')
this = V[(dt,scenario)]
gca().set_xlim(this.l,this.r)
gca().set_ylim(this.b,this.t)
plt.legend(scatterpoints=1)
plt.savefig('flood/figs/abm-people-per-metre.pdf',bbox_inches='tight')

"""For the whole of carlisle"""
dt_name = {'invdistprob':'Inverse distance','nearest':'Nearest'}
scenario_name = {'ia':'Simulation','ff':'Free-flow'}

plt.figure(figsize=(10,8))
for dt in destin_type:
    for s in ['ff','ia']:
        plt.hist(T[(dt,s)],bins=100,cumulative=True,histtype='step',normed=True,label='{} ({})'.format(dt_name[dt],scenario_name[s]))
plt.legend(fontsize=15,loc='lower right')
plt.xlabel('Time [minutes]',fontsize=15)
plt.ylabel('Cumulative fraction of agents',fontsize=15)
plt.ylim(0,1.1)
plt.savefig('flood/figs/abm-hist-time.pdf',bbox_inches='tight')

"""For each catchment area"""

plt.close('all')
plt.figure(figsize=(12,12))
T_destin = {}
count = 0
for s in ['ff','ia']:
    for dt in destin_type:    
        plt.subplot(221+count)        
        count += 1
        plt.title('{} ({})'.format(dt_name[dt],scenario_name[s]))
        for x in sim.h.destins:
            T_destin[(dt,s,x)] = []
        for t,x in zip(T[(dt,s)], X[(dt,s)]):
            T_destin[(dt,s,x)].append(t)
        for v in ec:
            try:
                plt.hist(T_destin[(dt,s,v['nearest_node'])],color=v['color'],bins=100,cumulative=True,histtype='step',normed=True,label=v['label'])
            except KeyError:
                pass
        plt.xlabel('Time [minutes]',fontsize=15)
        plt.ylabel('Cumulative fraction of agents',fontsize=15)
        plt.legend(loc='lower right',fontsize=15)        
        plt.ylim(0,1.1)

plt.savefig('flood/figs/abm-hist-time-catchment.pdf',bbox_inches='tight')

from collections import OrderedDict
temp=OrderedDict()
for dt in destin_type: 
    for s in ['ff','ia']:
        temp[(dt_name[dt],scenario_name[s],'Overall')] = pandas.Series(T[(dt,s)]).describe(percentiles=[0.5,0.9])
        for v in ec:
            try:
                temp[(dt_name[dt],scenario_name[s],v['label'])] = pandas.Series(T_destin[(dt,s,v['nearest_node'])]).describe(percentiles=[0.5,0.9])
            except KeyError:
                pass

t1=pandas.DataFrame(temp).T
out1 = t1.applymap(lambda x: '{:0.0f}'.format(x))
out1.to_latex('flood/sim.tex')

ratios = t1.reorder_levels([1,0,2]).loc['Simulation']/t1.reorder_levels([1,0,2]).loc['Free-flow']-1
out2 = ratios.applymap(lambda x: '{:0.0f}%'.format(x*100))
out2.to_latex('flood/ratios.tex')

"""Look at the proportional allocation of people to exits"""
df=pandas.DataFrame()
df['Inverse distance']=t1['count']['Inverse distance']['Simulation']
df['Nearest']=t1['count']['Nearest']['Simulation']
df = df[['Nearest','Inverse distance']]
df.sort_values(by='Nearest',ascending=False,inplace=True)
df=df/df.sum()*100
df['Difference'] = df['Nearest'] - df['Inverse distance']

"""Work out the gini index of allocation of agents to exits
from http://planspace.org/2013/06/21/how-to-calculate-gini-coefficient-from-raw-data-in-python/"""

def gini(list_of_values):
    sorted_list = sorted(list_of_values)
    height, area = 0, 0
    for value in sorted_list:
        height += value
        area += height - value / 2.
    fair_area = height * len(list_of_values) / 2.
    return (fair_area - area) / fair_area

for k,v in X.iteritems():
    print k,gini([v.count(x) for x in sim.h.destins])

