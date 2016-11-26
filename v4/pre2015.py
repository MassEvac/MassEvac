from core import db,abm
import pandas
import shelve
import numpy as np
import random
import gzip
import pickle
import matplotlib.pyplot as plt
reload(db),reload(abm)

scenarios = ['k7', 'k6', 'k5', 'k7-idp', 'k6-idp', 'k5-idp']
allocation = ['Nearest', 'Nearest', 'Nearest', 'Inverse Distance', 'Inverse Distance', 'Inverse Distance']

scenario_name = {'k5-idp':'Inverse distance','k5':'Nearest'}

sim = 'bristol25'

# ---------------------------------------------------------
# First, produce the fundamental diagram

fd = abm.FundamentalDiagram(speedup=1,k_lim=7,k_vmin=5.0)
fd.figure(metrics=True)

# ---------------------------------------------------------
# Pretty label
pl = {
    'N':'N',
    'W':'W',
    'Dmax':'D^{max}',
    'Ds':'D^{\sigma}',
    'Dm':'\overline{D}',
    'D50':'D^{50\%}',
    'D90':'D^{90\%}',
    'Tmax':'T^{max}',
    'Tmaxf':'T^{max}_f',
    'Ts':'T^{\sigma}',
    'Tm':'\overline{T}',
    'T90':'T^{90\%}',
    'T90f':'T^{90\%}_f',
    'T90f1i':'T^{90\%}_{f1}',
    'T90f2i':'T^{90\%}_{f2}',
    'T90b':'T^{90\%}_b',
    'Qmax':'Q^{max}',
    'Qs':'Q^{\sigma}',
    'Qm':'\overline{Q}',
    'Qmf':'\overline{Q}_f',
    'Qmf1':'\overline{Q}_{f1}',
    'Qmf2':'\overline{Q}_{f2}',
    'Qmb':'\overline{Q}_b',
    'Q50':'Q^{50\%}',
    'Q50f':'Q^{50\%}_f',
    'Q50f1':'Q^{50\%}_{f1}',
    'Q50f2':'Q^{50\%}_{f2}',
    'Q50b':'Q^{50\%}_b',
    'Q90':'Q^{90\%}',
    'Q90f':'Q^{90\%}_f',
    'Q90f1':'Q^{90\%}_{f1}',
    'Q90f2':'Q^{90\%}_{f2}',
    'Q90b':'Q^{90\%}_b',
    'Qp':'Q_c',     
     }

unit = {
    'N':'[ped]',
    'W':'[m]',
    'Dmax':'[m]',
    'Ds':'[m]',
    'Dm':'[m]',
    'D50':'[m]',
    'D90':'[m]',
    'Tmax':'[s]',
    'Tmaxf':'[s]',
    'Ts':'[s]',
    'Tm':'[s]',
    'T90':'[s]',
    'T90f':'[s]',
    'T90f1i':'[s]',
    'T90f2i':'[s]',
    'T90b':'[s]',
    'Qmax':'[ped/(ms)]',
    'Qs':'[ped/(ms)]',
    'Qm':'[ped/(ms)]',
    'Qmf':'[ped/(ms)]',
    'Qmf1':'[ped/(ms)]',
    'Qmf2':'[ped/(ms)]',
    'Qmb':'[ped/(ms)]',
    'Q50':'[ped/(ms)]',
    'Q50f':'[ped/(ms)]',
    'Q50f1':'[ped/(ms)]',
    'Q50f2':'[ped/(ms)]',
    'Q50b':'[ped/(ms)]',
    'Q90':'[ped/(ms)]',
    'Q90f':'[ped/(ms)]',
    'Q90f1':'[ped/(ms)]',
    'Q90f2':'[ped/(ms)]',
    'Q90b':'[ped/(ms)]',
    'Qp':'[ped/(ms)]',
     }


places = sorted(abm.Places(sim).names)

pop = {}
for place in places:
    # Load poopulation
    pop[place] = db.Population(place,table='pop_gpwv4_2015')
total_pop = [sum(pop[place].pop) for place in places]
sorted_places,sorted_pop = zip(*sorted(zip(places,total_pop),key=lambda tot: tot[1],reverse=True))

W_dict = {}
for place in places:
    W_dict[place] = pandas.DataFrame([(k,sum(v.values())) for k,v in shelve.open('metadata/bristol25/{}/common.shelve'.format(place))['destin_width'].iteritems()],columns=['destin','W']).set_index('destin')
W = pandas.concat(W_dict)['W']

DF = {}
for scenario in scenarios:
    DF_dict = {}
    print 'loading', scenario
    for place in places:
        hdf = pandas.HDFStore('metadata/bristol25/{}/agents.hdf'.format(place)) 
        df = hdf.get(scenario)
        hdf.close()
        DF_dict[place] = df 
    DF[scenario] = pandas.concat(DF_dict)

for scenario in scenarios:  
    DF[scenario].set_index('destin',append=True,inplace=True)


scenario = 'k5'

SDF = {}
Q = {}
N = {}
T90 = {}
D90 = {}
T90f = {}
Qmf = {}
Qp = {}
Qr = {}
Tr = {}
Qf = {}
for scenario in scenarios:
    print 'loading', scenario
    grouped = DF[scenario].groupby(level=[0,2])
    T90[scenario] = grouped['time2x'].quantile(0.9)
    D90[scenario] = grouped['dist2x'].quantile(0.9)
    T90f[scenario] = D90[scenario]/60/1.34
    N[scenario] = grouped['dist2x'].count()
    Q[scenario] = grouped['time2x'].agg(lambda v: tuple(np.histogram(v,bins=range(int(max(v)+2)))[0]))
    Qmf[scenario] = Q[scenario].copy(deep=True)
    Qp[scenario] = Q[scenario].copy(deep=True)
    Qmf[scenario][:] = None
    Qp[scenario][:] = None

    for place,destin in Q[scenario].index:
        Qmf[scenario].loc[place,destin] = np.mean(np.array(Q[scenario].loc[place,destin])[:int(T90f[scenario].loc[place,destin])+1])/W.loc[place,destin]/60
        Qp[scenario].loc[place,destin] = N[scenario].loc[place,destin]/T90f[scenario].loc[place,destin]/W.loc[place,destin]/60
    Tr[scenario] = T90[scenario]/T90f[scenario]
    Qr[scenario] = Qmf[scenario]/Qp[scenario]

for scenario in scenarios:
    print 'loading', scenario
    df = DF[scenario]['dist2x'].apply(lambda x: x/60/1.34)
    grouped = df.groupby(level=[0,2])
    Qf[scenario] = grouped.agg(lambda v: tuple(np.histogram(v,bins=range(int(max(v)+1)))[0]))

def gini(list_of_values):
    sorted_list = sorted(list_of_values)
    height, area = 0, 0
    for value in sorted_list:
        height += value
        area += height - value / 2.
    fair_area = height * len(list_of_values) / 2.
    return (fair_area - area) / fair_area

cityDF = {}

for scenario in ['k5','k5-idp']:
    cityDF[scenario] = pandas.DataFrame({
                'T90f': DF[scenario]['dist2x'].groupby(level=0).quantile(q=0.9)/1.34/60,
                'T90': DF[scenario]['time2x'].groupby(level=0).quantile(q=0.9),
            })

    
    cityDF[scenario]['T90/T90f'] = (cityDF[scenario]['T90']/cityDF[scenario]['T90f'])
    cityDF[scenario]['rank'] = cityDF[scenario]['T90/T90f'].rank(ascending=False)
    cityDF[scenario]['gini'] = N[scenario].groupby(level=0).apply(gini).round(2).loc[list(sorted_places)]
    cityDF[scenario]['T90/T90f'] = cityDF[scenario]['T90/T90f'].round(2)
    cityDF[scenario]['T90'] = cityDF[scenario]['T90'].round()
    cityDF[scenario]['T90f'] = cityDF[scenario]['T90f'].round()

cDF = pandas.concat(cityDF)[['T90','T90f','T90/T90f','rank','gini']].unstack(level=0).T.swaplevel(1,0).sort_index(level=0).T.loc[list(sorted_places)]
cDF['rank'] = range(1,51)
cDF=cDF.set_index('rank',append=True).swaplevel()
cDF.to_latex('pre2015/cityDF.tex')

norm_pl = '{}/{}'.format(pl['T90'],pl['T90f'])

# cityETE hist
plt.figure(figsize=(10,12))
i = 0
for label,pretty_label in zip(['T90','T90f','T90/T90f'],[pl['T90'],pl['T90f'],norm_pl]):
    plt.subplot(411+i)
    i += 1
    max_val = max(cDF[('k5',label)].max(),cDF[('k5-idp',label)].max())
    for scenario in ['k5','k5-idp']:
        # Number of bins is proportional to the largest value
        plt.hist(cDF[(scenario,label)],bins=int(30*cDF[(scenario,label)].max()/max_val),label=scenario_name[scenario],histtype='step',cumulative=False)
    plt.xlabel('${}$'.format(pretty_label),fontsize=15)
    plt.ylabel('Number of cities',fontsize=15)
    plt.legend(loc='upper right')
# Gini index
plt.subplot(411+i)
for scenario,color in zip(['k5','k5-idp'],['blue','red']):
    plt.scatter(cDF[(scenario,'gini')],cDF[(scenario,'T90/T90f')],label=scenario_name[scenario],marker='x',c=color)
plt.xlabel('Gini index of cities to indicate inequality of agents allocated per exit',fontsize=15)    
plt.ylabel('${}$'.format(norm_pl),fontsize=15)
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('pre2015/cityETEhist.pdf',bbox_inches='tight')

from scipy import stats
print stats.pearsonr(cDF['k5','T90/T90f'],sorted_pop)
print stats.pearsonr(cDF['k5-idp','T90/T90f'],sorted_pop)

print stats.spearmanr(cDF['k5','T90/T90f'],sorted_pop)
print stats.spearmanr(cDF['k5-idp','T90/T90f'],sorted_pop)


import networkx as nx
s = abm.Sim('bristol25','City of Bristol')
s.h.init_route()
pos = nx.get_node_attributes(s.h.G,'pos')

destins = s.h.destins

colors =   ["#0070ad",
            "#ffcf0a",
            "#dc54ff",
            "#679200",
            "#fb00b6",
            "#01fde7",
            "#ff075e",
            "#018862",
            "#9574ff",
            "#ff6322",
            "#001919",
            "#ffd3bc",
            "#2f1000",
            "#d0d5ff",
            "#983600",
            "#6a002e",
            "#8d0020"]

destin_color = dict(zip(destins,colors))

sample = [48154, 21817, 9957]

attr = dict(zip(sample, [{'label':'CA01','zorder':2},
                            {'label':'CA02','zorder':1},
                            {'label':'CA03','zorder':3},

                        ]))
"""FIG 2A"""
# Catchment areas
plt.close('all')
plt.figure(figsize=(14,6))
for i,scenario in enumerate(['k5','k5-idp']):

    ie = DF[scenario].loc['City of Bristol']['initedge']

    agents_to_show = list(np.random.choice(range(len(ie)),len(ie)/50))

    # plt.figure()
    # for x,edge in ie.loc[agents_to_show].values:
    #     nxedges = nx.draw_networkx_edges(h.G,pos=pos,arrows=False,edgelist=[edge],edge_color=destin_color[x],width=1,alpha=0.1)

    plt.subplot(121+i)
    for k,v in ie.loc[agents_to_show].iteritems():
        _,x = k
        u,_ = v
        plt.scatter(*pos[u],c=destin_color[x],marker='.',alpha=0.5,linewidth=0)
        # nxedges = nx.draw_networkx_edges(h.G,pos=pos,arrows=False,edgelist=[edge],edge_color=destin_color[x],width=1,alpha=0.1)

    for x in destins:
        plt.scatter(*pos[x],s=200,c=destin_color[x],alpha=0.5,marker='o')

    for x,v in attr.iteritems():
        plt.annotate(
            v['label'],
            xy = pos[x], xytext = (10,10),
            textcoords = 'offset points', ha = 'left', va = 'bottom',
            bbox = dict(boxstyle = 'round,pad=0.5', fc = destin_color[x], alpha = 0.5))


    # for x in destins:
    #     plt.annotate(
    #         x,
    #         xy = pos[x], xytext = (10,10),
    #         textcoords = 'offset points', ha = 'left', va = 'bottom',
    #         bbox = dict(boxstyle = 'round,pad=0.5', fc = destin_color[x], alpha = 0.5))

    plt.title(scenario_name[scenario])
    plt.xlabel('Longitude',fontsize=15)
    plt.ylabel('Latitude',fontsize=15)
    plt.axis('equal')

plt.savefig('pre2015/Bristol-CA.pdf',bbox_inches='tight')  

"""FIG 2B"""
# Number of agents for a sample of catchement areas in Bristol

plt.figure(figsize=(14,6))
for i,scenario in enumerate(['k5','k5-idp']):
    dx = DF[scenario]['dist2x'].loc['City of Bristol'].groupby(level=1).agg(lambda x: tuple(x))

    xlim = 0
    ylim = 0

    plt.subplot(121+i)

    import matplotlib.pyplot as plt
    for x in sample:
        try:
            data = dx.loc[x]
            h,b,p = plt.hist(dx.loc[x],bins=int(np.max(data)/100),alpha=0.5,label='$\mathrm{{{0}}}$'.format(attr[x]['label']),facecolor=destin_color[x], histtype='stepfilled',zorder=attr[x]['zorder'])
            ylim = max(ylim,h.max())
            xlim = max(xlim,b.max())
        except KeyError:
            pass
        
    plt.xlim(0,xlim+100);plt.ylim(0,ylim+100)

    plt.title(scenario_name[scenario])
    plt.xlabel('$\mathrm{D \ [m]}$',fontsize=15)
    plt.ylabel('$\mathrm{N \ [ped]}$',fontsize=15)
    plt.legend(framealpha=0.5)
    #plt.grid()

plt.savefig('pre2015/Bristol-DvN.pdf',bbox_inches='tight')  


"""FIG 3A"""
plt.figure(figsize=(14,6))
for i,scenario in enumerate(['k5','k5-idp']):
    CA_per_city = N[scenario].groupby(level=0).count()
    print scenario
    print 'std',CA_per_city.std()
    print 'mean',CA_per_city.mean()
    print 'sum',CA_per_city.sum()

    plt.subplot(121+i)    
    plt.hist(CA_per_city,bins=20)
    plt.title(scenario_name[scenario])    
    plt.xlabel('Number of CA',fontsize=15)
    plt.ylabel('Number of cities',fontsize=15)
plt.savefig('pre2015/hist-num-CA.pdf',bbox_inches='tight')

"""FIG 3B and 3C"""
# Histogram of population and D90

import math
def lognorm_hist(dictionary,scenarios,label):
    plt.figure(figsize=(14,6))
    for i,scenario in enumerate(scenarios):
        data = dictionary[scenario]
        Nlog = np.log10(data)
        lmin = min(Nlog)-1
        lmax = max(Nlog)

        print label,scenario
        print 'mean',np.mean(data)
        print 'std',np.std(data)    
        print 'log of mean',np.log10(np.mean(data))
        print 'log of min',np.log10(np.min(data))
        print 'log of max',np.log10(np.max(data))    
            
        bins=np.logspace(lmin, lmax, 50)
        plt.subplot(121 + i)
        plt.title(scenario_name[scenario])
        plt.hist(data,bins=bins)
        plt.xlabel('$\mathrm{{{0} \ {1}}}$'.format(pl[label],unit[label]),fontsize=15)
        plt.ylabel('Number of CA',fontsize=15)
        plt.xscale('log')
    plt.savefig('pre2015/lognorm-hist-{0}.pdf'.format(label),bbox_inches='tight')

# 3B
lognorm_hist(D90,['k5','k5-idp'],'D90')
# 3C
lognorm_hist(N,['k5','k5-idp'],'N')

"""FIG 4"""
# Example of catchment area free flow and simulated flow

key = ('City of Bristol',sample[1])

plt.close('all')

fig = plt.figure(figsize=(12,6))

qf = np.array(Qf[scenario].loc[key])/60./W.loc[key]
q = np.array(Q[scenario].loc[key])/60./W.loc[key]
plt.plot(qf,label='$\mathrm{Free-flow}$')
plt.plot(q,label='$\mathrm{Simulation}$')

Qmax = 1.22

plt.axhline(Qmax,linestyle=':',color='y',label='$\mathrm{ Q_{max} = %0.2f \ [ped/(ms)] }$'%Qmax)
plt.axhline(Qp[scenario].loc[key],linestyle='-.',color='r',label='$\mathrm{ Q_c = %0.2f \ [ped/(ms)] }$'%Qp[scenario].loc[key])
plt.axhline(Qmf[scenario].loc[key],linestyle='--',color='m',label='$\mathrm{ \overline{Q}_{f} = %0.2f \ [ped/(ms)] }$'%Qmf[scenario].loc[key])

plt.axvline(T90f[scenario].loc[key],linestyle='-.',color='c',label='$\mathrm{ T^{90\%%}_f = %0d \ [s] }$'%(T90f[scenario].loc[key]*60))
plt.axvline(T90[scenario].loc[key],linestyle='--',color='g',label='$\mathrm{ T^{90\%%} = %0d \ [s] }$'%(T90[scenario].loc[key]*60))

plt.xlim(0,len(q))

xticks = range(0,len(q),200)
xticklabs = np.array(xticks)*60
plt.xticks(xticks,xticklabs)

handles, labels = plt.gca().get_legend_handles_labels()
lgd = plt.legend(handles, labels, loc='upper right',fontsize=15)

plt.xlabel('$\mathrm{T \ [s]}$',fontsize=15)
plt.ylabel('$\mathrm{Q \ [ped/(ms)]}$',fontsize=15)

plt.savefig('pre2015/single-CA-T-Q.pdf',bbox_inches='tight')

"""FIG 5"""
# Tracks

for scenario in ['k5','k5-idp']:
    s.init_scenario(scenario)
    s.load_events()

    sum_velocity = {k:{} for k in s.h.destins}
    sum_agent_count = {k:{} for k in s.h.destins}

    t90 = [e[-1][1] for e in s.events]

    max_vals = DF[scenario].loc['City of Bristol'][['dist2x','time2x']].groupby(level=[1]).max()

    for x,v in max_vals.iterrows():
        d2x,t2x = v
        sum_velocity[x] = np.zeros((int(d2x)+1,int(t2x)+1))
        sum_agent_count[x] = np.zeros((int(d2x)+1,int(t2x)+1))

    for event in s.events:
        destin, _ = event[-1]
        u, tu = event[0]
        for v, tv in event[1:]:
            distance = s.h.G[u][v]['distance']
            du = int(distance + s.h.route_length[destin][v])
            dv = int(s.h.route_length[destin][v])
            velocity = distance / (tv-tu) / 60
            # dist from v < dist from u
            for d,t in zip(np.linspace(du,dv,du-dv+1,dtype=int),np.linspace(tu,tv,du-dv+1,dtype=int)):
                if t >= 0:
                    sum_velocity[destin][d,t] += velocity
                    sum_agent_count[destin][d,t] += 1
            u = v
            tu = tv

    for x in sample:
        mean_velocity = sum_velocity[x]/sum_agent_count[x]
        with gzip.open('pre2015/{}-{}.mv.gz'.format(scenario,x),'w') as f:
            pickle.dump(mean_velocity,f)

# Produce figures
for scenario in ['k5','k5-idp']:
    s.init_scenario(scenario)
    s.load_events()    
    for x in sample:
        with gzip.open('pre2015/{}-{}.mv.gz'.format(scenario,x),'r') as f:
            mean_velocity = pickle.load(f)

        dD,dT = mean_velocity.shape
        aspect = float(dT)/dD*0.4

        agent = 0
        while True:
            event = s.events[agent]
            if event[-1][0] == x and event[-1][1] > dT/2:
                break
            agent += 1

        plt.close('all')
        plt.figure(figsize=(10,5))
        plt.imshow(mean_velocity,interpolation='none',cmap='Spectral',aspect=aspect,origin='lower')
        plt.xlabel('$\mathrm{T \ [s]}$',fontsize=15)
        plt.ylabel('$\mathrm{D \ [m]}$',fontsize=15)
        plt.xlim(0,None)
        plt.ylim(0,None)
        max_dist,max_time = mean_velocity.shape
        xticks = range(0,max_time,200)
        xticklabs = np.array(xticks)*60
        plt.xticks(xticks,xticklabs)
        dax,tax = zip(*[(s.h.route_length[x][n],t) for n,t in event])
        plt.plot(tax,dax,label='Randomly picked agent trajectory',linewidth=2,c='black',alpha=0.7)
        cbar = plt.colorbar(orientation='horizontal')
        cbar.set_label('$\mathrm{v \ [m/s]}$',fontsize=15)
        plt.legend()
        plt.savefig('pre2015/mv-{}-{}.pdf'.format(scenario,x),bbox_inches='tight')

"""FIG 6"""
# def CI_fig(CI,nbins = 1000):
CI = 68

'''civ needs to be a value between 0-100'''

upper = 50 + CI/2
lower = 50 - CI/2

scenario = 'k5-idp'

# Looking at all cities here
# Flow vs time with 50%-CI/2, 50%, 50%+CI  marked
all_x=[]
all_y=[]

# Normalised flow vs normalised time with 50%-CI/2, 50%, 50%+CI/2  marked
all_norm_x=[]
all_norm_y=[]

for k,y in Q[scenario].iteritems():
    # Only proceed if there are agents present in this destination
    x = np.array(range(len(y)))
    y = np.array(y)/W.loc[k]/60.
    all_x.extend(x)
    all_y.extend(y)
    all_norm_x.extend(x/T90f[scenario].loc[k])
    all_norm_y.extend(y/Qp[scenario].loc[k])

all_x = np.array(all_x)*60.
all_y = np.array(all_y)
all_norm_x = np.array(all_norm_x)
all_norm_y = np.array(all_norm_y)

# Regular figures
nbins = {
            'k5':5000,
            'k5-idp':1000,
        }

H,xedges=np.histogram(all_x,bins=nbins[scenario])
digitized = np.digitize(all_x, xedges)
x_range = np.array(range(len(xedges)))+1
x_len = np.array([len(all_x[digitized == i]) for i in x_range])
x_mean = np.array([all_x[digitized == i].mean() for i in x_range])
y_lower = np.array([np.percentile(all_y[digitized == i],lower) for i in x_range])
y_median = np.array([np.percentile(all_y[digitized == i],50) for i in x_range])
y_upper = np.array([np.percentile(all_y[digitized == i],upper) for i in x_range])

from matplotlib import gridspec

myxticks = {
    'k5': [0,100000,200000,300000],
    'k5-idp': [0,100000,200000],
}

fig = plt.figure(figsize=(10,8))
gs = gridspec.GridSpec(2, 2,height_ratios=[2,1])
# LHS
ax0 = plt.subplot(gs[0])
ax0.plot(x_mean,y_median,label='$\mathrm{Q \ [ped/(ms)]}$')
ax0.fill_between(x_mean,y_upper,y_lower,facecolor='gray',alpha=0.5)
ax0.set_ylabel('$\mathrm{Q \ [ped/(ms)]}$',fontsize=15)
ax0.set_xticks(myxticks[scenario])
ax1 = plt.subplot(gs[2])
ax1.plot(x_mean,x_len,label='$\mathrm{No.\ of\ Data\ Points}$',color='g',alpha=0.5)
ax1.set_xlabel('$\mathrm{T \ [s]}$',fontsize=15)
ax1.set_ylabel('$\mathrm{No.\ of\ Data\ Points}$',fontsize=15)
ax1.set_yscale('log')
ax1.set_xticks(myxticks[scenario])
# RHS

xlim = {'k5':10000,'k5-idp':20000}

ax3 = plt.subplot(gs[1])
ax3.plot(x_mean,y_median)
ax3.fill_between(x_mean,y_upper,y_lower,facecolor='gray',alpha=0.5)
ax3.set_xlim(None,xlim[scenario])   
ax4 = plt.subplot(gs[3])
ax4.plot(x_mean,x_len,color='g',alpha=0.5)
ax4.set_xlabel('$\mathrm{T \ [s]}$',fontsize=15)
ax4.set_yscale('log',fontsize=15)
ax4.set_xlim(None,xlim[scenario])
ax4.axvline(1,c='r',linestyle='-.',label='$\mathrm{{T/{0}}} = 1$'.format(pl['T90f']))    
plt.savefig('pre2015/{}-{}-all-T-Q.pdf'.format(CI,scenario),bbox_inches='tight')

# Regular figures
nbins = {
            'k5':5000,
            'k5-idp':1000,
        }

# Normalised figures
H,xedges=np.histogram(all_norm_x,bins=nbins[scenario])
digitized = np.digitize(all_norm_x, xedges)
x_range = np.array(range(len(xedges)))+1
x_len = np.array([len(all_norm_x[digitized == i]) for i in x_range])
x_mean = np.array([all_norm_x[digitized == i].mean() for i in x_range])
y_lower = np.array([np.percentile(all_norm_y[digitized == i],lower) for i in x_range])
y_median = np.array([np.percentile(all_norm_y[digitized == i],50) for i in x_range])
y_upper = np.array([np.percentile(all_norm_y[digitized == i],upper) for i in x_range])    
#***
fig = plt.figure(figsize=(10,8))
gs = gridspec.GridSpec(2, 2, height_ratios=[2,1], width_ratios=[1,1])
# LHS
ax0 = plt.subplot(gs[0])
ax0.plot(x_mean,y_median)
ax0.fill_between(x_mean,y_upper,y_lower,facecolor='gray',alpha=0.5)
ax0.set_ylabel('$\mathrm{{Q/{0}}}$'.format(pl['Qp']),fontsize=15)
ax2 = plt.subplot(gs[2])
ax2.plot(x_mean,x_len,label='$\mathrm{No.\ of\ Data\ Points}$',color='g',alpha=0.5)
ax2.set_xlabel('$\mathrm{{T/{0}}}$'.format(pl['T90f']),fontsize=15)
ax2.set_ylabel('$\mathrm{No.\ of\ Data\ Points}$',fontsize=15)
ax2.set_yscale('log',fontsize=15)
# RHS
ax3 = plt.subplot(gs[1])
ax3.plot(x_mean,y_median)
ax3.fill_between(x_mean,y_upper,y_lower,facecolor='gray',alpha=0.5)
ax3.set_xlim(None,2)   
ax3.axvline(1,c='r',linestyle='-.',label='$\mathrm{T/T_f^{90\%} = 1}$')
ax4 = plt.subplot(gs[3])
ax4.plot(x_mean,x_len,color='g',alpha=0.5)
ax4.set_xlabel('$\mathrm{{T/{0}}}$'.format(pl['T90f']),fontsize=15)
ax4.set_yscale('log',fontsize=15)
ax4.set_xlim(None,2)
ax4.axvline(1,c='r',linestyle='-.',label='$\mathrm{{T/{0}}} = 1$'.format(pl['T90f']))    
plt.savefig('pre2015/{}-{}-all-norm-T-Q.pdf'.format(CI,scenario),bbox_inches='tight')

"""FIG 7"""
# Qc vs Qmf
import random

normdict = {'D':'D90','T':'T90f','Q':'Qp'}

nl = {}
npl = {}

for l in pl.keys():
    try:
        # label to normalise with
        nw = normdict[l[0]]
        # matrix to normalise with
        nl[l] = '{0}/{1}'.format(l,nw)
        npl[l] = '{0}/{1}'.format(pl[l],pl[nw])
    except KeyError:
        nl[l] = l
        npl[l] = pl[l]
        print 'Could not normalise for',l

"""FIG 7A"""
theta = {}
gamma = {}

for scenario in scenarios:
    l1 = 'Qp'
    l2 = 'Qmf'

    f=shelve.open('pre2015/train-test-{}'.format(scenario))
    if not f.keys():
        noofCA = len(Qp[scenario])
        dataindex = range(noofCA)
        random.shuffle(dataindex)
        train = dataindex[:int(noofCA/2)]
        test = dataindex[int(noofCA/2):]
        f['train']=train
        f['test']=test
    train = f['train']
    f.close()

    x = list(Qp[scenario].iloc[train])
    y = list(Qmf[scenario].iloc[train])

    logx = np.log10(x)
    logy = np.log10(y)

    tol_order = 1

    f = np.polyfit(logx,logy,tol_order)
    c = np.corrcoef(logx,logy)[0][1]**2

    x_range = np.linspace(min(x),max(x),100)

    theta[scenario],gamma[scenario] = f
    gamma[scenario] = 10**gamma[scenario]

    # For fitting y = gamma x^theta
    # take the logarithm of both side gives log y = theta log x + log gamma.
    # So just fit log y against x.

    Qmax = max(s.fd.q)

    plt.figure()
    plt.scatter(x,y,color='g',label='$\mathrm{r^2 = %0.2f}$'%c)
    plt.xscale('log')
    plt.yscale('log')
    # plt.plot(10**x_range,10**np.polyval(f,x_range),'b-',label='$\mathrm{{ {1} = \gamma ({0})^{{\\theta}} }}$'.format(pl[l1],pl[l2])) # A red solid line
    plt.plot(x_range,gamma[scenario]*x_range**theta[scenario],'b-',label='$\mathrm{{ {1} = \gamma ({0})^{{\\theta}} }}$'.format(pl[l1],pl[l2])) # A red solid line
    plt.plot(x_range,x_range,'r--',label='$\mathrm{{{1} = {0}}}$'.format(pl[l1],pl[l2]))
    plt.axvline(Qmax,c='y',linestyle='-.',label='$\mathrm{Q_c = Q_{max}}$')    
    plt.legend(loc='upper left') # make a legend in the best location
    plt.xlabel('$\mathrm{{{0} \ {1}}}$'.format(pl[l1],unit[l1]),fontsize=15)
    plt.ylabel('$\mathrm{{{0} \ {1}}}$'.format(pl[l2],unit[l2]),fontsize=15) # labels again
    plt.savefig('pre2015/{}-{}-{}.pdf'.format(scenario,l1,l2),bbox_inches='tight')



"""FIG 7B"""

alpha = {}
beta = {}

for scenario in scenarios:
    # Tr vs Qr
    l1 = 'Qmf'
    l2 = 'T90'

    f=shelve.open('pre2015/train-test-{}'.format(scenario))
    train = f['train']
    f.close()

    x = list(Qr[scenario].iloc[train])
    y = list(Tr[scenario].iloc[train])
    logx = np.log10(x)
    logy = np.log10(y)

    tol_order = 1

    # from the graph above, using higher order than 3 does not offer significant advantage in minimising deviation
    f = np.polyfit(logx,logy,tol_order)
    c = np.corrcoef(logx,logy)[0][1]**2

    alpha[scenario],beta[scenario] = f
    beta[scenario] = 10**beta[scenario]

    # For fitting y = Ae^(Bx), 
    # take the logarithm of both side gives log y = Bx + log A.
    # So just fit log y against x.    

    x_range = np.linspace(min(x),max(x),100)

    plt.figure()
    plt.scatter(x,y,color='y',label='$\mathrm{r^2 = %0.2f}$'%c)
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(x_range,beta[scenario]*x_range**alpha[scenario],'b-',label='$\mathrm{{ {1} = \\beta ({0})^{{\\alpha}} }}$'.format(npl[l1],npl[l2])) # A red solid line
    plt.legend(loc='lower left') # make a legend in the best location
    plt.xlabel('$\mathrm{{{0}}}$'.format(npl[l1]),fontsize=15)
    plt.ylabel('$\mathrm{{{0}}}$'.format(npl[l2]),fontsize=15) # labels again
    plt.savefig('pre2015/{}-{}-{}.pdf'.format(scenario,nl[l1].replace('/',':'),nl[l2].replace('/',':')),bbox_inches='tight')

for scenario in scenarios:    
    print scenario, 'theta = ', theta[scenario], 'gamma = ',gamma[scenario]

for scenario in scenarios:
    print scenario, 'alpha = ', alpha[scenario], 'beta = ',beta[scenario]

"""FIG 8"""

phi = {}
omega = {}

zeta = {}
eta = {}

x = {}
y = {}
cl = {}
f = {}

for scenario in scenarios:

    # Calculate phi and omega
    omega[scenario] = beta[scenario]*gamma[scenario]**alpha[scenario]
    phi[scenario] = alpha[scenario]*(theta[scenario]-1)

    f=shelve.open('pre2015/train-test-{}'.format(scenario))
    test = f['test']
    f.close()

    x0 = list(T90f[scenario].iloc[test])
    x1 = list(Qp[scenario].iloc[test])
    x[scenario] = list(omega[scenario]*np.array(x0)*np.array(x1)**phi[scenario])
    y[scenario] = list(T90[scenario].iloc[test])
    logx = np.log10(x[scenario])
    logy = np.log10(y[scenario])
    f = np.polyfit(logx,logy,tol_order)
    zeta[scenario], eta[scenario] = f
    eta[scenario] = 10**eta[scenario]
    cl[scenario] = np.corrcoef(logx,logy)[0,1]**2

plt.figure(figsize=(12,10))
for i,scenario in enumerate(scenarios):
    plt.subplot(121+i/3)
    x_range = np.linspace(min(x[scenario]),max(x[scenario]))
    yoffset = np.array(10**(2*(2-i%3)))
    plt.scatter(x[scenario],yoffset*y[scenario],color='gray')
    plt.plot(x_range,yoffset*eta[scenario]*x_range**zeta[scenario],'-',label='$\mathrm{{k_{{lim}}}} = {:0.1f} \ \mathrm{{ped/m^2}},\ \zeta = {:0.2f},\ \eta = {:0.2f}$'.format(abm.settings[scenario]['k_lim'],zeta[scenario],eta[scenario])) # A red solid line
    # plt.plot(x_range,yoffset*eta[scenario]*x_range**zeta[scenario],'-',label='$\mathrm{{{0}_{{simulated}}}} = \\eta \mathrm{{{0}_{{calculated}}}}^\\zeta$'.format(pl[l2],abm.settings[scenario]['k_lim'])) # A red solid line
    if i%3 == 2:
        plt.plot(x_range,yoffset*x_range,'g--',label='$\mathrm{{{0}_{{simulated}}}} = \mathrm{{{0}_{{calculated}}}}$'.format(pl[l2])) # Diagonal
    else:
        plt.plot(x_range,yoffset*x_range,'g--') # Diagonal
    try:
        plt.title(scenario_name[scenario])
    except KeyError:
        pass
    plt.xscale('log')
    plt.yscale('log')
    plt.axis('equal')
    plt.ylim(10**-1,10**10)
    plt.xlim(10**-2,10**4)
    plt.legend(loc='upper left')
    plt.xlabel('$\mathrm{{{0}_{{calculated}}}}$'.format(pl[l2]),fontsize=15)
    plt.ylabel('$\mathrm{{{0}_{{simulated}}}}$'.format(pl[l2]),fontsize=15) # labels again
plt.savefig('pre2015/T90-T90.pdf'.format(scenario),bbox_inches='tight')

for scenario in scenarios:
    print scenario, 'omega = ',omega[scenario], 'phi=', phi[scenario], 'r^2'
for scenario in scenarios:
    print scenario, 'zeta = ',zeta[scenario], 'eta=', eta[scenario], 'r^2', cl[scenario]

# Print phi, omega, zeta, eta table
k_lim = {k:v['k_lim'] for k,v in abm.settings.iteritems()}

coefftab =  pandas.DataFrame({'allocation':dict(zip(scenarios,allocation)),'k_lim':k_lim,'phi':phi,'omega':omega,'zeta':zeta,'eta':eta},columns=['allocation','k_lim','phi','omega','zeta','eta'])
coefftab.set_index(['allocation','k_lim'],inplace=True)
coefftab.sort_index(ascending=[0,1],inplace=True)

coefftab.round(3).to_latex('pre2015/coefftab.tex')

T90c = {}
dT90 = {}

for scenario in scenarios:
    T90c[scenario] = omega[scenario]*T90f[scenario]*Qp[scenario]**phi[scenario]
    dT90[scenario] = T90[scenario] - T90c[scenario]

"""FIG 9"""

def load_json(metric,place,destin):
    print metric
    folder = folder_structure.format(sim, metric, place)
    print folder
    fname = file_structure.format(folder, destin)
    with gzip.open(fname, 'r') as infile:
        intermediate = json.load(infile)
        # Convert key to number or tuple before returning
        if metric == 'summary':
            return intermediate
        else:
            return {eval(key):value for key,value in intermediate.iteritems()}

folder_structure = 'metrics/{}/{}/{}'
file_structure = '{}/{}.json.gz'

scenario = 'k5'
lj = {}
for p,x in T90[scenario].index:
    v = load_json('summary',p,x)
    lj[(p,x)] = v
M = pandas.DataFrame(lj).applymap(lambda x: np.float(x)).T.loc[T90[scenario].index]
keys = M.columns
M['dT'] = (dT90[scenario]).apply(lambda x: np.float(x))
M['dTnorm'] = (dT90[scenario]/T90f[scenario]).apply(lambda x: np.float(x))
del M['min_sdsp']
del M['min_tdsp']
del M['num_attractcom']
del M['num_weakconcom']



"""dT90 norm histogram"""
plt.figure(figsize=(10,4))
# plt.hist(M['dTnorm'],bins=100,cumulative=True,histtype='step',normed=False)
plt.hist(M['dT'],bins=100,cumulative=False,histtype='step',normed=False)
plt.ylabel('Number of CAs',fontsize=15)
plt.xlabel('$\delta T^{90\%}$',fontsize=15)
plt.savefig('pre2015/hist-dT90.pdf'.format(scenario),bbox_inches='tight')

"""dT90 norm histogram"""
plt.figure(figsize=(10,4))
# plt.hist(M['dTnorm'],bins=100,cumulative=True,histtype='step',normed=False)
plt.hist(M['dTnorm'],bins=100,cumulative=False,histtype='step',normed=False)
plt.ylabel('Number of CAs',fontsize=15)
plt.xlabel('$\delta T^{90\%}_n$',fontsize=15)
plt.savefig('pre2015/hist-dT90-norm.pdf'.format(scenario),bbox_inches='tight')


corrnorm = pandas.DataFrame()
corrnorm['pearson'] = M.corr('pearson')['dTnorm'].loc[keys]
corrnorm['spearman'] = M.corr('spearman')['dTnorm'].loc[keys]
corrnorm.to_csv('pre2015/corr.csv')

all_keys = list(corrnorm.index)

keys = [   
        'deg_cen',
        'in_deg_cen',
        'out_deg_cen',
        'bet_cen',
        'bet_cen_exit',
        'load_cen_exit',
        'eivec_cen_exit',
        'close_cen_exit',
        'sq_clust'
        ]

def components(key):
    return [
            'mean_{}'.format(key),
            'mean_{}_mean_sdsp'.format(key),
            'mean_{}_10pc_sdsp'.format(key),
            'mean_{}_50pc_sdsp'.format(key),
            'mean_{}_90pc_sdsp'.format(key),
            'mean_{}_2000m'.format(key),
            'mean_{}_1000m'.format(key),
            'mean_{}_500m'.format(key),
            'mean_{}_250m'.format(key),
            'mean_{}_mean_tdsp'.format(key),
            'mean_{}_10pc_tdsp'.format(key),
            'mean_{}_50pc_tdsp'.format(key),
            'mean_{}_90pc_tdsp'.format(key),
            'mean_{}_8_tdsp'.format(key),
            'mean_{}_4_tdsp'.format(key),
            'mean_{}_2_tdsp'.format(key),
            'mean_{}_1_tdsp'.format(key)
        ]

labels = [
    'All',
    'Mean Spatial Distance',
    '10% Spatial Distance',
    '50% Spatial Distance',
    '90% Spatial Distance',
    '2000 metres',
    '1000 metres',
    '500 metres',
    '250 metres',
    'Mean Topological Distance',
    '10% Topological Distance',
    '50% Topological Distance',
    '90% Topological Distance',
    '8 Topological Distance',
    '4 Topological Distance',
    '2 Topological Distance',
    '1 Topological Distance'
]

remaining = all_keys[:]
v = {}
for k in keys:
    this = components(k)
    v[k] = corrnorm.loc[this]
    v[k].index = labels
    for t in this:
        remaining.remove(t)

# Eigenvector and closeness are not for exits
tabular = pandas.concat(v).T.stack().unstack(level=0)

tabular[['in_deg_cen','out_deg_cen','deg_cen']].round(3).to_latex('pre2015/deg_cen.tex')
tabular[['bet_cen','bet_cen_exit']].round(3).to_latex('pre2015/bet_cen.tex')
tabular[['eivec_cen_exit','close_cen_exit','sq_clust']].round(3).to_latex('pre2015/others.tex')

keys_ns = [
    '{}_mean_sdsp',
    '{}_10pc_sdsp',
    '{}_50pc_sdsp',
    '{}_90pc_sdsp',
    '{}_2000m',
    '{}_1000m',
    '{}_500m',
    '{}_250m',
    '{}_mean_tdsp',
    '{}_10pc_tdsp',
    '{}_50pc_tdsp',
    '{}_90pc_tdsp',
    '{}_8_tdsp',
    '{}_4_tdsp',
    '{}_2_tdsp',
    '{}_1_tdsp',
    '{}_0_in_deg',
    '{}_1_in_deg',
    '{}_2_in_deg',
    '{}_3+_in_deg',
    '{}_0_out_deg',
    '{}_1_out_deg',
    '{}_2_out_deg',
    '{}_3+_out_deg',
]    

labels_ns = [
    'Mean Spatial Distance',
    '10% Spatial Distance',
    '50% Spatial Distance',
    '90% Spatial Distance',
    '2000 metres',
    '1000 metres',
    '500 metres',
    '250 metres',
    'Mean Topological Distance',
    '10% Topological Distance',
    '50% Topological Distance',
    '90% Topological Distance',
    '8 Topological Distance',
    '4 Topological Distance',
    '2 Topological Distance',
    '1 Topological Distance',
    '0 In-Degree',
    '1 In-Degree',
    '2 In-Degree',
    '3+ In-Degree',
    '0 Out-Degree',
    '1 Out-Degree',
    '2 Out-Degree',
    '3+ Out-Degree',
]    

ns = {}
for key in ['frac_nodes','num_nodes']:
    this = [k.format(key) for k in keys_ns]
    ns[key] = corrnorm.loc[this]
    ns[key].index = labels_ns
    for t in this:
        remaining.remove(t)

tabular_ns = pandas.concat(ns).T.stack().unstack(level=0)
tabular_ns.round(3).to_latex('pre2015/nodestats.tex')

keys_sp = [
    'mean_{}sp',
    '10pc_{}sp',
    '50pc_{}sp',
    '90pc_{}sp',
    'max_{}sp',
]

labels_sp = [
    'Mean',
    '10%',
    '50%',
    '90%',
    'Maximum',
]

sp = {}
for key in ['sd','td']:
    this = [k.format(key) for k in keys_sp]
    sp[key] = corrnorm.loc[this]
    sp[key].index = labels_sp
    for t in this:
        remaining.remove(t)

tabular_sp = pandas.concat(sp).T.stack().unstack(level=0)
tabular_sp.round(3).to_latex('pre2015/shortestpath.tex')

labels_final = [
    'Number of Nodes',
    'Number of Edges',
    'Mean Edge Width',
    'Mean Edge Length',
    'Mean Edge Area',
    'Total Edge Area',
    'Total Edge Length',
    'Mean Degree',
    'Graph Density',
    'Graph Transitivity',
    'Strongly Connected Components',
    'Mean Edge Betweenness Centrality',
    'Mean Edge Betweenness Centrality for Exit',    
]

keys_final = [
    u'num_nodes',
    u'num_edges',
    u'mean_edge_width',
    u'mean_edge_length',
    u'mean_edge_area',
    u'sum_edge_area',
    u'sum_edge_length',
    u'mean_degree',
    u'density',
    u'transitivity',
    u'num_strgconcom',
    u'mean_edge_bet_cen',
    u'mean_edge_bet_cen_exit',
]

tabular_final = corrnorm.loc[keys_final]
tabular_final.index = labels_final
tabular_final.round(3).to_latex('pre2015/remaining.tex')

# 'edge_betweenness_centrality',
# 'edge_betweenness_centrality_exit',

showing = {
    'in_degree_centrality':'In-degree centrality',
    'out_degree_centrality':'Out-degree centrality',
    'degree_centrality':'Degree centrality',
    'betweenness_centrality':'Betweenness centrality',
    'betweenness_centrality_exit':'Exit betweenness centrality',
    'eigenvector_centrality':'Eigenvector centrality',
    'closeness_centrality':'Closeness centrality',
    'square_clustering':'Square clustering',
    'spatial_distance':'Spatial distance from exit (metres)',
    'topological_distance':'Topological distance from exit (node hops)'
    }

for show,label in showing.iteritems():
    plt.close('all')
    plt.figure(figsize=(10,2.5))
    for i,x in enumerate(sample):
        ax = plt.subplot(131+i,frameon=False)
        bc = load_json(show,s.place,x)
        values = np.array(bc.values(),dtype=float)
        nxnodes = nx.draw_networkx_nodes(s.h.G,pos=nx.get_node_attributes(s.h.G,'pos'),nodelist=bc.keys(),node_color=values,linewidths=0,node_size=1)
        cbar = plt.colorbar(nxnodes,ticks=[values.min(),(values.min()+values.max())/2,values.max()],orientation='horizontal')
        plt.scatter(*pos[x],s=200,c=destin_color[x],alpha=0.5,marker='o',zorder=100)
        plt.title(attr[x]['label'])
        # plt.xlabel('Longitude',fontsize=15)
        # plt.ylabel('Latitude',fontsize=15)
        ax.axis('off')
        plt.axis('equal')
    plt.tight_layout()
    plt.suptitle(label,fontsize=15,y=0)
    plt.savefig('pre2015/metric-sample-bristol-{}.pdf'.format(show),bbox_inches='tight')

frac_p50_btc = []
frac_p90_btc = []
frac_mean_btc = []
for k,v in M['mean_bet_cen'].iteritems():
    j= np.array(load_json('betweenness_centrality',*k).values())
    p50,p90 = np.percentile(j,[50,90])
    frac_p50_btc.append(sum(j>p50)/float(j.size))
    frac_p90_btc.append(sum(j>p90)/float(j.size))
    frac_mean_btc.append(sum(j>np.mean(j))/float(j.size))
