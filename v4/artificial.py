"""
    In order to interface with the minions, make sure you run the following on a separate screen:

        $ ipcluster start
"""
%pylab

import ipyparallel as ipp
c = ipp.Client()

# Syncronise modules with the minions
with c[:].sync_imports():
    from core import abm,db
    import networkx
    import numpy
    from matplotlib import pyplot
    import shapely
    import copy

def artificial(param):
    levels,branches,initial_occupancy,n_per_edge,speedup,k_vmin,k_lim = param

    place = 'l={}|b={}|i={}|n={}|s={}'.format(*param)

    # Use custom settings up update abm settings
    this = copy.deepcopy(abm.settings['k5'])
    this['k_vmin'] = k_vmin
    this['k_lim'] = k_lim

    if k_vmin == 5:
        scenario = 'k{}'.format(k_lim)
    else:
        scenario = 'k{}-kvmin{}'.format(k_lim,k_vmin)
    abm.settings[scenario] = this

    # Loadup a new instance of the simulation
    sim=abm.Sim('artificial',None,fresh=True,fresh_db=True,speedup=speedup,save_route=False)
    G=sim.h.G

    # 
    sim.place = place

    # Iterate over levels
    for level in range(levels):
        if level == 0:
            this_node = 0
            G.add_node(this_node,pos=(0,level),level=level)
        # This is the number of nodes at this level
        gen = ((n,d) for n,d in G.nodes(data=True) if d['level']==level)
        for that_node,d in gen:
            ypos = level+1      
            # At each level, add new branches
            for branch in range(branches):
                this_node += 1
                that_level = level + 1
                # xpos = d['pos'][0]+(branches/2.-branch)/that_level
                G.add_node(this_node,level=that_level)
                G.add_edge(this_node,that_node,blocked=False,level=that_level)

    # Generate position of nodes
    pos = networkx.graphviz_layout(G,prog='dot')
    nodes = pos.keys()
    coord = pos.values()
    lon,lat = numpy.array(zip(*coord))
    lon -= min(lon)
    if max(lon) > 0:
        lon /= max(lon)
    lat -= min(lat)
    if max(lat) > 0:
        lat /= max(lat)
    pos = dict(zip(nodes, zip(lon,lat))) #
    sim.h.l,sim.h.r = min(lon),max(lon)
    sim.h.b,sim.h.t = min(lat),max(lat)

    networkx.set_node_attributes(G,'pos',pos)
    networkx.get_node_attributes(G,'pos')
    networkx.set_edge_attributes(G,'pop_dist',1./G.number_of_edges())

    sim.h.destins = [0]
    sim.scenarios=[scenario]
    sim.n = n_per_edge*G.number_of_edges()
    sim.init_scenario(scenario)

    area = n_per_edge/sim.fd.k_max/initial_occupancy
    # Length of an edge is what an agent can travel in 60 seconds/number of levels
    distance = sim.fd.v_ff*600/levels
    assumed_width = area/distance

    networkx.set_edge_attributes(G,'area',area)
    networkx.set_edge_attributes(G,'distance',distance)
    networkx.set_edge_attributes(G,'assumed_width',assumed_width)
    networkx.set_edge_attributes(G,'hiclass','road')
    sim.h.boundary=shapely.geometry.MultiPoint(networkx.get_node_attributes(G,'pos').values()).envelope
    return sim

from time import sleep
def print_progress(params,r):
    no_of_cores = len(c)
    while len(params) > r.progress:
        print('Progress',r.progress,'of',len(params))
        # Print one line for each active job
        left = min(no_of_cores,len(params)-r.progress)
        for i in range(left):
            idx = i+r.progress
            for line in r.stdout[idx].split('\n')[-1:]:
                print idx,line
        sleep(no_of_cores)

# Synchronise this function
c[:]['artificial'] = artificial
# Create a pointer to the minions in load balanced view where jobs will be sent
lbview = c.load_balanced_view()

# The job that we want the minions to do
def job(param):

    sim = artificial(param)
    sim.run_by_agent(rerun=False,verbose=False,log_events_percent=100,metadata=True)
    sim.load_events()

    tstep = max([e[-1][1] for e in sim.events])*sim.speedup
    return tstep

# The scope of the parameter space
import itertools
_levels = [1,2,3,4,5,6]
_branches = [1,2,3]
_initial_occupancy = [0.01,0.02,0.04,0.08]
_n_per_edge = [1,10,100]
_speedup = [100,10,1]
_k_lim = [5,5.5,6,6.5,7,7.5,8,8.5,9]
_k_vmin = [5.0,5.05,5.1,5.15,5.2,5.25,5.3,5.35]

# ------------- TREES -------------------

# Draw 6*3 grid with artificial graphs
if False:
    plt.close('all')
    plt.figure(figsize=(10,12));
    count = 0
    for l in _levels:
        for b in _branches:
            count = count + 1
            plt.subplot(len(_levels),len(_branches),count)
            sim = artificial((l,b,0.01,10,1,'k5'))
            pos = networkx.get_node_attributes(sim.h.G,'pos')
            levels = networkx.get_edge_attributes(sim.h.G,'level')
            edgelist = levels.keys()
            edge_color = levels.values()
            networkx.draw_networkx(sim.h.G,pos=pos,edgelist=edgelist,with_labels=False,node_size=1,arrows=False,edge_color=edge_color)
            ax = gca()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.title('$l = {}$, $b = {}$'.format(l,b),fontsize=15)
    # plt.tight_layout()
    plt.savefig('artificial/6-3-branches.pdf',bbox_inches='tight')

# ------------- PART 1 -------------------

params = list(itertools.product(_levels,_branches,_initial_occupancy,_n_per_edge,_speedup,[5.0],[5]))
print 'PART 1',len(params), 'scenarios'

# Process the results and generate output
if False:
    r = lbview.map(job,params)

    while len(params) > r.progress:
        print('Progress',r.progress,'of',len(params))
        sleep(1)

    import pandas
    df = pandas.DataFrame(params)
    df['r'] = r.result()
    df['r']/= 600
    df.columns=['l','b','i','a','t','k_vmin','k_lim','r']    
    df.to_csv('artificial/first.csv')

import pandas
df = pandas.DataFrame.from_csv('artificial/first.csv')

# -------------------------------
# r**2 value for update intervals
y = df.query('t==1')['r']  
x1 = df.query('t==10')['r']
x2 = df.query('t==100')['r']

import scipy.stats as ss
lr1 = ss.linregress(x1,y)
print(lr1)
# LinregressResult(slope=1.0004095427190762, intercept=-0.0054820311795076293, rvalue=0.99993084226531626, pvalue=0.0, stderr=0.00080431995484583864)
# r**2 = 0.9998616893134248
lr2 = ss.linregress(x2,y)
print(lr2)
# LinregressResult(slope=1.0011082381434915, intercept=0.0058062577185156172, rvalue=0.99992654802123926, pvalue=0.0, stderr=0.00082949700485965804)
# r**2 = 0.9998531014376717
# so Standard error for s==10 slightly lower than s==100
# Use update interval of 10 rather than 100 but not much difference

# Difference in the values
dx1 = y.values-x1.values
dx2 = y.values-x2.values

if False:
    plt.figure(figsize=(12,5))
    plt.subplot(121)
    plt.scatter(x1,y+10,label='$x = T_{10t}/T_f$, $y = T_{1t}/T_f + y_{offset}$',c='g',s=5,linewidths=0)
    plt.scatter(x2,y,label='$x = T_{100t}/T_f$, $y = T_{1t}/T_f$',c='b',s=5,linewidths=0)
    plt.legend(loc='upper left')
    plt.xlabel('$x$',fontsize=15)
    plt.ylabel('$y$',fontsize=15)
    plt.xlim([-10,None])
    plt.ylim([-10,None])
    plt.axis('equal')
    # Update interval error histogram
    plt.subplot(122)
    plt.hist(dx1,label='$(T_{1t} - T_{10t})/T_f$',bins=50,histtype='step',color='g')
    plt.hist(dx2,label='$(T_{1t} - T_{100t})/T_f$',bins=40,histtype='step',color='b')
    plt.xlabel('$(T_{1t} - T_{?t})/T_f$',fontsize=15)
    plt.ylabel('Number of scenarios',fontsize=15)
    plt.ylim([0,250])
    plt.legend()
    plt.savefig('artificial/update-interval-error.pdf',bbox_inches='tight')

# -------------------------------
# r**2 value for number of agents
x1 = df.query('t==1 and a==1')['r']  
x2 = df.query('t==1 and a==10')['r']
y = df.query('t==1 and a==100')['r']

import scipy.stats as ss
lr1 = ss.linregress(x1,y)
print(lr1)
# LinregressResult(slope=1.2929538335591408, intercept=-0.80225250144621718, rvalue=0.98241228775648703, pvalue=9.3206295810829002e-53, stderr=0.029372565333049674)
# r**2 = 0.9651339031349346
lr2 = ss.linregress(x2,y)
print(lr2)
# LinregressResult(slope=0.99999921817815518, intercept=-0.011392803035930932, rvalue=0.99999830952664392, pvalue=3.1201057345593029e-193, stderr=0.00021977098465217275)
# r**2 = 0.9999966190561456
# So n==10 > n==1
# Hence, use 10 people rather than 1
# Good fit but n==1 not making much contribution given by the fact that c_{1} = -0.0025373

dx1 = y.values-x1.values
dx2 = y.values-x2.values

if False:
    plt.figure(figsize=(12,5))
    plt.subplot(121)
    plt.scatter(x1,y+10,label='$x = T_{1a}/T_f$, $y = T_{100a}/T_f + y_{offset}$',c='r',s=5,linewidths=0)
    plt.scatter(x2,y,label='$x = T_{10a}/T_f$, $y = T_{100a}/T_f$',c='g',s=5,linewidths=0)
    plt.legend(loc='upper left')
    plt.xlabel('$x$',fontsize=15)
    plt.ylabel('$y$',fontsize=15)
    plt.xlim([-10,None])
    plt.ylim([-10,None])
    plt.axis('equal')
    # Update interval error histogram
    plt.subplot(122)
    plt.hist(dx1,label='$(T_{100a} - T_{1a})/T_f$',bins=150,histtype='step',color='r')
    plt.hist(dx2,label='$(T_{100a} - T_{10a})/T_f$',bins=1,histtype='step',color='g')
    plt.xlabel('$(T_{100a} - T_{?t})/T_f$',fontsize=15)
    plt.ylabel('Number of scenarios',fontsize=15)
    plt.legend()
    plt.savefig('artificial/number-of-agents-error.pdf',bbox_inches='tight')

# -------------------------------
# r**2 value for number of agents
minidf = df.query('t==1 and a==100').copy()

Nedges=[]
for b,l in minidf[['b','l']].values:
    # If b == 1, n = l
    if b == 1:
        Nedges.append(l)
    # If b != 1, n is as follows
    else:
        Nedges.append((b**(l+1)-b)/(b-1))
    print(b,l,Nedges[-1])

minidf['Nedges'] = Nedges

exp_e = np.linspace(0,3,1000)
rsq = []
for e in exp_e:
    minidf['i^e*Nedges']=minidf['i']**e*minidf['Nedges']
    rsq.append(minidf.corr()['r']['i^e*Nedges']**2)

max_rsq = max(rsq)
max_idx = rsq.index(max_rsq)
# Optimal value of m
e_opt = exp_e[max_idx]
print(e_opt,max_rsq)
# (1.075075075075075, 0.99493883072121214)

# Optimal e
if False:
    plt.figure()    
    plt.plot(exp_e,rsq)
    plt.scatter(e_opt,max_rsq,label='$e_{opt} = %0.3f$'%m_opt,c='r',zorder=10)
    plt.xlabel('Exponent $e$',fontsize=15)
    plt.ylabel('Pearson $r^2$',fontsize=15)
    plt.xlim([0,3])
    plt.ylim([None,1.05])
    plt.legend(scatterpoints=1)
    plt.savefig('artificial/exponent-e-vs-r^2.pdf',bbox_inches='tight')

    minidf['i^eopt*Nedges']=minidf['i']**e_opt*minidf['Nedges']
    x = minidf['i^eopt*Nedges']
    y = minidf['r']

pf = np.polyfit(x,y,1)
# array([ 2.38247238,  0.30895032])

# Where do the two lines intersect
x_intersect = (1-pf[1])/pf[0]
# 0.29005569672782699

xf = np.logspace(np.log10(x.min()), np.log10(x.max()),100)
yf = np.polyval(pf,xf)

# Draw the figure
# i^m*b^l vs T/Tf

from matplotlib import cm
from matplotlib.colors import LogNorm
norm = LogNorm(vmin=_initial_occupancy[0], vmax=_initial_occupancy[-1])
clim = [_initial_occupancy[0],_initial_occupancy[-1]]

if False:
    plt.figure(figsize=(12,6))
    ax=plt.subplot(121)
    sc = plt.scatter(x=x,y=y,c=minidf['i'],cmap='Greys_r',clim=clim,norm=norm,label=None)
    plt.plot(xf,yf+1-min(yf),label='$T/T_f = m i^{e_{opt}} N_{edges} + c$')
    plt.plot(xf,np.ones(shape(xf)),label='$T/T_f = 1$')
    plt.legend(loc='upper left')
    plt.xlabel('$i^{e_{opt}} N_{edges}$',fontsize=15)
    plt.ylabel('$T/T_f$',fontsize=15)
    # Draw the figure
    # i^m*b^l vs T/Tf
    ax = plt.subplot(122)
    sc = plt.scatter(x=x,y=y,c=minidf['i'],cmap='Greys_r',clim=clim,norm=norm,label=None)
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(xf,yf+1-min(yf),label='$T/T_f = m i^{e_{opt}} N_{edges} + c$')
    plt.plot(xf,np.ones(shape(xf)),label='$T/T_f = 1$')
    plt.legend(loc='upper left')
    plt.xlabel('$i^{e_{opt}} N_{edges}$',fontsize=15)
    plt.ylabel('$T/T_f$',fontsize=15)
    cb = plt.colorbar(sc,ax=plt.gcf().axes,shrink=0.5,ticks=_initial_occupancy, format='$%.2f$',orientation='horizontal')
    cb.set_label(label='Initial occupancy rate $i$',fontsize=15)
    plt.savefig('artificial/i^eopt*Nedges-vs-T-Tf+1-log-log.pdf',bbox_inches='tight')

# ------------- PART 2 -------------------

# _levels,_branches,_initial_occupancy,_n_per_edge,_speedup,_scenarios
params2 = list(itertools.product([6],[3],_initial_occupancy,[10],[10],_k_vmin,_k_lim))
print 'PART 2',len(params2), 'scenarios'


x_offset = 0.2
y_offset = 0.2

plt.close('all')
for velocity in [True,False]:
    plt.figure(figsize=(10,10))

    for count,t in enumerate(itertools.product([5,7,9],[5,5.3])):
        k_lim,k_vmin = t

        if velocity:
            ax = plt.subplot(321+count)
        else:
            ax = plt.subplot(321+count)

        fd = abm.FundamentalDiagram(speedup=1,k_vmin=k_vmin,k_lim=k_lim)

        k = fd.k
        v = [v/fd.speedup for v in fd.v]
        q = [q/fd.speedup for q in fd.q]

        if velocity:
            ax.plot(k,v,'r-',linewidth=4,label='$v_{max} = %0.2f \ \mathrm{[m/s]}$'%max(v))
            if k_vmin == 5:
                ax.set_ylabel(fd.velocityLabel,fontsize=15)
            ax.set_ylim(0,fd.v_ff+y_offset)
            ax.axvline(fd.k_vmin,c='y',linestyle='-.',linewidth=3,label='$k_{v,min} = %0.2f \ \mathrm{[ped/m^2]}$'%fd.k_vmin)
            ax.axvline(fd.k_lim,c='b',linestyle=':',linewidth=3,label='$k_{lim} = %0.2f \ \mathrm{[ped/m^2]}$'%fd.k_lim)        
        else:
            ax.plot(k,q,'g--',linewidth=4,label='$Q_{max} = %0.2f \ \mathrm{[ped/(ms)]}$'%max(q))
            if k_vmin == 5:
                ax.set_ylabel(fd.flowLabel,fontsize=15)
            ax.set_ylim(min(q),max(q)+y_offset)
            ax.axvline(fd.k_vmin,c='y',linestyle='-.',linewidth=3)
            ax.axvline(fd.k_lim,c='b',linestyle=':',linewidth=3)

        plt.title('$k_{{v_{{min}}}} = {:0.2f}$, $k_{{lim}} = {:0.2f}$'.format(k_vmin,k_lim))

        ax.tick_params(axis='both')

        ax.set_xlim(0,10)
        ticks = [0,3,6,9]
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticks)

        if k_lim == 9:
            ax.set_xlabel(fd.densityLabel,fontsize=15)
        ax.yaxis.set_major_locator(LinearLocator(4))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))        


    if velocity:
        plt.savefig('artificial/fd-velocity.pdf',bbox_inches='tight')
    else:
        plt.savefig('artificial/fd-flow.pdf',bbox_inches='tight')



# Process the results
if False:
    r = lbview.map(job,params2)

    print_progress(params2,r)

    import pandas
    df2 = pandas.DataFrame(params2)
    df2['r'] = r.result()
    df2['r']/= 600
    df2.columns=['l','b','i','a','t','k_vmin','k_lim','r']
    df2.to_csv('artificial/second.csv')

import pandas
df2 = pandas.DataFrame.from_csv('artificial/second.csv')

# Draw 3D plot
if True:
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter

    for normed in [True,False]:
        fig = plt.figure(figsize=(10,10))
        for sp,i in enumerate(_initial_occupancy):
            ax = plt.subplot(221+sp,projection='3d')

            thisdf = df2.query('i=={}'.format(i))
            x = thisdf['k_lim']
            y = thisdf['k_vmin']
            z = thisdf['r']

            shape = (len(_k_vmin),len(_k_lim))
            X = np.reshape(x,shape)
            Y = np.reshape(y,shape)
            Z = np.reshape(z,shape)
            if normed:
                Z/=min(z)

            surf = ax.plot_surface(X,Y,Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                                   linewidth=0, antialiased=True)

            # ax.xaxis.set_major_locator(LinearLocator(5))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

            # ax.yaxis.set_major_locator(LinearLocator(4))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

            if normed:
                ax.set_zlim(1, 11)
            else:
                ax.set_zlim(0, 1500)
            
            ax.zaxis.set_major_locator(LinearLocator(6))
            ax.zaxis.set_major_formatter(FormatStrFormatter('%.0f'))

            ax.set_xlabel('$k_{lim}}$', fontsize=15)
            ax.set_ylabel('$k_{v,min}}$', fontsize=15)
            if normed:
                ax.set_zlabel('$T/T_{min}}$', fontsize=15)
            else:
                ax.set_zlabel('$T/T_f}$', fontsize=15)

            ax.xaxis._axinfo['label']['space_factor'] = 2.5
            ax.yaxis._axinfo['label']['space_factor'] = 2.5
            ax.zaxis._axinfo['label']['space_factor'] = 2.5

            plt.title('$i = {}$'.format(i))

        plt.tight_layout()

        if normed:
            fname = 'artificial/surface-normed.pdf'.format(i)
        else:
            fname = 'artificial/surface.pdf'.format(i)
        plt.savefig(fname,bbox_inches='tight',pad_inches=0.2)


# Regress i against each other
if True:

