import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

        # Plot figures of distances to exit in flooded and non flooded case
        # db.plt.figure()
        # db.plt.hist(normal.route_length[hospital].values(),bins=50,histtype='step')
        # db.plt.hist(flooded.route_length[hospital].values(),bins=50,histtype='step')
        # db.plt.xlabel('distance (m)')
        # db.plt.ylabel('number of nodes')


        # Plot figures of distances to hospitals from nodes on a map            
        # db.plt.figure()
        # flooded.fig_highway()
        # nx.draw_networkx_nodes(flooded.F, flooded.G.node, nodelist=diff[case].keys(), node_color=diff[case].values(), alpha=0.5,node_shape='.',node_size=40,linewidths=0)
        # db.plt.xlabel('lon')
        # db.plt.ylabel('lat')

"""Original and binary flood map"""
original = db.Flood('flood/res1_1_original.tif')
binary = db.Flood('flood/res1_1_binary_30cm.tif')
plt.close('all')
plt.figure(figsize=(10,14))
plt.subplot(211)
plt.imshow(original.Map*100,extent=[self.minx,self.maxx,self.miny,self.maxy],vmin=0,alpha=1.0, cmap=plt.get_cmap('Blues'))
cb=plt.colorbar(ticks=[30,1000],orientation='horizontal',shrink=0.5)
cb.set_ticklabels(['$30$ cm','$1000$ cm'])
plt.xlabel('Longitude',fontsize=15)
plt.ylabel('Latitude',fontsize=15)
plt.subplot(212)
plt.imshow(binary.Map,extent=[self.minx,self.maxx,self.miny,self.maxy],vmin=0,alpha=1.0, cmap=plt.get_cmap('Blues'))
cb=plt.colorbar(ticks=[0,1],orientation='horizontal',shrink=0.5)
cb.set_ticklabels(['Not flooded','Flooded'])
plt.xlabel('Longitude',fontsize=15)
plt.ylabel('Latitude',fontsize=15)
plt.savefig('flood/figs/flood-map.pdf',bbox_inches='tight')

"""Location of hospital and reference node on map showing flooded areas"""
ref_node = 1 # Distance to hospital from this node
this = diff2h.loc[8*100+8]
with_access = this.dropna().index
no_access = set(nodes_in_bbox).difference(with_access)
plt.close('all')
normal.fig_highway()
fig = plt.gcf()
fig.set_size_inches(12,8,forward=True)
self=flood
plt.imshow(binary.Map,extent=[self.minx,self.maxx,self.miny,self.maxy],vmin=0,alpha=1.0, cmap=plt.get_cmap('Blues'))
cb=plt.colorbar(ticks=[0,1],orientation='horizontal',shrink=0.5)
cb.set_ticklabels(['Not flooded','Flooded'])
plt.plot(*bbox.boundary.xy,label='Flood map extent')
x,y=zip(*[normal.G.node[n] for n in with_access])
plt.scatter(x,y,c='gray',alpha=0.3,linewidth=0,label='With access')
x,y=zip(*[normal.G.node[n] for n in no_access])
plt.scatter(x,y,c='r',alpha=0.3,marker='x',label='No access')
plt.scatter(*normal.G.node[hospital],s=200,c='g',alpha=0.5,marker='o',label='Hospital')
plt.scatter(*normal.G.node[ref_node],s=200,c='y',alpha=0.5,marker='v',label='Reference node')
plt.legend(scatterpoints=1)
plt.xlabel('Longitude',fontsize=15)
plt.ylabel('Latitude',fontsize=15)
plt.axis('equal')
plt.xlim(flood.minx-0.03, flood.maxx+0.03)
plt.ylim(flood.miny-0.03, flood.maxy+0.03)  
plt.savefig('flood/figs/carlisle+floodmap+hospital+ref-node.pdf',bbox_inches='tight')


"""Draw polgons in general"""
import matplotlib.pyplot as plt
from descartes import PolygonPatch
from matplotlib.collections import PatchCollection
def draw_polygons(shapes,colors,label,show_hospital=True): 
 patches=[]
 for s in shapes:
     patches.append(PolygonPatch(s))
 pc = PatchCollection(patches, cmap=plt.get_cmap('RdYlBu'), alpha=1.0)
 pc.set_array(np.array(colors))
 pc.set_clim([np.min(colors),np.max(colors)])
 fig = plt.figure(figsize=(12,8))
 ax = plt.gca()
 ax.add_collection(pc)
 cb=plt.colorbar(pc,orientation='horizontal',shrink=0.5)
 cb.set_label(label,fontsize=15)
 plt.xlabel('Longitude',fontsize=15)
 plt.ylabel('Latitude',fontsize=15)
 plt.axis('equal')
 plt.xlim(flood.minx-0.01, flood.maxx+0.055)
 plt.ylim(flood.miny, flood.maxy+0.015)  
 if show_hospital:
  plt.scatter(*normal.G.node[hospital],s=200,c='g',alpha=0.5,marker='o')
  plt.scatter(*normal.G.node[hospital],s=10,c='g',alpha=1.0,marker='o')

"""Draw polygons with number of nodes""" 
plt.close('all')

shapes,colors = zip(*[(OA_shapes[id],len(nodes)) for id,nodes in OA_nodes.iteritems()])
draw_polygons(shapes,colors,'Number of nodes')
plt.savefig('flood/figs/OAA-no-of-nodes.pdf',bbox_inches='tight')


plt.close('all')
for setting in settings:    
    """Draw polygons with percent nodes that can reach hospitals under 08 mins averages across 10000 scenarios"""
    shapes,colors = zip(*[(OA_shapes[id], baseline_value) for id,baseline_value in baseline[setting].iteritems()])
    draw_polygons(shapes,colors,'Baseline Access Score in $< {}$ mins'.format(settings[setting]))
    plt.savefig('flood/figs/OA-baseline-access-{}.pdf'.format(setting),bbox_inches='tight')

    """Draw polygons with percent nodes that can reach hospitals under 08 mins averages across 10000 scenarios"""
    shapes,colors = zip(*[(OA_shapes[id], mean_value) for id,mean_value in access[setting].mean().iteritems()])
    draw_polygons(shapes,colors,'$\mu$(Access Score) in $< {}$ mins'.format(settings[setting]))
    plt.savefig('flood/figs/OA-mean-access-{}.pdf'.format(setting),bbox_inches='tight')

    """Draw polygons with percent nodes that can reach hospitals under 30 mins std across 10000 scenarios"""
    shapes,colors = zip(*[(OA_shapes[id], std_value) for id,std_value in access[setting].std().iteritems()])
    draw_polygons(shapes,colors,'$\sigma$(Access Score) in $< {}$ mins'.format(settings[setting]))
    plt.savefig('flood/figs/OA-std-access-{}.pdf'.format(setting),bbox_inches='tight')

"""Inflows histogram"""
def subhist(key,color):
    this = inflows[key]
    bins=np.logspace(np.log10(this.min()),  np.log10(this.max()), 50)
    plt.hist(key_inflows,bins=bins,histtype='step',color=color,label=key,normed=True,log=False)
    plt.xscale('log')
    plt.legend(fontsize=12)
    plt.xlabel('$\mathrm{m^3/s}$',fontsize=15)
    print key,this.mean(),this.std()
plt.figure(figsize=(15,3))
plt.subplot(131)
subhist('Eden','r')
plt.subplot(132)
subhist('Caldew','g')
plt.subplot(133)
subhist('Petterill','b')
plt.savefig('flood/figs/inflows-subplots.pdf', bbox_inches='tight' )

"""Histogram subplots of distance to hospital in normal condition and during flood for 4 flood conditions"""
plt.close('all')
plt.figure(figsize=(12,8))
normal_dist2h.hist(bins=50,histtype='step',cumulative=True,label='No flood',normed=True)
for case in range(4):
    this = inflows.loc[case]
    dist2h.loc[case].hist(bins=50,histtype='step',cumulative=True,label='Flood: Eden {:0.0f}, Petterill {:0.0f}, Caldew {:0.0f} [$\mathrm{{m^3/s}}$] '.format(this['Eden'],this['Petterill'],this['Caldew']),normed=True)
    plt.xlabel('Distance to hospital (m)',fontsize=15)
    plt.ylabel('Cumulative proportion of nodes',fontsize=15)
    plt.legend(loc='lower right',fontsize=15)
plt.tight_layout()
plt.savefig('flood/figs/dist2h-normal-flood.pdf',bbox_inches='tight')

"""Scatter inflows vs dist2h and diff2h"""
plt.close('all')
fig = plt.figure(figsize=(16,10))
ax = fig.add_subplot(121, projection='3d')
sc = ax.scatter(xs=inflows['Eden'], ys=inflows['Petterill'], zs=inflows['Caldew'], c=np.log(dist2h[ref_node]),marker='.',linewidth=0,alpha=0.6)
cb = plt.colorbar(sc,orientation='horizontal')
cb.set_label('Log distance to hospital during flooding (m)',fontsize=15)
ax.set_xlabel('Eden inflow ($\mathrm{m^3/s}$)',fontsize=15)
ax.set_ylabel('Petterill inflow ($\mathrm{m^3/s}$)',fontsize=15)
ax.set_zlabel('Caldew inflow ($\mathrm{m^3/s}$)',fontsize=15)
xlim,ylim,zlim=ax.get_xlim(),ax.get_ylim(),ax.get_zlim()
ax = fig.add_subplot(122, projection='3d')
sc = ax.scatter(xs=inflows['Eden'], ys=inflows['Petterill'], zs=inflows['Caldew'], c=np.log(diff2h[ref_node]),marker='.',linewidth=0,alpha=0.6)
cb = plt.colorbar(sc,orientation='horizontal')
cb.set_label('Log difference in distance to hospital (m)',fontsize=15)
ax.set_xlabel('Eden inflow ($\mathrm{m^3/s}$)',fontsize=15)
ax.set_ylabel('Petterill inflow ($\mathrm{m^3/s}$)',fontsize=15)
ax.set_zlabel('Caldew inflow ($\mathrm{m^3/s}$)',fontsize=15)
ax.set_xlim(*xlim),ax.set_ylim(*ylim),ax.set_zlim(*zlim)
plt.savefig('flood/figs/ref-node-inflows-vs-dist2h+diff2h.pdf',bbox_inches='tight')

"""Histogram of scores"""
plt.close('all')
plt.figure(figsize=(12,6))
bins = np.linspace(0,1,21)
for setting in settings:
    baseline[setting].hist(bins=bins,histtype='step',normed=True,label='Baseline: $< {}$ minutes'.format(settings[setting]),cumulative=True)    
    access[setting].mean().hist(bins=bins,histtype='step',normed=True,label='Flood: $< {}$ minutes'.format(settings[setting]),cumulative=True)
plt.xlabel('$\mu$(Access score)',fontsize=15)
plt.ylabel('Cumulative proportion of output areas (OA)',fontsize=15)
plt.legend(loc='upper left')
plt.ylim(-0.1,1.1)
plt.xlim(-0.1,1.1)
plt.savefig('flood/figs/OA-mean-access-score-CDF.pdf', bbox_inches='tight' )

"""Histogram of standard deviation of scores"""
plt.figure(figsize=(12,6))
bins = np.linspace(0,0.4,21)
for setting in settings:
    access[setting].std().hist(bins=bins,histtype='step',normed=True,label='Flood: $< {}$ minutes'.format(settings[setting]),cumulative=True)
plt.xlabel('$\sigma$(Access score)',fontsize=15)
plt.ylabel('Cumulative proportion of output areas (OA)',fontsize=15)
plt.legend(loc='upper left')
plt.ylim(-0.1,1.1)
plt.xlim(-0.1,0.5)
plt.savefig('flood/figs/OA-std-access-score-CDF.pdf', bbox_inches='tight' )

"""AUC figures"""
plt.ion()
plt.close('all')
for setting in settings:
    plt.figure(figsize=(16,10))
    # plt.suptitle('Hospital reach within {} minutes'.format(settings[setting]),fontsize=18)
    for j,clf_name in enumerate(classifiers.keys()):
        plt.subplot(2,3,j+1)
        plt.title('Hospital within {} minutes [{}]'.format(settings[setting],clf_name),fontsize=15)        
        this = (results['classifier']==clf_name)&(results['setting']== settings[setting])
        for i in results[this].index:
            plt.plot(fpr[i],tpr[i],label='Access < ${}$ AUC ${:0.2f}$'.format(results['criteria'][i],results['AUC'][i]))
            plt.legend(loc='lower right')
            plt.axis('equal')
            plt.xlim(-0.1,1)
            plt.ylim(0,1.1)
            plt.plot([-2,2],[-2,2],'--',c='gray')
            plt.xlabel('False Alarm Rate',fontsize=15)
            plt.ylabel('Probability of Detection',fontsize=15)
    plt.tight_layout()            
    plt.savefig('flood/figs/{}min-AUC.pdf'.format(settings[setting]),bbox_inches='tight')

"""Histogram of OA node count"""
plt.figure();
OA_node_count.hist(bins=50,histtype='step',cumulative=True)

