import sys
sys.path.append('core')
import abm, db
reload(abm)
reload(db)

# Load flood scenario
f = db.Flood('flood/qgis/severe_binary.tif')
print f.floodStatus((f.maxx,f.maxy),(f.minx,f.miny))
print f.isFlooded((f.maxx,f.maxy),(f.minx,f.miny))
plt.ion()
f.fig()

# Load simulation framework
s = abm.Sim('flood','Carlisle')

# properties=[{'flooded':f.isFlooded(s.h.nodes[u],s.h.nodes[v])} for u,v,d in s.h.edges]
# s.h.geojson_edges('flood/test.json',properties)

# nx.draw_networkx_edges(s.h.G,s.h.nodes,arrows=False,edgelist=s.h.edges,edge_color=flooded)
s.blocked=[f.isFlooded(s.h.nodes[u],s.h.nodes[v]) for u,v,d in s.h.edges]

s.scenarios=['ia']
s.random_successor = True
s.n=100000
s.run()