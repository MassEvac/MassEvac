import abm
reload(abm)
import db
reload(db)
f = db.Flood('flood/1.tif')
# print f.floodStatus((f.maxx,f.maxy),(f.minx,f.miny))
# print f.isFlooded((f.maxx,f.maxy),(f.minx,f.miny))
# plt.ion()
# f.fig()
s = abm.Sim('flood',(f.maxx,f.maxy,f.minx,f.miny))
# properties=[{'flooded':f.isFlooded(s.h.nodes[u],s.h.nodes[v])} for u,v,d in s.h.edges]
# s.h.geojson_edges('flood/test.json',properties)

# nx.draw_networkx_edges(s.h.G,s.h.nodes,arrows=False,edgelist=s.h.edges,edge_color=flooded)
s.blocked=[f.isFlooded(s.h.nodes[u],s.h.nodes[v]) for u,v,d in s.h.edges]

s.scenarios=['ia']
s.random_successor = True
s.n=80000
s.run_sim()