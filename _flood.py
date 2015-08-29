import db

reload(db)

# Test
f = db.Flood('flood/1.tif')

# print f.floodStatus((f.maxx,f.maxy),(f.minx,f.miny))
# print f.isFlooded((f.maxx,f.maxy),(f.minx,f.miny))
# plt.ion()
# f.fig()

h = db.Highway((f.maxx,f.maxy,f.minx,f.miny))

# nx.draw_networkx_edges(s.h.G,s.h.nodes,arrows=False,edgelist=s.h.edges,edge_color=flooded)

properties=[{'flooded':f.isFlooded(h.nodes[u],h.nodes[v])} for u,v,d in h.edges]

h.geojson_edges('flood/test.json',properties)