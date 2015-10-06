# For the non-flooded case:
# - Remove all small roads from self.G
# - Loop n times
# 	- Pick a random location
# 	- Determine the time it takes to reach hospital from that location
#	- Look at the distribution of time across n

# For the flooded case:
# - Remove all small roads from self.G
# - Remove all blocked roads from self.G
# - Loop n times
# 	- Pick a random location
# 	- Determine the time it takes to reach hospital from that location
#	- Look at the distribution of time across n

import sys
sys.path.append('core')

import db

reload(db)

self = db.Highway((-2.88612967859241, 54.91133135582125, -2.9609706894330743, 54.88327220183169))
hospital = self.nearest_node(-2.955547,54.8955621)
self.destins = [hospital]
self.init_route()
self.fig_destins()
db.plt.ion()
db.plt.show()