import sys
sys.path.append('core')
import abm, db
reload(abm)
reload(db)

self = abm.Sim('test_abm_Qc',(-2.88612967859241, 54.91133135582125, -2.9609706894330743, 54.88327220183169))
self.n = 200000
self.h.init_destins()
self.init_scenario('ia')
self.tracks_file(mode='load')

import matplotlib.pyplot as plt
plt.close('all')

v = abm.Viewer(self,video=False)
v.path(which='speed')