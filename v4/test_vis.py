import sys
sys.path.append('core')
import abm, db
reload(abm)
reload(db)

sim = abm.Sim('test_abm_Qc',(-2.88612967859241, 54.91133135582125, -2.9609706894330743, 54.88327220183169))
sim.n = 200000
sim.h.init_destins()
sim.init_scenario('ia')
sim.tracks_file(mode='load')

import matplotlib.pyplot as plt
plt.close('all')

sim.init_scenario('ia')
sim.load_events()
v=abm.Viewer(sim,bbox=bbox.buffer(0.015).bounds,video=False)
v.path(which='speed')