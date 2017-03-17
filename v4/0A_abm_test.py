import sys
sys.path.append('core')
import abm, db
reload(abm)
reload(db)

sim = abm.Sim('abm_test','City of London')
sim.scenarios = ['k5']
sim.n = 1000
sim.run_by_agent(verbose=False)

# Generate video
import matplotlib.pyplot as plt
plt.close('all')
sim.init_scenario('k5')
sim.load_events()
v=abm.Viewer(sim,video=True)
v.path(which='speed')