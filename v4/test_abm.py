import sys
sys.path.append('core')
import abm, db
reload(abm)
reload(db)

# self = abm.Sim('flood','Carlisle')
self = abm.Sim('test_abm',(-2.88612967859241, 54.91133135582125, -2.9609706894330743, 54.88327220183169))
self.n = 100000
self.use_buffer = False
self.run(agent_progress_bar=False)