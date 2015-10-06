import sys
sys.path.append('core')

import abm
import db

reload(abm)
reload(db)
# s=abm_new.Sim('flood','Carlisle')
self=abm_new.Sim('flood',(-2.88612967859241, 54.91133135582125, -2.9609706894330743, 54.88327220183169))
self.n = 100000
self.run()