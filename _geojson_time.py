import abm
reload(abm)
import db
reload(db)
s=abm.Sim('test','City of Bristol')
s.n=500000
properties = s.load_agent_times()
fname = '{0}/time.json'.format(s.agents_file())
s.h.geojson_edges(fname,properties)