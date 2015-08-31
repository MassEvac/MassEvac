import abm
reload(abm)
s=abm.Sim('test','City of Bristol')
s.n=500000
properties = load_agent_times(s)
fname = '{0}/time.json'.format(s.agents_file())
s.h.geojson_edges(fname,properties)