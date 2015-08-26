# import abm
# s = abm.Sim('test','City of Bristol')
# s.scenario = 'ia'
# s.n = 500000
# s.load_agent_meta()
# s.h.init_route()


def d2x(self,agent):
	x = self.X[agent]
	e = self.E[agent]
	l = self.L[agent]
	n1,n2,d = self.h.edges[e]

	return d['distance'] - l + s.h.route_length[x][n2]
