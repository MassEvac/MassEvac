import abm
reload(abm)
s=abm.Sim('new',(-2.88612967859241, 54.91133135582125, -2.9609706894330743, 54.88327220183169))
s.scenarios=['ia']
s.n=10000
s.run_sim()