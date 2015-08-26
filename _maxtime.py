# Make a list of all agents who take the longest to traverse the simulation

reload(abm)
import abm
s = abm.Sim('bristol25','City of Bristol')
s.scenarios = ['ia']
s.scenario = 'ia'
s.load_agent_meta()
s.load_result_meta()

track_list_dict = {}

for k in s.T_destin:
	maximum = max(s.T_destin[k])
	max_index = s.T_destin[k].index(maximum)
	agent_index = s.T_destin_index[k][max_index]
	track_list_dict[k] = agent_index
	print k,agent_index

s.track_list = track_list_dict.values()

s.load_result_meta()

# s.run_sim(rerun=True)