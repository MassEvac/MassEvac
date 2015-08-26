import abm
reload(abm)
s=abm.Sim('bristol25','Birmingham')
s.init_scenario('ia')
s.load_agent_meta()
s.load_result_meta()
d=10941
import scipy.stats as ss

n = s.n_destin[d]
w = s.h.destin_width_dict[d]

dx90 = ss.scoreatpercentile(s.DX_destin[d],90)
        
# in minutes
t90ff = dx90/abm.fd.vFf/60
t90 = ss.scoreatpercentile(s.T_destin[d],90)

# integer form so that it can be used as index
it90ff = int(np.ceil(t90ff))

qmeff = np.mean(s.Q_destin[d][:it90ff])
qmebn = np.mean(s.Q_destin[d][it90ff:])            

qmdff = np.median(s.Q_destin[d][:it90ff])
qmdbn = np.median(s.Q_destin[d][it90ff:])                

q90ff = ss.scoreatpercentile(s.Q_destin[d][:it90ff],90)

q90bn = ss.scoreatpercentile(s.Q_destin[d][it90ff:],90)

q90 = ss.scoreatpercentile(s.Q_destin[d],90)

qpred = n/t90ff/w

print d, qpred, q90ff