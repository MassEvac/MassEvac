import massevac_v2 as me

a=me.abm('massevac','City of Bristol',n=100000,destins=[2421])
a.run_sim(fresh=True,live_video=True)
