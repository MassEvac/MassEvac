import sys
sys.path.append('core')
import abm, db, csv, numpy as np
reload(abm)
reload(db)

self = abm.Sim('test_abm_fd',(-2.88612967859241, 54.91133135582125, -2.9609706894330743, 54.88327220183169))
self.h.destins = [2551]
self.scenarios = ['ia']

x = [50000]#[25000,50000,100000,200000]
y = [4.4,4.5,4.6,4.7,4.8,4.9,5.0,5.1,5.2,5.3]
z = [5.0,5.5,6.0,6.5,7.0,7.5,8.0,8.5,9.0]
y = np.arange(start=4.4,stop=5.36,step=0.1)
z = np.arange(start=5.0,stop=9.1,step=0.5)
print len(x),len(y),len(z),len(x)*len(y)*len(z)

resarray = np.zeros((len(x),len(y),len(z)))

for i,self.n in enumerate(x):
    with open('data/tgf2015/{}.csv'.format(self.n), 'w') as fp:
        a = csv.writer(fp, delimiter=',');
        for j,abm.fd.k_vmin in enumerate(y):
            for k,abm.fd.k_lim in enumerate(z):
                abm.fd.precomputation()
                self.use_buffer = False
                self.run(agent_progress_bar=False)
                resarray[i][j][k] = self.tstep
                data = [[self.n,abm.fd.k_vmin,abm.fd.k_lim,self.tstep]]
                print data
                a.writerows(data)

import pickle
with open('file.pickle','wb') as f:
    pickle.dump(resarray,f)

with open('file.pickle','rb') as f:
    resarray = pickle.load(f)



for n in [25000,50000,100000,200000]:

    x = []
    y = []
    z = []
    d = []

    with open('data/tgf2015/{}.csv'.format(n),'r') as f:
        a = csv.reader(f, delimiter=',')
        for r in a:
            x.append(int(r[0]))
            y.append(float(r[1]))
            z.append(float(r[2]))
            d.append(float(r[3]))

    print min(d)            

    # x = [100000,200000,300000,400000]
    # y = [2.8, 3.2, 3.6, 4.4, 4.8, 5.2]
    # z = [5.0,6.0,7.0,8.0,9.0]

    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    import matplotlib.pyplot as plt
    import numpy as np

    fig = plt.figure(figsize=(10,10),dpi=100)
    ax = fig.gca(projection='3d')
    Y, Z = np.meshgrid(z, y)

    Y = np.reshape(y,(10,9))
    Z = np.reshape(z,(10,9))

    D = np.round(np.reshape(d,(10,9)))

    surf = ax.plot_surface(Z, Y, D, rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=True)
    # ax.set_zlim(-1.01, 1.01)

    ax.zaxis.set_major_locator(LinearLocator(6))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.0f'))

    # fig.colorbar(surf, shrink=0.5, aspect=5)

    ax.set_xlabel('$k_{lim}$',fontsize=25)
    ax.set_ylabel('$k_{v,min}$',fontsize=25)
    ax.set_zlabel('$T$',fontsize=25)

    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(axis='both', which='minor', labelsize=15)

    plt.savefig('figs/{}.pdf'.format(n))
    plt.show()

    # NORMED


    fig = plt.figure(figsize=(10,10),dpi=100)
    ax = fig.gca(projection='3d')
    Y, Z = np.meshgrid(z, y)

    Y = np.reshape(y,(10,9))
    Z = np.reshape(z,(10,9))

    D = np.round(np.reshape(d,(10,9)))/min(d)

    surf = ax.plot_surface(Z, Y, D, rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=True)
    ax.set_zlim(1, 16)

    ax.zaxis.set_major_locator(LinearLocator(6))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.0f'))

    # fig.colorbar(surf, shrink=0.5, aspect=5)

    ax.set_xlabel('$k_{lim}$',fontsize=25)
    ax.set_ylabel('$k_{v,min}$',fontsize=25)
    ax.set_zlabel('$T/T_{min}$',fontsize=25)

    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(axis='both', which='minor', labelsize=15)

    plt.savefig('figs/{}-norm.pdf'.format(n))
    plt.show()