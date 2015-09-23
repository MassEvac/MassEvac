# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# This class defines the fundamental diagram.

# <markdowncell>

# #Optimum density
# 
# Analytically, we can show that,
# 
# When $v = m.p + c$
# 
# And $f = v.p$
# 
# $f = m.p^2 + c.p$
# 
# $df/dp = 2.m.p + c$
# 
# When $df/dp = 0$,
# 
# $2.m.p + c = 0$
# 
# $ p = -c/2.m$

# <codecell>

import numpy
import matplotlib.pyplot as plt
import math

# <codecell>

class fd:
    # Maximum velocity (m/s)
    vMax = 1.66   
    
    # Minimum velocity (m/s)
    vMin = vMax/10
    
    # Maximum density (agent/m^2)
    pMax = 5.5
    
    # Minimum density (agent/m^2)    
    pMin = 0.00

    # Density upto which velocity is flat (agent/m^2)
    pFlat = 1.00
    
    # Gradient
    m = (vMax - vMin)/(pMin + pFlat - pMax)
    
    # Offset
    c = vMin - m*pMax
    
    # Optimum density to maximise flow - see explaination below
    pOpt = -c/2/m
    
    # Number of bins to profile the density with
    bins = int(math.ceil(pMax-pMin))
    
    # This is the average density in each bin
    # Note that the last bin is 5.25 = (5+5.5)/2
    bin_mean = []
    for i in range(bins):
        lowerBound = i + pMin
        upperBound = lowerBound + 1
        if upperBound > pMax:
            upperBound = pMax
        bin_mean.append((lowerBound+upperBound)/2)
    
    def __init__(self,speedup=60):
        self.speedup = speedup
        self.Qmax = self.velocity(self.pOpt)*self.pOpt
        self.Qmin = self.velocity(self.pMax)*self.pMax
        
        
    # This should form the basis for calculation of speed from density using,
    # velocity = m*density + c
    def velocity(self, p):
        # Calculate velocity from the input density within the boundary limits
        v = self.m * p + self.c
        if p < self.pFlat:
            v = self.vMax
        elif v < self.vMin:
            v = self.vMin
        return v*self.speedup
    
    def figure(self):
        p = numpy.linspace(self.pMin,self.pMax,int(self.pMax*10)+1)
        v = p.copy()
        
        for i,j in enumerate(p):
            v[i] = self.velocity(j)/self.speedup
        
        f = v*p
        
        offset = 0.5
        
        fontsize=20
        fig = plt.figure(figsize=(9,6),dpi=70)
        ax = fig.add_subplot(111)
        ax.plot(p,v,'r-',linewidth=4,label='$V_{max}$ = %0.2f $m/s$'%max(v))
        ax.set_xlabel('Density $agent/m^{2}$',fontsize=fontsize)
        ax.set_ylabel('Velocity $m/s$',fontsize=fontsize)
        ax.set_xlim(self.pMin,self.pMax+offset)
        ax.set_ylim(self.vMin,self.vMax+offset)
        #ax.legend(loc=2,fontsize=fontsize)

        ax2 = ax.twinx()
        ax2.set_ylabel('Flow $agent/ms$',fontsize=fontsize)
        ax2.plot(p,f,'g--',linewidth=4,label='$F_{max}$ = %0.2f  $agent/ms$'%max(f))
        ax2.set_ylim(f.min(),f.max()+offset)
        #ax2.legend(loc=0,fontsize=fontsize)
        
        plt.show()
        plt.savefig('fd-model.pdf',dpi=300)
        return fig
    
    def which_bin(self,density):
        bin = int(density-self.pMin)
        
        # Sometimes, the initial position of the agents may lead the density to exceed the
        # maximum density, in which case revert the bin index to less than maximum bin index
        if not bin < self.bins:
            bin = self.bins - 1
            
        return bin

