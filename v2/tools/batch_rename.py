# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import os
dirs=os.listdir('cities/osm_gb')

# <codecell>

for i,f in enumerate(sorted(dirs)):
    if i > 0:
        os.rename('cities/osm_gb/{0}/nodes.{1}.mat'.format(f,i+1),'cities/osm_gb/{0}/nodes.mat'.format(f))

# <codecell>


