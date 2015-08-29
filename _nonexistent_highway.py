import os
import db
reload(db)
dirs = os.listdir('db/osm_gb/')

for d in dirs:
    fname = 'db/osm_gb/'+d+'/highway.edges.gz'
    if os.path.isdir('db/osm_gb/'+d):
        # if not os.path.isfile(fname):
            try:
                db.Highway(d)
            except ValueError:
                print d