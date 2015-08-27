import json
from shapely.geometry import LineString,mapping

fname = '{0}/time.json'.format(s.agents_file())
features = []

def geojson_edges(h, properties, fname):
    '''Produces a geojson file with feature tag 

    Inputs
    ------
        edges: List
            List of edges

        nodes: List
            List of node coordinate tuples

        properties: List
            List of dicts where the index corresponds to edge index

        fname:  LineString
            Path to the file where the geojson file will be dumped

    Output
    ------

    '''

    for i,ep in enumerate(zip(h.edges,properties)):

        e,p = ep
        u,v,d = e

        # Generate properties

        p["index"] = i
        p["u"] = u
        p["v"] = v
        p["hiclass"] = h.hiclass[d['highway']]
        p["awidth"] = h.width[p["hiclass"]]
        p.update(d)    

        l = LineString([h.nodes[u],h.nodes[v]])
        feature = {
            "type": "Feature",
            "properties": p,
            "geometry": mapping(l)
        }

        features.append(feature)

    out = {
        "type": "FeatureCollection",
        "features": features
    }

    with open (fname,'w') as f:
        json.dump(out,f,indent=True)     

properties=[{'flooded':f.isFlooded(h.nodes[u],h.nodes[v])} for u,v,d in h.edges]

geojson_edges(h,properties,'flood/test.json')