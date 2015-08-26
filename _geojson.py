import abm
s=abm.Sim('test','City of Bristol')
s.n=500000

# Temporarily define scenario so that this can be logged
s.scenario = 'ia'
s.load_agents()

# Prepare output

mT = {}
sT = {}

for scenario in s.scenarios:
  s.scenario = scenario
  s.load_results()

  T = {}

  # Iterate through list with edge number that agents are on
  for e,t in zip(s.E, s.T):
  	# Create a list of time per edge
  	try:
  		T[e].append(t)
  	except KeyError:
  		T[e] = [t]

  mT[scenario] = {}
  sT[scenario] = {}

  for i in T:
  	mT[scenario][i] = mean(T[i])
  	sT[scenario][i] = std(T[i])


# Write to json file
s.h.init_route()

from shapely.geometry import LineString,mapping

fname = '{0}/time.json'.format(s.agents_file())
features = []
for i in T:
    u,v,d = s.h.edges[i]

    # Determine nearest catchment areas
    nearest = np.inf
    nearest_x = np.nan
    for x in s.h.destins:
      try:
        dist1 = s.h.route_length[x][u]
        dist2 = s.h.route_length[x][v]

        # Take the max of dist1 and dist2
        if dist1 > dist2:
          dist = dist1
        else:
          dist = dist2

        if dist < nearest:
          nearest_x = x
          nearest = dist
      except KeyError:
        pass

    # Generate properties

    properties = {}
    properties["index"] = i
    properties["nearest_x"] = nearest_x
    properties["dist_to_x"] = nearest

    for scenario in s.scenarios:
      properties["{0}_mean_time".format(scenario)] = mT[scenario][i]
      properties["{0}_stdv_time".format(scenario)] = sT[scenario][i]

    l = LineString([s.h.nodes[u],s.h.nodes[v]])
    feature = {
      "type": "Feature",
      "properties": properties,
      "geometry": mapping(l)
    }

    features.append(feature)

out = {
  "type": "FeatureCollection",
  "features": features
}
with open (fname,'w') as f:
    json.dump(out,f,indent=True)     