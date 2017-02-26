"""
- 	This only works on the Macbook Pro laptop because the
	dataset relies on tho top 50 UK cities and amenities
	which is only available on this machine.
-	Make sure you have PostgreSQL running	
"""
# Run:
	# ipython --pylab
# Then:
	# %ed aac.py
import sys
sys.path.append('core')
import db
import pandas
import abm
import pickle
import numpy as np
import matplotlib.pyplot as plt

critical = ['doctors','police','fire_station','hospital','emergency_phone','fuel']

# -----------------------
# -----------------------
# Produce an amenity rank for points

result = db.Query('SELECT p.amenity, COUNT(*) AS amenityCount from planet_osm_point AS p GROUP BY p.amenity ORDER BY amenityCount DESC;').result
point = pandas.DataFrame(result,columns=['Amenity','Point Count'])

# Produce an amenity rank for polygons
result = db.Query('SELECT p.amenity, COUNT(*) AS amenityCount from planet_osm_polygon AS p GROUP BY p.amenity ORDER BY amenityCount DESC;').result
poly = pandas.DataFrame(result,columns=['Amenity','Polygon Count'])

combined = pandas.merge(point,poly, on='Amenity',how='outer').set_index('Amenity').fillna(0)
combined['Total Count'] = combined.sum(axis=1)
combined = combined.sort('Total Count',ascending=0)

# -----------------------
# Select 40 and mark the common ones

top40_am = combined[1:41]
top40_am['Rank'] = range(1,41)
top40_am.to_latex('static/top40_am.tex')

# -----------------------
# Load all places and their amenities
places = abm.Places('bristol25').names
p = {}
a_point = {}
a_poly = {}
b = {}
for place in places:
	# Load poopulation
	p[place] = db.Population(place,table='pop_gpwv4_2015')
	# Load boundary
	b[place] = db.Boundary(place).shape

# Query point amenities
try:
	# Load from file if it exists
	with open('static/a-point.pickle','r') as f:
	    a_point = pickle.load(f)
except IOError:
	for place in places:	
		# Load amenities from database
		a_point[place] = {}	
		for amenity in top40_am.index:
			q = """SELECT DISTINCT ST_X(p.way), ST_Y(p.way) 
					FROM planet_osm_point AS p, (SELECT ST_GeomFromText(%s,4326) AS way) AS b WHERE p.amenity=%s AND ST_Intersects(p.way, b.way);"""
			a_point[place][amenity] = db.Query(q, (b[place].wkt,amenity,)).result
	# Cache this to file as this takes some time to compute
	with open('static/a-point.pickle','w') as f:
	    pickle.dump(a_point,f)

# Query polygon amenities
try:
	# Load from file if it exists
	with open('static/a-poly.pickle','r') as f:
	    a_poly = pickle.load(f)
except IOError:
	for place in places:
		# Load amenities from database
		a_poly[place] = {}		
		for amenity in top40_am.index:
			q = """SELECT DISTINCT ST_X(c.way), ST_Y(c.way) 
					FROM (SELECT DISTINCT ST_Centroid(p.way) AS way 
						FROM planet_osm_polygon AS p, (SELECT ST_GeomFromText(%s,4326) AS way) AS b WHERE p.amenity=%s AND ST_Intersects(p.way, b.way)) AS c;"""
			a_poly[place][amenity] = db.Query(q, (b[place].wkt,amenity,)).result

	# Cache this to file as this takes some time to compute
	with open('static/a-poly.pickle','w') as f:
	    pickle.dump(a_poly,f)

# Sort places by population size
total_pop = [sum(p[place].pop) for place in places]
sorted_places,sorted_pop = zip(*sorted(zip(places,total_pop),key=lambda tot: tot[1],reverse=True))

# Join the two lists and count them
a = {}
amenities = pandas.DataFrame()
for place in sorted_places:
	a[place] = {}
	a_count = []
	for amenity in top40_am.index:
		a[place][amenity] = a_point[place][amenity]+a_poly[place][amenity]
		a_count.append(len(a[place][amenity]))
	amenities[place] = a_count
amenities.index = top40_am.index

# -----------------------
# Now convert the data points into pop and am grids
from scipy.ndimage.filters import gaussian_filter as gf
pop_grid = {}
nz_pop = {}
am_grid = {}
nz_am = {}
am_pop_grid = {}
nz_am_pop = {}
# Sigmas for gaussian filter
sigmas = np.linspace(0,1.5,16)
# Resolution of GPWv4
# Set out rules for a grid, use the population map resolution of 30 arc seconds
res = float(1)/120

for place in places:
	# Find corners
	x_min, x_max, y_min, y_max = min(p[place].lon), max(p[place].lon), min(p[place].lat), max(p[place].lat)
	x_cells, y_cells = int(round((x_max-x_min)/res))+1, int(round((y_max-y_min)/res))+1

	# In the grid, xi at x_min is 0, yi at y_min is 0
	xi = lambda(lon): min(int(round((lon-x_min)/res)),x_cells-1)
	yi = lambda(lat): min(int(round((y_max-lat)/res)),y_cells-1)

	# Map population to a grid cell
	pop_grid[place]=np.zeros((y_cells,x_cells))
	for lon,lat,pop in zip(p[place].lon,p[place].lat,p[place].pop):
		pop_grid[place][yi(lat),xi(lon)] = pop

	# Pick out the non zero population to compare amenity against
	nz = pop_grid[place] > 0
	nz_pop[place] = pop_grid[place][nz]
	pop_grid[place][np.logical_not(nz)] = np.nan

	# Make sure that the population is being correctly added
	pop_diff = sum(nz_pop[place]) - sum(p[place].pop)
	if pop_diff > 0.001:
		print 'Mismatch between nz and actual pop for ', place, 'is', pop_diff
		raise TypeError

	# Map amenity to a grid cell
	am_grid[place] = {}
	nz_am[place] = {}
	am_pop_grid[place] = {}		
	nz_am_pop[place] = {}

	for sigma in sigmas:
		am_grid[place][sigma] = {}
		nz_am[place][sigma] = {}		
		am_pop_grid[place][sigma] = {}
		nz_am_pop[place][sigma] = {}		

	for amenity in amenities.index:
		raw_am_grid=np.zeros((y_cells,x_cells))
		for lon,lat in a[place][amenity]:
			raw_am_grid[yi(lat),xi(lon)] += 1

		# Apply gaussian filter to amenity grid and correlate against population grid
		for sigma in sigmas:
			am_grid[place][sigma][amenity] = gf(raw_am_grid,sigma=sigma)
			am_grid[place][sigma][amenity][np.logical_not(nz)] = np.nan
			nz_am[place][sigma][amenity] = am_grid[place][sigma][amenity][nz]
			am_pop_grid[place][sigma][amenity] = am_grid[place][sigma][amenity] / pop_grid[place]
			am_pop_grid[place][sigma][amenity][np.logical_not(nz)] = np.nan
			nz_am_pop[place][sigma][amenity] = am_pop_grid[place][sigma][amenity][nz]

# -----------------------
# Query and make a list of places by areas
result = db.Query("SELECT * FROM (SELECT p.osm_id, p.name, p.admin_level, SUM(ST_Area(p.way,false))  AS area, COUNT(p.osm_id) AS polygon_count FROM planet_osm_polygon AS p WHERE boundary=%s GROUP BY p.osm_id, p.name,p.admin_level ORDER BY admin_level, area DESC) AS q WHERE polygon_count = 1 AND CAST(admin_level AS INT) < 10 AND name != '' AND area != 'NaN' AND area BETWEEN 235816681.819764*3/4 AND 235816681.819764*5/4",('administrative',)).result
places_by_area = dict([(city,area) for i,city,al,area,pc in result])
# Output list of cities, population and sample amenity count to a tex file
pop_count = pandas.DataFrame()
pop_count['Population'] = ['{:,}'.format(int(round(this))) for this in sorted_pop]
# Area in km^2
pop_count['Area'] = [int(round(places_by_area[this]/1000000)) for this in sorted_places]
# Density in persons/km^2
pop_count['Density']=np.round(sorted_pop/pop_count['Area'])
# Rank order
pop_count['Rank'] = range(1,51)
pop_count.index = sorted_places
pop_count.to_latex('static/pop-count.tex')

# -----------------------
# Calculate all correlation coefficient
# Initialise the r and p values
import scipy.stats as ss
try:
	# Load from file if it exists
	with open('static/correlations.pickle','r') as f:
	    pearson_pa,spearman_pa,pearson_aa,spearman_aa = pickle.load(f)	    
except IOError:
	pearson_pa = {}
	spearman_pa = {}	
	pearson_aa = {}
	spearman_aa = {}
	for place in places:
		pearson_pa[place] = {}
		spearman_pa[place] = {}
		pearson_aa[place] = {}
		spearman_aa[place] = {}
		for sigma in sigmas:
			# Population-amenity correlations
			data = pandas.DataFrame(nz_am[place][sigma])[amenities.index]
			data['population'] = nz_pop[place]
			pearson_pa[place][str(sigma)] = data.corr(method = 'pearson')['population'][:-1]
			spearman_pa[place][str(sigma)] = data.corr(method = 'spearman')['population'][:-1]
			# Amenity/person-amenity/person correlations
			data = pandas.DataFrame(nz_am_pop[place][sigma])[amenities.index]
			pearson_aa[place][str(sigma)] = data.corr(method = 'pearson')
			spearman_aa[place][str(sigma)] = data.corr(method = 'spearman')
	# Cache this to file as this takes some time to compute
	with open('static/correlations.pickle','w') as f:
	    pickle.dump([pearson_pa,spearman_pa,pearson_aa,spearman_aa],f)

# Pearson+Spearman r_pa for ALL amenity vs ALL cities mean value
# First, calculate the means and standard deviations
ppa = pandas.concat({place:pandas.DataFrame(pearson_pa[place]).transpose() for place in sorted_places}).pow(2)
spa = pandas.concat({place:pandas.DataFrame(spearman_pa[place]).transpose() for place in sorted_places}).pow(2)

# Pearson+Spearman r_aa for ALL amenity vs ALL cities mean value
# First, calculate the means and standard deviations
paa = pandas.concat({place:pandas.concat(pearson_aa[place]) for place in sorted_places}).pow(2)
saa = pandas.concat({place:pandas.concat(spearman_aa[place]) for place in sorted_places}).pow(2)

# Amenity per person grid
am_pop = amenities.divide(sorted_pop)

# Generate median rank tables
am_ranks_pa = pandas.DataFrame()
am_ranks_pa['Original']=amenities.index
am_ranks_pa = am_ranks_pa.set_index('Original')
r = ppa.stack().median(level=2).sort_values(ascending=False)
am_ranks_pa['Linear Order'] = r.index
am_ranks_pa['rpa2'] = np.round(r.values,3)
r = spa.stack().median(level=2).sort_values(ascending=False)
am_ranks_pa['Monotonous Order']= r.index
am_ranks_pa['rhopa2']= np.round(r.values,3)
am_ranks_pa['Rank'] = range(1,41)
am_ranks_pa.to_latex('static/am_ranks_pa.tex')

am_ranks_aa = pandas.DataFrame()
am_ranks_aa['Original']=amenities.index
am_ranks_aa = am_ranks_aa.set_index('Original')
r = paa.stack().median(level=2).sort_values(ascending=False)
am_ranks_aa['Linear Order'] = r.index
am_ranks_aa['raa2'] = np.round(r,3)
r = saa.stack().median(level=2).sort_values(ascending=False)
am_ranks_aa['Monotonous Order'] = r.index
am_ranks_aa['rhoaa2'] = np.round(r,3)
am_ranks_aa['Rank'] = range(1,41)
am_ranks_aa.to_latex('static/am_ranks_aa.tex')

pl_ranks_pa = pandas.DataFrame()
pl_ranks_pa['Original']=sorted_places
pl_ranks_pa = pl_ranks_pa.set_index('Original')
r = ppa.stack().median(level=0).sort_values(ascending=False)
pl_ranks_pa['Linear Order'] = r.index
pl_ranks_pa['rpa2'] = np.round(r.values,3)
r = spa.stack().median(level=0).sort_values(ascending=False)
pl_ranks_pa['Monotonous Order']= r.index
pl_ranks_pa['rhopa2']= np.round(r.values,3)
pl_ranks_pa['Rank'] = range(1,51)
pl_ranks_pa.to_latex('static/pl_ranks_pa.tex')

pl_ranks_aa = pandas.DataFrame()
pl_ranks_aa['Original']=sorted_places
pl_ranks_aa = pl_ranks_aa.set_index('Original')
r = paa.stack().median(level=0).sort_values(ascending=False)
pl_ranks_aa['Linear Order'] = r.index
pl_ranks_aa['raa2'] = np.round(r.values,3)
r = saa.stack().median(level=0).sort_values(ascending=False)
pl_ranks_aa['Monotonous Order'] = r.index
pl_ranks_aa['rhoaa2'] = np.round(r.values,3)
pl_ranks_aa['Rank'] = range(1,51)
pl_ranks_aa.to_latex('static/pl_ranks_aa.tex')

# Filtering critical amenities only
critical=list(amenities.ix[critical].sum(axis=1).sort_values(ascending=False).keys())

am_ranks_pa_critical = pandas.DataFrame()
am_ranks_pa_critical['Original']=critical
am_ranks_pa_critical = am_ranks_pa_critical.set_index('Original')
r = ppa[critical].stack().median(level=2).sort_values(ascending=False)
am_ranks_pa_critical['Linear Order (critical)'] = r.index
am_ranks_pa_critical['rpa2'] = np.round(r.values,3)
r = spa[critical].stack().median(level=2).sort_values(ascending=False)
am_ranks_pa_critical['Monotonous Order (critical)']= r.index
am_ranks_pa_critical['rhopa2']= np.round(r.values,3)
am_ranks_pa_critical['Rank'] = range(1,len(critical)+1)
am_ranks_pa_critical.to_latex('static/am_ranks_pa_critical.tex')

am_ranks_aa_critical = pandas.DataFrame()
am_ranks_aa_critical['Original']=critical
am_ranks_aa_critical = am_ranks_aa_critical.set_index('Original')
r = paa[critical].stack().median(level=2)[critical].sort_values(ascending=False)
am_ranks_aa_critical['Linear Order (critical)'] = r.index
am_ranks_aa_critical['raa2'] = np.round(r.values,3)
r = saa[critical].stack().median(level=2)[critical].sort_values(ascending=False)
am_ranks_aa_critical['Monotonous Order (critical)'] = r.index
am_ranks_aa_critical['rhoaa2'] = np.round(r.values,3)
am_ranks_aa_critical['Rank'] = range(1,len(critical)+1)
am_ranks_aa_critical.to_latex('static/am_ranks_aa_critical.tex')

pl_ranks_pa_critical = pandas.DataFrame()
pl_ranks_pa_critical['Original']=sorted_places
pl_ranks_pa_critical = pl_ranks_pa_critical.set_index('Original')
r = ppa[critical].stack().median(level=0).sort_values(ascending=False)
pl_ranks_pa_critical['Linear Order (critical)'] = r.index
pl_ranks_pa_critical['rpa2'] = np.round(r.values,3)
r = spa[critical].stack().median(level=0).sort_values(ascending=False)
pl_ranks_pa_critical['Monotonous Order (critical)']= r.index
pl_ranks_pa_critical['rhopa2']= np.round(r.values,3)
pl_ranks_pa_critical['Rank'] = range(1,51)
pl_ranks_pa_critical.to_latex('static/pl_ranks_pa_critical.tex')

pl_ranks_aa_critical = pandas.DataFrame()
pl_ranks_aa_critical['Original']=sorted_places
pl_ranks_aa_critical = pl_ranks_aa_critical.set_index('Original')
r = paa[critical].stack().median(level=0).sort_values(ascending=False)
pl_ranks_aa_critical['Linear Order (critical)'] = r.index
pl_ranks_aa_critical['raa2'] = np.round(r.values,3)
r = saa[critical].stack().median(level=0).sort_values(ascending=False)
pl_ranks_aa_critical['Monotonous Order (critical)'] = r.index
pl_ranks_aa_critical['rhoaa2'] = np.round(r.values,3)
pl_ranks_aa_critical['Rank'] = range(1,51)
pl_ranks_aa_critical.to_latex('static/pl_ranks_aa_critical.tex')

# Economy of scale
offset = 10000
stats = pandas.DataFrame()
pf = {}
df = amenities.transpose()
df['pop'] = np.array(sorted_pop)/offset
df = np.log10(df)
df.replace([-np.inf,np.inf],np.nan,inplace=True)
for amenity in amenities.index:
	v = df[['pop',amenity]].dropna()
	pf[amenity] = np.polyfit(v['pop'],v[amenity],1)
	stats[amenity] = [pf[amenity][0]]+list(ss.pearsonr(v['pop'],v[amenity]))
stats.index = ['Exponent','Pearson r','P-value']
stats = np.round(stats.transpose(),3)
stats.loc[stats['P-value']<0.001,'P-value']='$<0.001$'
stats.loc[stats['P-value']<0.005,'P-value']='$<0.005$'
stats.loc[stats['P-value']<0.01,'P-value']='$<0.01$'
stats.loc[stats['P-value']<0.05,'P-value']='$<0.05$'
stats.to_latex('static/economy.tex')
# Draw
plt.close('all')
plt.figure(figsize=(10,10))
markers = ['.','o','+','d','x',',']
colours = ['c','y','m','r','g','b']
alphas = [0.4,0.4,0.4,0.4,0.4,0.4]
upper = 0
lower = 0
dummy = [-100,100]
for i,amenity in enumerate(critical):
	v = df[['pop',amenity]].dropna()
	upper = max(v['pop'].max(),v[amenity].max(),upper)
	lower = min(v['pop'].min(),v[amenity].min(),lower)
	plt.scatter(v['pop'],v[amenity],marker=markers[i],c=colours[i],alpha=alphas[i],label='$\\beta={:.2f}$ ({})'.format(pf[amenity][0],amenity))
	plt.plot(dummy,np.polyval(pf[amenity],dummy),c=colours[i])
	plt.hold(True)
plt.plot(dummy,dummy,ls='--',c='.3',markeredgewidth=0)
plt.gca().set_aspect('equal', adjustable='box')
plt.xlim(lower-0.1,upper+0.1)
plt.ylim(lower-0.1,upper+0.1)
plt.legend(loc='lower right');
plt.ylabel('log($T_a$)',fontsize=15)
plt.xlabel('log($T_p \\times 10^{-%0.0f}$)'%np.log10(offset),fontsize=15)
plt.savefig('static/economy-of-scale.pdf',bbox_inches='tight')