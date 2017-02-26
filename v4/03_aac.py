# RUN static.py before running elements of this script

# -----------------------
# For ALL amenities
# Draw amenity rank vs amenity count
plt.close('all')
ram = list(reversed(amenities.index))
top40_am[['Point Count','Polygon Count','Total Count']].reindex(ram).plot(kind='barh')
plt.xscale('log')
plt.xlabel('Amenity Count',fontsize=15)
plt.ylabel('')
# Save
# savefig('static/amenity-rank.pdf',bbox_inches='tight')

# -----------------------
# For ALL cities
# Show amenity count grid in log scale
plt.close('all')
imshow(np.log10(amenities),interpolation='None')
ax = gca()
yticks(range(40),amenities.index,fontsize=10)
xticks(range(50),amenities.keys(),fontsize=10,rotation=90)
cb = colorbar()
cb.set_ticks(range(4))
cb.set_ticklabels(['$10^{}$'.format(i) for i in range(4)])
cb.set_label('Number of amenities',fontsize=15)
draw()
# savefig('static/amenity-count.pdf',bbox_inches='tight')

# Amenity per person
plt.close('all')
imshow(am_pop,interpolation='None')
ax = gca()
yticks(range(40),am_pop.index,fontsize=10)
xticks(range(50),am_pop.keys(),fontsize=10,rotation=90)
cb = colorbar()
cb.set_label('Amenities/person',fontsize=15)
draw()
# savefig('static/amenity-per-person.pdf',bbox_inches='tight')

# Produce bar chart with population
plt.close('all')
data = list(sorted_pop)
data.reverse()
barh(range(50),data)
labels = list(sorted_places)
labels.reverse()
yticks(range(50),labels,fontsize=10)
plt.xscale('log')
plt.xlabel('Population',fontsize=15)
# savefig('static/pop-count.pdf',bbox_inches='tight')

# Produce population histogram
plt.close('all')
n_bins=50
log_pop = np.log10(sorted_pop)
bins=np.logspace(log_pop.min()-0.1, log_pop.max()+0.1, n_bins)
plt.hist(sorted_pop,bins=bins)
plt.ylim(0,11)
plt.xlabel('Population',fontsize=15)
plt.ylabel('Number of cities',fontsize=15)
plt.xscale('log')
# savefig('static/pop-hist.pdf',bbox_inches='tight')

# Draw 10x5 subplot with population grid for ALL cities
plt.close('all')
figure()
for count,place in enumerate(sorted_places):
	subplot(10,5,count+1)
	imshow(pop_grid[place],interpolation='None')
	ax = gca()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)
	title(place,fontsize=10)
tight_layout()
# savefig('static/50-cities-pop-grid.pdf',bbox_inches='tight')

# -----------------------
# Draw for a specific place
place = 'City of Bristol'

# Draw amenities inside boundary for ONE city
plt.close('all')
plt.figure(figsize=(10,8))
plt.plot(*b[place].boundary.xy)
markers = ['.','o','+','d','x',',']
colours = ['c','y','m','r','g','b']
alphas = [0.4,0.4,0.4,0.4,0.4,0.4]
for amenity,marker,colour,alpha in zip(critical,markers,colours,alphas):
	x,y = zip(*a[place][amenity])
	plt.scatter(x,y,c=colour,marker=marker,label=amenity,alpha=alpha)
plt.legend(fontsize=15,loc='lower left')
plt.xlim(-2.8,-2.45)
plt.ylim(51.35,51.55)
plt.xlabel('Longitude',fontsize=15)
plt.ylabel('Latitude',fontsize=15)
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig('static/city-of-bristol-amenities.pdf',bbox_inches='tight')

# Draw population as points for ONE city
plt.close('all')
plt.plot(*b[place].boundary.xy)
plt.scatter(p[place].lon,p[place].lat,c=p[place].pop)
plt.legend(fontsize=15,loc='upper left')
plt.xlabel('Longitude',fontsize=15)
plt.ylabel('Latitude',fontsize=15)
plt.savefig('static/city-of-bristol-pop.pdf',bbox_inches='tight')

# Draw the population grid for ONE city
plt.close('all')
x_min, x_max, y_min, y_max = min(p[place].lon), max(p[place].lon), min(p[place].lat), max(p[place].lat)
imshow(pop_grid[place],interpolation='None',extent=[x_min,x_max,y_min,y_max])
plt.plot(*b[place].boundary.xy)
plt.xlabel('Longitude',fontsize=15)
plt.ylabel('Latitude',fontsize=15)
cb = colorbar()
cb.label('Population count')
show()
# savefig('static/city-of-bristol-pop-grid.pdf',bbox_inches='tight')

# Draw 5*8 subplot with amenity grid for ONE city, ALL amenities
plt.close('all')
figure()
sigma = 0.0 # 0.0 or 1.5
for count,amenity in enumerate(amenities.index):
	subplot(8,5,count+1)
	imshow(am_grid[place][sigma][amenity],interpolation='None')
	ax = gca()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)
	title(amenity,fontsize=10)
tight_layout()
# savefig('static/city-of-bristol-40-am-grid-sigma-0.0.pdf',bbox_inches='tight')

# Draw 5*8 subplot with amenity grid for ONE city, ALL amenities
plt.close('all')
figure()
sigma = 1.5 # 0.0 or 1.5
for count,amenity in enumerate(amenities.index):
	subplot(8,5,count+1)
	imshow(am_grid[place][sigma][amenity],interpolation='None')
	ax = gca()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)
	title(amenity,fontsize=10)
tight_layout()
# savefig('static/city-of-bristol-40-am-grid-sigma-1.5.pdf',bbox_inches='tight')

# Draw 4x4 amenity subplot grid with increasing amenity sigma - ONE city, ONE amenity, ALL sigmas 
amenity = 'hospital'
plt.close('all')
for count,sigma in enumerate(sigmas):	
	subplot(4,4,count+1)
	imshow(am_grid[place][sigma][amenity],interpolation='None')
	ax = gca()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)	
	title('$\sigma = {}$'.format(sigma),fontsize=15)
tight_layout()
# savefig('static/city-of-bristol-hospital-am-grid-sigmas.pdf',bbox_inches='tight')

# Draw 4x4 amenity/person subplot grid with increasing amenity sigma - ONE city, ONE amenity, ALL sigmas 
amenity = 'hospital'
plt.close('all')
for count,sigma in enumerate(sigmas):	
	subplot(4,4,count+1)
	imshow(am_pop_grid[place][sigma][amenity],interpolation='None')
	ax = gca()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)	
	title('$\sigma = {}$'.format(sigma),fontsize=15)
tight_layout()
# savefig('static/city-of-bristol-hospital-am-pop-grid-sigmas.pdf',bbox_inches='tight')

# -----------------------------------
# Population amenity correlations	

# Pearson+Spearman r_pa for ONE city, ALL amenities and sigmas

place = 'City of Bristol'
vmax = max(ppa[ppa<1].loc[place].max().max(),spa[spa<1].loc[place].max().max())
plt.close('all')
subplot(1,2,1)
df = ppa.loc[place].transpose()
im = imshow(df,interpolation='none',vmin=0,vmax=vmax);
ax=gca();
ax.set_yticks(range(40));
ax.set_yticklabels(df.index,fontsize=10)
ax.set_xticks(range(0,16,5));
ax.set_xticklabels(df.keys()[::5])
# cb=colorbar();
title('Pearson $r_{pa}^2$',fontsize=15)
plt.xlabel('$\sigma$',fontsize=15)
subplot(1,2,2)
df = spa.loc[place].transpose()
imshow(df,interpolation='none',vmin=0,vmax=vmax);
ax=gca();
ax.yaxis.tick_right()
ax.set_yticks(range(40));
ax.set_yticklabels(df.index,fontsize=10)
ax.set_xticks(range(0,16,5));
ax.set_xticklabels(df.keys()[::5])
title('Spearman $\\rho_{pa}^2$',fontsize=15)
plt.xlabel('$\sigma$',fontsize=15)
# Shared colorbar
cax=gcf().add_axes([0.2, 0.08, 0.6, 0.04])
colorbar(im,cax,orientation='horizontal');
# savefig('static/pa-city-of-bristol-pearson+spearman.pdf',bbox_inches='tight')

# Median Pearson r_pa^2 for ALL amenity and ALL cities
plt.close('all')
subplot(2,1,1)
df = ppa.median(level=0).transpose()
imshow(df,interpolation='none',vmin=0)
ax = gca()
ax.set_yticks(range(40));
ax.set_yticklabels(df.index,fontsize=8)
ax.xaxis.tick_top()
ax.set_xticks(range(50));
ax.set_xticklabels(df.keys(),rotation=90,fontsize=8)
cb = colorbar()
cb.set_label('Median Pearson $r_{pa}^2$',fontsize=15)
# Median Spearman r_pa for ALL amenity and ALL cities
subplot(2,1,2)
df = spa.median(level=0).transpose()
imshow(df,interpolation='none',vmin=0)
ax = gca()
ax.set_yticks(range(40));
ax.set_yticklabels(df.index,fontsize=8)
ax.set_xticks(range(50));
ax.set_xticklabels(df.keys(),rotation=90,fontsize=8)
cb = colorbar()
cb.set_label('Median Spearman $\\rho_{pa}^2$',fontsize=15)
draw()
# savefig('static/pa-med-pearson+spearman-all-cities-amenities.pdf',bbox_inches='tight')

# Boxplot for ALL cities
rpl = list(reversed(sorted_places))
plt.close('all')
subplot(1,2,1)
df = ppa.transpose().stack()[rpl]
df.boxplot(vert=False)
plt.xlabel('Pearson $r_{pa}^2$',fontsize=15)
xlim(0,1)
subplot(1,2,2)
df = spa.transpose().stack()[rpl]
df.boxplot(vert=False)
plt.xlabel('Spearman $\\rho_{pa}^2$',fontsize=15)
xlim(0,1)
ax = gca()
ax.yaxis.tick_right()
# savefig('static/pa-boxplot-pearson+spearman-all-cities.pdf',bbox_inches='tight')

# Boxplot for ALL amenities
ram = list(reversed(amenities.index))
plt.close('all')
subplot(1,2,1)
df = ppa[ram]
df.boxplot(vert=False)
plt.xlabel('Pearson $r_{pa}^2$',fontsize=15)
xlim(0,1)
subplot(1,2,2)
df = spa[ram]
df.boxplot(vert=False)
plt.xlabel('Spearman $\\rho_{pa}^2$',fontsize=15)
xlim(0,1)
ax = gca()
ax.yaxis.tick_right()
# savefig('static/pa-boxplot-pearson+spearman-all-amenities.pdf',bbox_inches='tight')

# Q2(Mean(r_pa+rho_pa)) vs Total Amenity Count
plt.close('all')
subplot(2,1,1)
x = ppa.median()
y = amenities.sum(axis=1)
print ss.pearsonr(x,y)
print ss.spearmanr(x,y)
# plot(x, numpy.poly1d(numpy.polyfit(x, y, 1))(x),c='r')
scatter(x,y,c='r',linewidth=0,alpha=0.5,label='Median $r_{pa}^2$ vs. Amenity Count')
x = spa.median()
y = amenities.sum(axis=1)
print ss.pearsonr(x,y)
print ss.spearmanr(x,y)
# plot(x, numpy.poly1d(numpy.polyfit(x, y, 1))(x),c='b')
scatter(x,y,c='b',linewidth=0,alpha=0.5,label='Median $\\rho_{pa}^2$ vs. Amenity Count',marker='D')
legend(fontsize=12,loc='lower right',borderpad=1)
plt.xlabel('Median Coefficient of Determination', fontsize=15)
plt.ylabel('Total Amenity Count', fontsize=15)
xlim(0,0.6)
# ylim(-3000,15000)
yscale('log')
# Q2(Mean(r_pa+rho_pa)) vs Total Population Count
subplot(2,1,2)
x = ppa.transpose().stack().median()[list(sorted_places)]
y = sorted_pop
print ss.pearsonr(x,y)
print ss.spearmanr(x,y)
# plot(x, numpy.poly1d(numpy.polyfit(x, y, 1))(x),c='r')
scatter(x,y,c='r',linewidth=0,alpha=0.5,label='Median $r_{pa}^2$ vs. Population Count')
x = spa.transpose().stack().median()[list(sorted_places)]
y = sorted_pop
print ss.pearsonr(x,y)
print ss.spearmanr(x,y)
# plot(x, numpy.poly1d(numpy.polyfit(x, y, 1))(x),c='b')
scatter(x,y,c='b',linewidth=0,alpha=0.5,label='Median $\\rho_{pa}^2$ vs. Population Count',marker='D')
legend(fontsize=12,loc='lower right',borderpad=1)
plt.xlabel('Median Coefficient of Determination', fontsize=15)
plt.ylabel('Total Population Count', fontsize=15)
yscale('log')
xlim(0,0.6)
# ylim(-20000,1200000)
tight_layout()
# savefig('static/pa-vs-total-amenity+population-count.pdf',bbox_inches='tight')

# Critical amenities
p1 = (pearson_pa_med.transpose()-pearson_pa_mean.median(axis=1)).transpose()
s1 = (spearman_pa_med.transpose()-spearman_pa_mean.median(axis=1)).transpose()
p2 = pearson_pa_med-pearson_pa_med.median()
s2 = spearman_pa_med-spearman_pa_med.median()

plt.close('all')
# Relative difference from amenity medians
subplot(2,1,1)
imshow(p1,interpolation='none');colorbar()
# Relative difference from city medians
subplot(2,1,2)
imshow(p2,interpolation='none');colorbar()

plt.close('all')
# Relative difference from amenity medians
subplot(2,1,1)
imshow(s1,interpolation='none');colorbar()
# Relative difference from city medians
subplot(2,1,2)
imshow(s2,interpolation='none');colorbar()

# p*s
plt.close('all')
subplot(2,1,1)
imshow(p1*p2,interpolation='none');colorbar()
subplot(2,1,2)
imshow(s1*s2,interpolation='none');colorbar()

# -----------------------------------
# Inter amenity correlations

# Pearson r_aa for ONE city, ALL amenities, ALL sigmas - Draw 4x4 subplot grid for increasing amenity sigma
place = 'City of Bristol'
plt.close('all')
for count,sigma in enumerate(sigmas):
	subplot(4,4,count+1)
	df = paa.loc[place,str(sigma)]
	df[df==1] = NaN
	im = imshow(df,interpolation='none')
	ax = gca()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)	
	colorbar()
	title('$\sigma = {}$'.format(sigma),fontsize=15)
tight_layout()
# savefig('static/aa-pearson-city-of-bristol.pdf',bbox_inches='tight')

# Spearman r_aa for ONE city, ALL amenities, ALL sigmas - Draw 4x4 subplot grid for increasing amenity sigma
plt.close('all')
for count,sigma in enumerate(sigmas):
	subplot(4,4,count+1)
	df = saa.loc[place,str(sigma)]
	df[df==1] = NaN	
	im = imshow(df,interpolation='none')
	ax = gca()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)	
	colorbar()
	title('$\sigma = {}$'.format(sigma),fontsize=15)
tight_layout()
# savefig('static/aa-spearman-city-of-bristol.pdf',bbox_inches='tight')

# Median amenity amenity correlation
plt.close('all')
subplot(2,1,1)
df=paa[paa<1].median(level=2)
imshow(df,interpolation='none')
cb = colorbar()
cb.set_label('Median Pearson $r_{aa}^2$',fontsize=15)
ax=gca()
ax.set_yticks(range(40));
ax.set_yticklabels(df.keys(),fontsize=8)
ax.xaxis.tick_top()
ax.set_xticks(range(40));
ax.set_xticklabels(df.index,rotation=90,fontsize=8)
subplot(2,1,2)
df=saa[saa<1].median(level=2)
imshow(df,interpolation='none')
cb = colorbar()
cb.set_label('Median Spearman $\\rho_{aa}^2$',fontsize=15)
ax=gca()
ax.set_yticks(range(40));
ax.set_yticklabels(df.keys(),fontsize=8)
ax.set_xticks(range(40));
ax.set_xticklabels(df.index,rotation=90,fontsize=8)
# savefig('static/aa-med-pearson+spearman-all-cities.pdf',bbox_inches='tight')

# Median Pearson r_aa for ALL amenity and ALL cities
dfp = paa.median(level=0).reindex(sorted_places).transpose()
dfs = saa.median(level=0).reindex(sorted_places).transpose()
plt.close('all')
subplot(2,1,1)
imshow(dfp,interpolation='none',vmin=0)
ax = gca()
ax.set_yticks(range(40));
ax.set_yticklabels(dfp.index,fontsize=8)
ax.xaxis.tick_top()
ax.set_xticks(range(50));
ax.set_xticklabels(dfs.keys(),rotation=90,fontsize=8)
cb = colorbar()
cb.set_label('Median Pearson $r_{aa}^2$',fontsize=15)
# Median Spearman r_aa for ALL amenity and ALL cities
subplot(2,1,2)
imshow(dfs,interpolation='none',vmin=0)
ax = gca()
ax.set_yticks(range(40));
ax.set_yticklabels(dfs.index,fontsize=8)
ax.set_xticks(range(50));
ax.set_xticklabels(dfs.keys(),rotation=90,fontsize=8)
cb = colorbar()
cb.set_label('Median Spearman $\\rho_{aa}^2$',fontsize=15)
# savefig('static/aa-med-pearson+spearman-all-cities-amenities.pdf',bbox_inches='tight')

# Boxplot for ALL amenities
ram = list(reversed(amenities.index))
plt.close('all')
subplot(1,2,1)
df = paa[ram]
df.boxplot(vert=False)
plt.xlabel('Pearson $r_{aa}^2$',fontsize=15)
xlim(0,1)
subplot(1,2,2)
df = saa[ram]
df.boxplot(vert=False)
plt.xlabel('Spearman $\\rho_{aa}^2$',fontsize=15)
xlim(0,1)
ax = gca()
ax.yaxis.tick_right()
# savefig('static/aa-boxplot-pearson+spearman-all-amenities.pdf',bbox_inches='tight')

# Boxplot for ALL cities
rpl = list(reversed(sorted_places))
plt.close('all')
subplot(1,2,1)
df = paa.transpose().stack().stack()[rpl]
df.boxplot(vert=False)
plt.xlabel('Pearson $r_{aa}^2$',fontsize=15)
xlim(0,1)
subplot(1,2,2)
df = saa.transpose().stack().stack()[rpl]
df.boxplot(vert=False)
plt.xlabel('Spearman $\\rho_{aa}^2$',fontsize=15)
xlim(0,1)
ax = gca()
ax.yaxis.tick_right()
# savefig('static/aa-boxplot-pearson+spearman-all-cities.pdf',bbox_inches='tight')

# Q2(Mean(r_pa+rho_pa)) vs Total Amenity Count
plt.close('all')
subplot(2,1,1)
y = am_pop.sum(axis=1)
x = paa.median()
print ss.pearsonr(x,y)
print ss.spearmanr(x,y)
# plot(x, numpy.poly1d(numpy.polyfit(x, y, 1))(x),c='r')
scatter(x,y,c='r',linewidth=0,alpha=0.5,label='Median $r_{pa}^2$ vs. Mean Amenity/person')
x = saa.median()
print ss.pearsonr(x,y)
print ss.spearmanr(x,y)
# plot(x, numpy.poly1d(numpy.polyfit(x, y, 1))(x),c='b')
scatter(x,y,c='b',linewidth=0,alpha=0.5,label='Median $\\rho_{pa}^2$ vs. Mean Amenity/person',marker='D')
legend(fontsize=12,loc='upper right',borderpad=1)
plt.xlabel('Median Coefficient of Determination', fontsize=15)
plt.ylabel('Mean Amenity/person over Amenities', fontsize=15)
xlim(-0.1,0.6)
# ylim(-3000,15000)
yscale('log')
# Q2(Mean(r_pa+rho_pa)) vs Total Population Count
subplot(2,1,2)
y = am_pop.sum(axis=0)
x = paa.transpose().stack().stack().median()[list(sorted_places)]
print ss.pearsonr(x,y)
print ss.spearmanr(x,y)
# plot(x, numpy.poly1d(numpy.polyfit(x, y, 1))(x),c='r')
scatter(x,y,c='r',linewidth=0,alpha=0.5,label='Median $r_{pa}^2$ vs. Mean Amenity/person')
x = saa.transpose().stack().stack().median()[list(sorted_places)]
print ss.pearsonr(x,y)
print ss.spearmanr(x,y)
# plot(x, numpy.poly1d(numpy.polyfit(x, y, 1))(x),c='b')
scatter(x,y,c='b',linewidth=0,alpha=0.5,label='Median $\\rho_{pa}^2$ vs. Mean Amenity/person',marker='D')
legend(fontsize=12,loc='lower right',borderpad=1)
plt.xlabel('Median Coefficient of Determination', fontsize=15)
plt.ylabel('Mean Amenity/person over Cities', fontsize=15)
yscale('log')
xlim(-0.1,0.6)
# ylim(-20000,1200000)
tight_layout()
# savefig('static/aa-vs-total-amenity+population-count.pdf',bbox_inches='tight')

