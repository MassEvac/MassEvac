def genfig(key):
	# Draw the image
	directory = 'analysis/all_weighted_density_fig'
	try:
		os.makedirs(directory)
	except OSError:
		pass

	plt.figure()

	plt.imshow(all_weighted_density[key].T,interpolation='none',cmap='Reds',aspect=0.1)

	# Draw the trajectory
	for k in s.tracked_agent.keys():
		if x == s.X[k]:
			agent_index = k

	yax = s.tracked_agent[agent_index]	
	xax = range(len(yax))

	plt.plot(xax,yax,':',label='Randomly picked agent trajectory',linewidth=5,c='lime')

	plt.xlim(0,len(all_weighted_density[key]))
	plt.ylim(0,len(all_weighted_density[key][0]))	

	# Label and save the figure
	fontsize = 20
	plt.xlabel('$\mathrm{T \ [s]}$',fontsize=fontsize)
	plt.ylabel('$\mathrm{D \ [m]}$',fontsize=fontsize)

	xticks = range(0,len(all_weighted_density[key]),200)
	xticklabs = np.array(xticks)*60

	plt.xticks(xticks,xticklabs)

	print len(xax)

	cbar = plt.colorbar()
	
	cbar.set_label('$\mathrm{k \ [ped/m^2]}$',fontsize=fontsize)

	# plt.gca().tight_layout()

	plt.legend()

	plt.savefig('{0}/{1}.pdf'.format(directory,key))

x = 61116
genfig(x)