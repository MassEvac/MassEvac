import abm
p = abm.Places('bristol25')

for name in p.names:
	directory = 'abm/bristol25/' + name
	files=os.listdir(directory)
	for f in files:
		s = f.split('.')
		if 'DX' in s:
			n = s[0]
			d = s[1]
			os.mkdir('{0}/{1}.{2}'.format(directory,n,d))
			os.mkdir('{0}/{1}.{2}/ia'.format(directory,n,d))
			os.rename('{0}/{1}.{2}.DX'.format(directory,n,d),'{0}/{1}.{2}/DX'.format(directory,n,d))
			os.rename('{0}/{1}.{2}.agents'.format(directory,n,d),'{0}/{1}.{2}/agents'.format(directory,n,d))
			os.rename('{0}/ia.{1}.{2}.mp4'.format(directory,n,d),'{0}/{1}.{2}/ia/video.mp4'.format(directory,n,d))
			os.rename('{0}/ia.{1}.{2}.T'.format(directory,n,d),'{0}/{1}.{2}/ia/T'.format(directory,n,d))		