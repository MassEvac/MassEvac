import db

reload(db)

# Test
f = db.Flood('/Users/bharatkunwar/Desktop/flood/1.tif')

print f.floodStatus((f.maxx,f.maxy),(f.minx,f.miny))
print f.isFlooded((f.maxx,f.maxy),(f.minx,f.miny))

plt.ion()
f.fig()

h = db.Highway((f.maxx,f.maxy,f.minx,f.miny))

