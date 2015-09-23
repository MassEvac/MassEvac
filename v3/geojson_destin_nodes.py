import db
reload(db)
h=db.Highway('City of Bristol')
properties = {	40174: {'color': '#FF0000', 'textbg': '#FF0000', 'label': 'CA01'}, # this
				61116: {'color': '#00FF00', 'textbg': '#00FF00', 'label': 'CA02'}, # this
				39778: {'color': '#0000FF', 'textbg': '#0000FF','label': 'CA03'}, # this
				32741: {'color': '#770077', 'textbg': '#FFFFFF'},				
				19678: {'color': '#BBBB00', 'textbg': '#FFFFFF'},
				1021:  {'color': '#00BBBB', 'textbg': '#FFFFFF'},
				8888:  {'color': '#BB00BB', 'textbg': '#FFFFFF'},
				52523: {'color': '#BB0000', 'textbg': '#FFFFFF'},
				6186:  {'color': '#00BB00', 'textbg': '#FFFFFF'},
				17232: {'color': '#0000BB', 'textbg': '#FFFFFF'},
				47968: {'color': '#777700', 'textbg': '#FFFFFF'},
				49325: {'color': '#007777', 'textbg': '#FFFFFF'},
				15318: {'color': '#770000', 'textbg': '#FFFFFF'},
				16547: {'color': '#007700', 'textbg': '#FFFFFF'},
				6037:  {'color': '#000077', 'textbg': '#FFFFFF'},
			}
fname = '{0}/destins.json'.format(s.agents_file())
h.geojson_nodes(fname,properties)