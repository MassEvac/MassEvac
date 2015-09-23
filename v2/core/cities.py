# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import os, shutil

# <codecell>

# Get the list of cities that have been enlisted
file = open("../OSM/urban/cities.txt", "r")
all = file.read().split('\n')
file.close()

threshold = 15

# Best estimate of population multiplier
# Assumes that population growth spread is uniform across the UK\
p2000 = 58.459
# Source: http://en.wikipedia.org/wiki/List_of_countries_by_population_in_2000
p2015 = 63.935
# Source: http://en.wikipedia.org/wiki/List_of_countries_by_future_population_(Medium_variant)
p_factor = p2015/p2000

ignore = []

# Ignore the cities in the ignore list
for i in ignore:
    try:
        all.remove(i)
    except:
        print 'We do not have data for {0}, remove from ignore list.'.format(i)

# <codecell>

def seemingly_done(sim,detail=False):
    # Get the list of cities that have a simulation folder created and seemingly done
    seemingly_done = os.listdir('abm/{0}'.format(sim))

    if detail:
        for d in seemingly_done:
            print (d,len(os.listdir('abm/{0}/{1}'.format(sim,d))))

    return seemingly_done

# <codecell>

def interrupted(sim, threshold=threshold):
    # Using threshold for the minimum number of files created in a folder after the simulation
    return [d for d in seemingly_done(sim) if len(os.listdir('abm/{0}/{1}'.format(sim,d))) < threshold]

# <codecell>

def done(sim, threshold=threshold, delete_interrupted=False):
    done = seemingly_done(sim)

    for i in interrupted(sim, threshold):
        # Interrupted counts as not done so remove from list of done
        done.remove(i)

        # Delete the interrupted simulation folders
        if delete_interrupted:
            try:
                shutil.rmtree('abm/{0}/{1}'.format(sim,i))
            except Exception as e:
                print 'abm/{0}/{1} folder could not be deleted.'.format(i)

    return done

# <codecell>

def not_done(sim, threshold=threshold, delete_interrupted=False):

    not_done = all[:]

    for d in done(sim, threshold, delete_interrupted):
        try:
            not_done.remove(d)
        except ValueError:
            pass

    return not_done
