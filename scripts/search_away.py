 # Script to run RVSearch on a given dataset.

import os
import sys

import numpy as np
import pandas as pd
import radvel

import cpsutils
from cpsutils import io
import rvsearch
from rvsearch import utils, search, periodogram

'''
Bin in 12-hour intervals
Parallelize
Run mcmc
'''

starname = sys.argv[1]
data = io.loadcps(starname, apf=True, lick=True, ctslim=3000, verbose=False)
tels = np.unique(data['tel'].values)
for tel in tels:
    if len(data.loc[data['tel'] == tel]) < 5:
        data = data[data.tel != tel]
        data = data.reset_index()

# Bin data in 12-hour intervals.
if 'time' in data.columns:
	t = data['time'].values
	tkey = 'time'
elif 'jd' in data.columns:
	t = data['jd'].values
	tkey = 'jd'
time, mnvel, errvel, tel = radvel.utils.bintels(t, data['mnvel'].values,
                                                data['errvel'].values,
						data['tel'].values,
						binsize=0.5)
bin_dict = {tkey:time, 'mnvel':mnvel, 'errvel':errvel, 'tel':tel}
data = pd.DataFrame(data=bin_dict)
        
searcher = search.Search(data, starname=starname, min_per=1.5, workers=16, verbose=True)
searcher.run_search()
