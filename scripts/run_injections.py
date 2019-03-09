#!/usr/bin/env python

import os
from glob import glob
import pylab as pl
import pandas as pd

import rvsearch
from radvel.utils import working_directory

search_list = sorted(glob('./*/search.pkl'))

plim = (1, 1e6)
klim = (0.1, 1000)
elim = (0, 0)

for sfile in search_list:
    sdir = os.path.abspath(os.path.dirname(sfile))
    sfile = os.path.abspath(sfile)
    print(sfile)

    with working_directory(sdir):
        if not os.path.exists('recoveries.csv'):
            # try:
            inj = rvsearch.inject.Injections(sfile, plim, klim, elim, num_sim=100)
            recoveries = inj.run_injections(num_cpus=25)

            # except OSError:
            #     print("WARNING: Problem with {}".format(sfile))
            #     continue
        else:
            recoveries = pd.read_csv('recoveries.csv')
            
            pl.clf()
            fig = rvsearch.inject.plot_recoveries(recoveries)
            fig.savefig('recoveries.pdf')

