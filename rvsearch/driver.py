"""
Driver functions for the rvsearch pipeline.\
These functions are meant to be used only with\
the `cli.py` command line interface.
"""
from __future__ import print_function
import os
from glob import glob
import pylab as pl
import pandas as pd

import rvsearch
from radvel.utils import working_directory


def injections(args):
    """Injection-recovery tests

    Args:
        args (ArgumentParser): command line arguments
    """

    plim = (args.minP, args.maxP)
    klim = (args.minK, args.maxK)
    elim = (args.minE, args.maxE)

    sdir = args.search_dir

    with working_directory(sdir):
        sfile = 'search.pkl'
        sfile = os.path.abspath(sfile)
        if not os.path.exists(sfile):
            print("No search file found in {}".format(sdir))
            os._exit(1)

        if not os.path.exists('recoveries.csv'):
            try:
                inj = rvsearch.inject.Injections(sfile, plim, klim, elim,
                                                 num_sim=args.num_inject)
                recoveries = inj.run_injections(num_cpus=args.num_cpus)
                inj.save()
            except OSError:
                print("WARNING: Problem with {}".format(sfile))
                os._exit(1)
        else:
            recoveries = pd.read_csv('recoveries.csv')

        pl.clf()
        fig = rvsearch.inject.plot_recoveries(recoveries)
        fig.savefig('recoveries.pdf')

