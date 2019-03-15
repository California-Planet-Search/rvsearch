"""
Driver functions for the rvsearch pipeline.\
These functions are meant to be used only with\
the `cli.py` command line interface.
"""
from __future__ import print_function
import os
from glob import glob
import copy
import pylab as pl
import pandas as pd

import rvsearch
import radvel
from radvel.utils import working_directory


def run_search(args):
    """Run a search from a given RadVel setup file

    Args:
        args (ArgumentParser): command line arguments
    """

    config_file = args.setupfn
    conf_base = os.path.basename(config_file).split('.')[0]

    P, post = radvel.utils.initialize_posterior(config_file)

    starname = conf_base
    data = P.data

    if args.known:
        ipost = copy.deepcopy(post)
        ipost = radvel.fitting.maxlike_fitting(post, verbose=True)
    else:
        ipost = None

    searcher = rvsearch.search.Search(data, starname=starname,
                                      min_per=args.minP,
                                      workers=args.num_cpus,
                                      post=ipost,
                                      verbose=True)
    searcher.run_search()


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

