"""
Driver functions for the rvsearch pipeline.\
These functions are meant to be used only with\
the `cli.py` command line interface.
"""
from __future__ import print_function
import os
import copy
import pandas as pd
import pickle

import radvel
from radvel.utils import working_directory
import rvsearch


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
        post = radvel.fitting.maxlike_fitting(post, verbose=True)
    else:
        post = None

    searcher = rvsearch.search.Search(data, starname=starname,
                                      min_per=args.minP,
                                      workers=args.num_cpus,
                                      post=post,
                                      trend=args.trend,
                                      verbose=args.verbose)
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
                                                 num_sim=args.num_inject,
                                                 full_grid=args.full_grid,
                                                 verbose=args.verbose)
                recoveries = inj.run_injections(num_cpus=args.num_cpus)
                inj.save()
            except IOError:
                print("WARNING: Problem with {}".format(sfile))
                os._exit(1)
        else:
            recoveries = pd.read_csv('recoveries.csv')


def plots(args):
    """
    Generate plots

    Args:
        args (ArgumentParser): command line arguments
    """
    sdir = args.search_dir

    with working_directory(sdir):
        sfile = os.path.abspath('search.pkl')
        run_name = sfile.split('/')[-2]
        if not os.path.exists(sfile):
            print("No search file found in {}".format(sdir))
            os._exit(1)
        else:
            searcher = pickle.load(open(sfile, 'rb'))

        for ptype in args.type:
            print("Creating {} plot for {}".format(ptype, run_name))

            if ptype == 'recovery':
                rfile = os.path.abspath('recoveries.csv')
                if not os.path.exists(rfile):
                    print("No recovery file found in {}".format(sdir))
                    os._exit(1)

                xcol = 'inj_au'
                ycol = 'inj_msini'
                xlabel = '$a$ [AU]'
                ylabel = r'M$\sin{i_p}$ [M$_\oplus$]'
                print("Plotting {} vs. {}".format(ycol, xcol))

                mstar = args.mstar

                comp = rvsearch.Completeness.from_csv(rfile, xcol=xcol,
                                                      ycol=ycol, mstar=mstar)
                cplt = rvsearch.plots.CompletenessPlots(comp)

                fig = cplt.completeness_plot(title=run_name,
                                             xlabel=xlabel,
                                             ylabel=ylabel)

                saveto = os.path.join(run_name+'_recoveries.{}'.format(args.fmt))

                fig.savefig(saveto)
                print("Recovery plot saved to {}".format(
                      os.path.abspath(saveto)))

            if ptype == 'summary':
                plotter = rvsearch.plots.PeriodModelPlot(searcher,
                            saveplot='{}_summary.{}'.format(searcher.starname, args.fmt))
                plotter.plot_summary()
