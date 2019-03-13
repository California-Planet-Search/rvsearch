"""
Command Line Interface
"""
import os
from argparse import ArgumentParser
import warnings

import rvsearch
import rvsearch.driver
import rvsearch.periodogram

warnings.simplefilter("ignore")
warnings.simplefilter('once', DeprecationWarning)


def main():
    psr = ArgumentParser(
        description="RadVel-Search: Automated planet detection pipeline", prog='rvsearch'
    )
    psr.add_argument('--version',
        action='version',
        version="%(prog)s {}".format(rvsearch.__version__),
        help="Print version number and exit."
    )

    subpsr = psr.add_subparsers(title="subcommands", dest='subcommand')

    # In the parent parser, we define arguments and options common to
    # all subcommands.
    psr_parent = ArgumentParser(add_help=False)
    psr_parent.add_argument('--num_cpus',
                            action='store', default=8, type=int,
                            help="Number of CPUs [8]")

    # Search
    psr_search = subpsr.add_parser('search', parents=[psr_parent], )
    psr_search.add_argument('setupfn', metavar="RadVel setup file", type=str,
                            help="Path to RadVel setup file.")
    psr_search.add_argument('--minP',
                          type=float, action='store', default=1.2,
                          help="Minimum search period [default=1.2]"
                          )
    # psr_search.add_argument('--maxP',
    #                       type=float, action='store', default=1e4,
    #                       help="Maximum search period [default=10000]"
    #                       )
    psr_search.add_argument('--known', action='store_true',
                          help="Include previously known planets [default=False]"
                          )
    psr_search.add_argument('--num_freqs',
                          action='store', default=None,
                          help="Number of test frequencies"
                          )


    # Injections
    psr_inj = subpsr.add_parser('inject', parents=[psr_parent], )
    psr_inj.add_argument('search_dir', metavar='search directory',
                          type=str,
                          help="path to existing radvel-search output directory"
                          )
    psr_inj.add_argument('--minP',
                          type=float, action='store', default=1.2,
                          help="Minimum injection period [default=1.2]"
                          )
    psr_inj.add_argument('--maxP',
                          type=float, action='store', default=1e6,
                          help="Maximum injection period [default=1e6]"
                          )
    psr_inj.add_argument('--minK',
                          type=float, action='store', default=0.1,
                          help="Minimum injection K [default=0.0]"
                          )
    psr_inj.add_argument('--maxK',
                          type=float, action='store', default=1000.0,
                          help="Maximum injection K [default=1000.0]"
                          )
    psr_inj.add_argument('--minE',
                          type=float, action='store', default=0.0,
                          help="Minimum injection eccentricity [default=0.0]"
                          )
    psr_inj.add_argument('--maxE',
                          type=float, action='store', default=0.0,
                          help="Maximum injection eccentricity [default=0.0]"
                          )

    psr_inj.add_argument('--num_inject',
                          type=int, action='store', default=100,
                          help="Number of injections [default=100]"
                          )

    psr_search.set_defaults(func=rvsearch.driver.run_search)
    psr_inj.set_defaults(func=rvsearch.driver.injections)

    args = psr.parse_args()

    args.func(args)


if __name__ == '__main__':
    main()
