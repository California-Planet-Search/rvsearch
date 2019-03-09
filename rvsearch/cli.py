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
    # psr_parent.add_argument(
    #     '-d', dest='outputdir', type=str,
    #     help="Working directory. Default is the same as the \
    #     configuration file (without .py)", default=None
    # )
    # psr_parent.add_argument('-s','--setup',
    #     dest='setupfn', type=str,
    #     help="Setup file."
    # )

    # Search
    # psr_peri = subpsr.add_parser('search', parents=[psr_parent], )
    # psr_peri.add_argument('-t', '--type',
    #                       type=str, nargs='+',
    #                       choices=rvsearch.periodogram.VALID_TYPES,
    #                       help="type of periodogram(s) to calculate"
    #                       )
    # psr_peri.add_argument('--minP',
    #                       type=float, action='store', default=1.2,
    #                       help="Minimum search period [default=1.2]"
    #                       )
    # psr_peri.add_argument('--maxP',
    #                       type=float, action='store', default=1e4,
    #                       help="Maximum search period [default=10000]"
    #                       )
    # psr_peri.add_argument('--num_known',
    #                       type=int, action='store', default=0,
    #                       help="Number of previously known planets [default=0]"
    #                       )
    # psr_peri.add_argument('--num_freqs',
    #                       action='store', default=None,
    #                       help="Number of test frequencies"
    #                       )


    # Injections
    psr_peri = subpsr.add_parser('inject', parents=[psr_parent], )
    psr_peri.add_argument('search_dir', metavar='search directory',
                          type=str,
                          help="path to existing radvel-search output directory"
                          )
    psr_peri.add_argument('--minP',
                          type=float, action='store', default=1.2,
                          help="Minimum injection period [default=1.2]"
                          )
    psr_peri.add_argument('--maxP',
                          type=float, action='store', default=1e6,
                          help="Maximum injection period [default=1e6]"
                          )
    psr_peri.add_argument('--minK',
                          type=float, action='store', default=0.1,
                          help="Minimum injection K [default=0.0]"
                          )
    psr_peri.add_argument('--maxK',
                          type=float, action='store', default=1000.0,
                          help="Maximum injection K [default=1000.0]"
                          )
    psr_peri.add_argument('--minE',
                          type=float, action='store', default=0.0,
                          help="Minimum injection eccentricity [default=0.0]"
                          )
    psr_peri.add_argument('--maxE',
                          type=float, action='store', default=0.0,
                          help="Maximum injection eccentricity [default=0.0]"
                          )

    psr_peri.add_argument('--num_inject',
                          type=int, action='store', default=100,
                          help="Number of injections [default=100]"
                          )
    psr_peri.add_argument('--num_cpus',
                          action='store', default=8,
                          help="Number of CPUs [8]"
                          )

    psr_peri.set_defaults(func=rvsearch.driver.injections)


    args = psr.parse_args()

    args.func(args)


if __name__ == '__main__':
    main()
