"""
Command Line Interface
"""
import os
from argparse import ArgumentParser
import warnings

import rvsearch.driver

warnings.simplefilter("ignore")
warnings.simplefilter('once', DeprecationWarning)

def main():
    psr = ArgumentParser(
        description="RadVel-Search: Automated planet detection pipeline", prog='RadVel-Search'
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
    psr_parent.add_argument(
        '-d', dest='outputdir', type=str,
        help="Working directory. Default is the same as the \
        configuration file (without .py)", default=None
    )
    psr_parent.add_argument('-s','--setup',
        dest='setupfn', type=str, 
        help="Setup file."
    )

    # Periodograms
    psr_peri = subpsr.add_parser('peri', parents=[psr_parent], )
    psr_peri.add_argument('-t', '--type',
                          type=str, nargs='+',
                          choices=['bic'],
                          help="type of periodogram(s) to calculate"
                          )
    psr_peri.add_argument('--minP',
                          type=float, action='store', default=1.2,
                          help="Minimum search period [default=1.2]"
                          )
    psr_peri.add_argument('--maxP',
                          type=float, action='store', default=1e4,
                          help="Maximum search period [default=10000]"
                          )
    psr_peri.add_argument('--num_known',
                          type=int, action='store', default=0,
                          help="Number of previously known planets [default=0]"
                          )


    psr_peri.set_defaults(func=rvsearch.driver.periodograms)

    # Plotting
    # psr_plot = subpsr.add_parser('plot', parents=[psr_parent],)
    # psr_plot.add_argument('-t','--type',
    #     type=str, nargs='+',
    #     choices=['rv','corner','trend', 'derived'],
    #     help="type of plot(s) to generate"
    # )
    # psr_plot.add_argument(
    #     '--plotkw', dest='plotkw',action='store', default="{}", type=eval,
    #     help='''
    #     Dictionary of keywords sent to MultipanelPlot or GPMultipanelPlot.
    #     E.g. --plotkw "{'yscale_auto': True}"'
    #     ''',
    # )
    # psr_plot.add_argument('--gp',
    # dest='gp',
    # action='store_true',
    # default=False,
    # help="Make a multipanel plot with GP bands. For use only with GPLikleihood objects"
    # )
    #
    # psr_plot.set_defaults(func=radvel.driver.plots)


    # Derive physical parameters
    # psr_physical = subpsr.add_parser(
    #     'derive', parents=[psr_parent],
    #     description="Multiply MCMC chains by physical parameters. MCMC must"
    #     + "be run first"
    # )
    #
    # psr_physical.set_defaults(func=radvel.driver.derive)
    
    # # Information Criteria comparison (BIC/AIC)
    # psr_ic = subpsr.add_parser('ic', parents=[psr_parent],)
    # psr_ic.add_argument('-t',
    #     '--type', type=str, nargs='+', default='trend',
    #     choices=['nplanets', 'e', 'trend', 'jit', 'gp'],
    #     help="parameters to include in BIC/AIC model comparison"
    # )
    #
    # psr_ic.add_argument('-m',
    #     '--mixed', dest='mixed', action='store_true' ,
    #     help="flag to compare all models with the fixed parameters mixed and matched rather than"\
    #         + " treating each model comparison separately. This is the default. "\
    # )
    # psr_ic.add_argument('-u',
    #     '--un-mixed', dest='mixed', action='store_false',
    #     help="flag to treat each model comparison separately (without mixing them) "\
    #         + "rather than comparing all models with the fixed parameters mixed and matched."
    # )
    # psr_ic.add_argument('-f',
    #     '--fixjitter', dest='fixjitter', action='store_true',
    #     help="flag to fix the stellar jitters at the nominal model best-fit value"
    # )
    # psr_ic.add_argument('-n',
    #     '--no-fixjitter', dest='fixjitter', action='store_false',
    #     help="flag to let the stellar jitters float during model comparisons (default)"
    # )
    # psr_ic.add_argument('-v',
    #     '--verbose', dest='verbose', action='store_true',
    #     help="Print some more detail"
    # )
    # psr_ic.set_defaults(func=radvel.driver.ic_compare, fixjitter=False, unmixed=False,\
    #                     mixed=True)
    #
    # # Tables
    # psr_table = subpsr.add_parser('table', parents=[psr_parent],)
    # psr_table.add_argument('-t','--type',
    #     type=str, nargs='+',
    #     choices=['params', 'priors', 'rv', 'ic_compare'],
    #     help="type of plot(s) to generate"
    # )
    # psr_table.add_argument(
    #     '--header', action='store_true',
    #     help="include latex column header. Default just prints data rows"
    # )
    # psr_table.add_argument('--name_in_title',
    # dest='name_in_title',
    # action='store_true',
    # default=False,
    # help='''
    #     Include star name in table headers. Default just prints
    #     descriptive titles without star name [False]
    # '''
    # )
    #
    # psr_table.set_defaults(func=radvel.driver.tables)
    #
    #
    # # Report
    # psr_report = subpsr.add_parser(
    #     'report', parents=[psr_parent],
    #     description="Merge output tables and plots into LaTeX report"
    # )
    # psr_report.add_argument(
    #     '--comptype', dest='comptype', action='store',
    #     default='ic', type=str,
    #     help='Type of model comparison table to include. \
    #     Default: ic')
    #
    # psr_report.add_argument(
    #     '--latex-compiler', default='pdflatex', type=str,
    #     help='Path to latex compiler'
    #     )
    #
    # psr_report.set_defaults(func=radvel.driver.report)
    
    args = psr.parse_args()

    if args.outputdir is None:
        setupfile = args.setupfn
        print(setupfile)
        system_name = os.path.basename(setupfile).split('.')[0]
        outdir = os.path.join('./', system_name)
        args.outputdir = outdir
            
    if not os.path.isdir(args.outputdir):
        os.mkdir(args.outputdir)
        
    args.func(args)


if __name__ == '__main__':
    main()
