#!/usr/bin/env python

import os
import sys
from glob import glob

from radvel.utils import working_directory

search_list = sorted(glob('./*/search.pkl'))

args = ' '.join(sys.argv[1:])

for sfile in search_list:
    sdir = os.path.abspath(os.path.dirname(sfile))
    sfile = os.path.abspath(sfile)

    with working_directory(sdir):
        if not os.path.exists('recoveries.csv'):
            print("rvsearch inject {} {}".format(args, sdir))
