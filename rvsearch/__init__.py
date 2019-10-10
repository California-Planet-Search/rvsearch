import os
import sys

from .utils import *
from .periodogram import *
from .search import *
from .plots import *
from .inject import *

__version__ = '0.1.2'

DATADIR = os.path.join(sys.prefix, 'rvsearch_example_data')
