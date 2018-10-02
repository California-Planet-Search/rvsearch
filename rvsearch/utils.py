#Utilities for loading data, checking for known planets, etc.
import numpy as np
import pandas as pd

import radvel

def read_from_csv(filename, verbose=True):
    data = pd.DataFrame.from_csv(filename)
    if 'tel' not in data.columns:
        if verbose:
            print('Telescope type not given, defaulting to HIRES.')
        data['tel'] = 'HIRES'
        #Question: DO WE NEED TO CONFIRM VALID TELESCOPE TYPE?
    return data

def read_from_arrs(t, mnvel, errvel, tel=None, verbose=True):
    data = pd.DataFrame()
    data['time'], data['mnvel'], data['errvel'] = t, mnvel, errvel
    if tel == None:
        if verbose:
            print('Telescope type not given, defaulting to HIRES.')
        data['tel'] = 'HIRES'
    else:
        data['tel'] = tel

def read_from_vst(filename, verbose=True):
