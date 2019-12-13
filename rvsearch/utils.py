"""Utilities for loading data, checking for known planets, etc."""

import numpy as np
import scipy.special as spec
from astropy import constants as c
import pandas as pd
import radvel
try:
    import cpsutils
    from cpsutils import io
except:
    RuntimeError()


"""Functions for posterior modification (resetting, intializing, etc.)
"""

def GaussianDiffFunc(inp_list):
    """Function to use in the HIRES gamma offset prior.
    Args:
        inp_list(list): pair of floats, the difference of which we want to
                        constrain with a Gaussian prior. Hard-coded params,
                        derived empirically from HIRES analysis.

    """
    x     = inp_list[1] - inp_list[0]
    mu    = 0.#inp_list[2]
    sigma = 2.#inp_list[3]
    return -0.5 * ((x - mu) / sigma)**2 - 0.5*np.log((sigma**2)*2.*np.pi)

def reset_params(post, default_pdict):
    # Reset post.params values to default values
    for k in default_pdict.keys():
        post.params[k].value = default_pdict[k]
    return post

def insolate(T, R, a):
    # Calculate stellar insolation.
    return (T/5778)**4 * R**2 * a**-2

def tequil(S, alb=0.3):
    # Calculate equilibrium temperature.
    return S**-0.25 * ((1-alb)/4.)**0.25

def initialize_default_pars(instnames=['inst'], times=None, linear=True,
                            fitting_basis='per tc secosw sesinw k'):
    """Set up a default Parameters object.

    None of the basis values are free params, for the initial 0-planet fit.
    Remember to reset .vary to True for all relevant params.

    Args:
        instnames (list): codes of instruments used
        times (array): optional, timestamps of observations.
        linear (bool): Determine whether to optimize gammas linearly.
        fitting_basis: optional

    Returns:
        Parameters object
    """

    anybasis_params = radvel.Parameters(num_planets=1, basis='per tc e w k')

    if times is None:
        anybasis_params['tc1'] = radvel.Parameter(value=2455200.0)
    else:
        anybasis_params['tc1'] = radvel.Parameter(value=np.median(times))
    anybasis_params['w1'] = radvel.Parameter(value=np.pi/2.)
    anybasis_params['k1'] = radvel.Parameter(value=0.0)
    anybasis_params['e1'] = radvel.Parameter(value=0.0)
    anybasis_params['per1'] = radvel.Parameter(value=100.0)

    anybasis_params['dvdt'] = radvel.Parameter(value=0.0)
    anybasis_params['curv'] = radvel.Parameter(value=0.0)

    for inst in instnames:
        if linear:
            anybasis_params['gamma_'+inst] = radvel.Parameter(value=0.0,
                                                              linear=True,
                                                              vary=False)
        else:
            anybasis_params['gamma_'+inst] = radvel.Parameter(value=0.0)
        anybasis_params['jit_'+inst] = radvel.Parameter(value=2.0)

    params = anybasis_params.basis.to_any_basis(anybasis_params, fitting_basis)

    params['secosw1'].vary = False
    params['sesinw1'].vary = False
    params['per1'].vary = False

    return params


def initialize_post(data, params=None, priors=[], linear=True, decorrs=None):
    """Initialize a posterior object with data, params, and priors.
    Args:
        data: a pandas dataframe.
        params: a list of radvel parameter objects.
        priors: a list of priors to place on the posterior object.
        decorrs: a list of decorrelation vectors.
    Returns:
        post (radvel Posterior object)

    """

    if params is None:
        # params = radvel.Parameters(1, basis='per tc secosw sesinw logk')
        params = initialize_default_pars(instnames=data.tel, times=data.time)
    iparams = radvel.basis._copy_params(params)

    # Allow for time to be listed as 'time' or 'jd' (Julian Date).
    if {'jd'}.issubset(data.columns):
        data['time'] = data['jd']

    # initialize RVModel
    time_base = np.mean([data['time'].max(), data['time'].min()])
    mod = radvel.RVModel(params, time_base=time_base)

    # initialize Likelihood objects for each instrument
    telgrps = data.groupby('tel').groups
    likes = {}

    for inst in telgrps.keys():
        # 10/8: ADD DECORRELATION VECTORS AND VARS, ONLY FOR SELECTED INST.
        likes[inst] = radvel.likelihood.RVLikelihood(
            mod, data.iloc[telgrps[inst]].time, data.iloc[telgrps[inst]].mnvel,
            data.iloc[telgrps[inst]].errvel, suffix='_'+inst)

        likes[inst].params['gamma_'+inst] = iparams['gamma_'+inst]
        likes[inst].params['jit_'+inst] = iparams['jit_'+inst]
    # Can this be cleaner? like = radvel.likelihood.CompositeLikelihood(likes)
    like = radvel.likelihood.CompositeLikelihood(list(likes.values()))

    post = radvel.posterior.Posterior(like)
    if priors == []:
        priors.append(radvel.prior.PositiveKPrior(post.params.num_planets))
        priors.append(radvel.prior.EccentricityPrior(post.params.num_planets))

        if not linear:
            if ('j' in telgrps.keys()) and ('k' in telgrps.keys()):
                TexStr = 'Gaussian Prior on HIRES offset'
                OffsetPrior = radvel.prior.UserDefinedPrior(['gamma_j', 'gamma_k'],
                                                            GaussianDiffFunc,
                                                            TexStr)
                priors.append(OffsetPrior)
        #for inst in telgrps.keys():
        #    priors.append(radvel.prior.Jeffrey('jit_'+inst, 0.01, 20.0))
    post.priors = priors

    return post


def window(times, freqs, plot=False):
    """Function to generate, and plot, the window function of observations.

    Args:
        time: times of observations in a dataset. FOR SEPARATE TELESCOPES?

    """
    W = np.zeros(len(freqs))
    for i, freq in enumerate(freqs):
        W[i] = np.absolute(np.sum(np.exp(-2*np.pi*1j*times*freq)))
    W /= float(len(times))
    return W

def read_from_csv(filename, binsize=0.0, verbose=True):
    """Read radial velocity data from a csv file into a Pandas dataframe.

    Args:
        filename (string): Path to csv file
        binsize (float): Times in which to bin data, in given units
        verbose (bool): Notify user if instrument types not given?

    """
    data = pd.read_csv(filename)
    if 'tel' not in data.columns:
        if verbose:
            print('Instrument types not given.')
        data['tel'] = 'Inst'
    if binsize > 0.0:
        if 'time' in data.columns:
            t = data['time'].values
            tkey = 'time'
        elif 'jd' in data.columns:
            t = data['jd'].values
            tkey = 'jd'
        else:
            raise ValueError('Incorrect data input.')
        time, mnvel, errvel, tel = radvel.utils.bintels(t, data['mnvel'].values,
                                                        data['errvel'].values,
                                                        data['tel'].values,
                                                        binsize=binsize)
        bin_dict = {tkey: time, 'mnvel': mnvel,
                    'errvel': errvel, 'tel': tel}
        data = pd.DataFrame(data=bin_dict)

    return data


def read_from_arrs(t, mnvel, errvel, tel=None, verbose=True):
    data = pd.DataFrame()
    data['time'], data['mnvel'], data['errvel'] = t, mnvel, errvel
    if tel == None:
        if verbose:
            print('Instrument type not given.')
        data['tel'] = 'Inst'
    else:
        data['tel'] = tel
    return data


def read_from_vst(filename, verbose=True):
    """Read radial velocity data from a vst file into a Pandas dataframe.

    Args:
        filename (string): Path to csv file
        verbose (bool): Notify user if instrument types not given?

    Note:
        Only relevant for HIRES users.

    """
    b = io.read_vst(filename)
    data = pd.DataFrame()
    data['time'] = b.jd
    data['mnvel'] = b.mnvel
    data['errvel'] = b.errvel
    data['tel'] = 'HIRES'

    data.to_csv(filename[:-3]+'csv')

    return data



def cartesian_product(*arrays):
    """
        Generate a cartesian product of input arrays.

    Args:
        arrays (arrays): 1-D arrays to form the cartesian product of.

    Returns:
        array: cartesian product of input arrays
    """

    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a

    return arr.reshape(-1, la)


# Test search-specific priors
def betafunc(x, a=0.867, b=3.03):#inp_list):
    """Function to use in the HIRES gamma offset prior.
    Args:
        inp_list(list): pair of floats, the difference of which we want to
                        constrain with a Gaussian prior. Hard-coded params,
                        derived empirically from HIRES analysis.

    """
    #x = inp_list[0]
    #a = inp_list[1]
    #b = inp_list[2]
    return spec.gamma(a+b)/(spec.gamma(a)*spec.gamma(b))*x**(a-1)*(1-x)**(b-1)
'''
class Beta(Prior):
    """Beta prior
    Beta prior on a given parameter. Default is Kipping eccentricity prior.
    Args:
        param (string): parameter label
        mu (float): center of Gaussian prior
        sigma (float): width of Gaussian prior
    """

    def __init__(self, alpha=0.867, beta=3.03, param):
        self.alpha = alpha
        self.beta = beta
        self.param = param

    def __call__(self, params):
        x = params[self.param].value
        return -0.5 * ((x - self.mu) / self.sigma)**2 - 0.5*np.log((self.sigma**2)*2.*np.pi)

    def __repr__(self):
        s = "Beta prior on {}, alpha={}, beta={}".format(
            self.param, self.alpha, self.beta
            )
        return s

    def __str__(self):
        try:
            tex = model.Parameters(9).tex_labels(param_list=[self.param])[self.param]

            s = "Beta prior on {}: $\\alpha={}, \\beta={}$ \\\\".format(tex, self.alpha, self.beta)
        except KeyError:
            s = self.__repr__()

        return s
'''

def derive(post, synthchains, mstar, mstar_err=0.0):
    """Derive physical parameters from posterior samples

    Args:
        post (radvel.Posterior): RadVel posterior object
        synthchains (DataFrame): MCMC chains in the RadVel synth basis
        mstar (float): stellar mass in solar units
        mstar_err (float): (optional) uncertainty on stellar mass
    """

    try:
        mstar = np.random.normal(
            loc=mstar, scale=mstar_err,
            size=len(synthchains)
        )
    except AttributeError:
        print("Unable to calculate derived parameters, stellar parameters not defined the config file.")
        return

    if (mstar <= 0.0).any():
        num_nan = np.sum(mstar <= 0.0)
        nan_perc = float(num_nan) / len(synthchains)
        mstar[mstar <= 0] = np.abs(mstar[mstar <= 0])
        print("WARNING: {} ({:.2f} %) of Msini samples are NaN. The stellar mass posterior may contain negative \
values. Interpret posterior with caution.".format(num_nan, nan_perc))

    outcols = []
    for i in np.arange(1, post.params.num_planets + 1, 1):
        # Grab parameters from the chain
        def _has_col(key):
            cols = list(synthchains.columns)
            return cols.count('{}{}'.format(key, i)) == 1

        def _get_param(key):
            if _has_col(key):
                return synthchains['{}{}'.format(key, i)]
            else:
                return post.params['{}{}'.format(key, i)].value

        def _set_param(key, value):
            synthchains['{}{}'.format(key, i)] = value

        def _get_colname(key):
            return '{}{}'.format(key, i)

        per = _get_param('per')
        k = _get_param('k')
        e = _get_param('e')

        mpsini = radvel.utils.Msini(k, per, mstar, e, Msini_units='earth')
        _set_param('mpsini', mpsini)
        outcols.append(_get_colname('mpsini'))
        low, med, high = np.quantile(mpsini, [0.159, 0.5, 0.841])
        post.medparams[_get_colname('mpsini')] = med
        post.uparams[_get_colname('mpsini')+'_err1'] = low-med
        post.uparams[_get_colname('mpsini')+'_err2'] = high-med

        mtotal = mstar + (mpsini * c.M_earth.value) / c.M_sun.value  # get total star plus planet mass
        a = radvel.utils.semi_major_axis(per, mtotal)  # changed from mstar to mtotal

        _set_param('a', a)
        outcols.append(_get_colname('a'))
        low, med, high = np.quantile(a, [0.159, 0.5, 0.841])
        post.medparams[_get_colname('a')] = med
        post.uparams[_get_colname('a')+'_err1'] = low-med
        post.uparams[_get_colname('a')+'_err2'] = high-med


        musini = (mpsini * c.M_earth) / (mstar * c.M_sun)
        _set_param('musini', musini)
        outcols.append(_get_colname('musini'))
        med, low, high = np.quantile(musini, [0.159, 0.5, 0.841])
        post.medparams[_get_colname('musini')] = med
        post.uparams[_get_colname('musini')+'_err1'] = low-med
        post.uparams[_get_colname('musini')+'_err2'] = high-med


    print("Derived parameters:", outcols)

    return post
