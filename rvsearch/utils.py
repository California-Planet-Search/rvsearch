#Utilities for loading data, checking for known planets, etc.
import pdb

import numpy as np
import scipy
import pandas as pd
import radvel
try:
	import cpsutils
	from cpsutils import io
except:
	RuntimeError()


"""Functions for posterior modification (resetting parameters, intializing, etc.)
"""

def reset_params(post, default_pdict):
	#Reset post.params values to default values
	for k in default_pdict.keys():
		post.params[k].value = default_pdict[k]
	return post

def initialize_default_pars(instnames=['inst'], fitting_basis='per tc secosw sesinw k'):
    """Set up a default Parameters object.

	None of the basis values are free params, for the initial 0-planet fit.
	Remember to reset .vary to True for all relevant params.

    Args:
        instnames (list): codes of instruments used
        fitting_basis: optional

    Returns:
        Parameters object
    """

    anybasis_params = radvel.Parameters(num_planets=1, basis='per tc e w k')

    anybasis_params['tc1'] = radvel.Parameter(value=2455200.0)
    anybasis_params['w1'] = radvel.Parameter(value=np.pi/2.)
    anybasis_params['k1'] = radvel.Parameter(value=0.0)
    anybasis_params['e1'] = radvel.Parameter(value=0.0)
    anybasis_params['per1'] = radvel.Parameter(value=100.0)

    anybasis_params['dvdt'] = radvel.Parameter(value=0.0)
    anybasis_params['curv'] = radvel.Parameter(value=0.0)

    for inst in instnames:
        anybasis_params['gamma_'+inst] = radvel.Parameter(value=0.0)
        anybasis_params['jit_'+inst] = radvel.Parameter(value=2.0)

    params = anybasis_params.basis.to_any_basis(anybasis_params, fitting_basis)

    params['secosw1'].vary = False
    params['sesinw1'].vary = False
    params['per1'].vary = False

    return params

def initialize_post(data, params=None, priors=None):
	"""Initialize a posterior object with data, params, and priors.
	Args:
		data: a pandas dataframe.
		params: a list of radvel parameter objects.
		priors: a list of priors to place on the posterior object.
	Returns:
		post (radvel Posterior object)

	TO-DO: MAKE OPTION FOR KNOWN MULTI-PLANET POSTERIOR
	"""

	if params == None:
		params = radvel.Parameters(1, basis='per tc secosw sesinw logk')
	iparams = radvel.basis._copy_params(params)

	# Allow for time to be listed as 'time' or 'jd' (Julian Date).
	if {'jd'}.issubset(data.columns):
		data['time'] = data['jd']

	#initialize RVModel
	time_base = np.mean([data['time'].max(), data['time'].min()])
	mod = radvel.RVModel(params, time_base=time_base)

	#initialize Likelihood objects for each instrument
	telgrps = data.groupby('tel').groups
	likes = {}

	for inst in telgrps.keys():
		likes[inst] = radvel.likelihood.RVLikelihood(
			mod, data.iloc[telgrps[inst]].time, data.iloc[telgrps[inst]].mnvel,
			data.iloc[telgrps[inst]].errvel, suffix='_'+inst)

		likes[inst].params['gamma_'+inst] = iparams['gamma_'+inst]
		likes[inst].params['jit_'+inst] = iparams['jit_'+inst]
	#Can this be cleaner? like = radvel.likelihood.CompositeLikelihood(likes), if likes is array, not dic.
	like = radvel.likelihood.CompositeLikelihood(list(likes.values()))

	post = radvel.posterior.Posterior(like)
	#FIX TO COMBINE GIVEN PRIORS AND NEEDED PRIORS
	if priors is not None:
		post.priors = priors
	else:
		priors = []
		priors.append(radvel.prior.PositiveKPrior(post.params.num_planets))
		priors.append(radvel.prior.EccentricityPrior(post.params.num_planets))
		#priors.append([radvel.prior.HardBounds('jit_'+inst, 0.0, 20.0) for inst in telgrps.keys()])
		post.priors = priors

	return post

def window(time, freqs, plot=False):
	"""Function to generate, and possibly plot, the window function of observations.
	Args:
		time: times of observations in a dataset. FOR SEPARATE TELESCOPES?
	"""
	W = np.zeros(len(freqs))
	for i, freq in enumerate(freqs):
		W[i] = np.sum(np.exp(-2*np.pi*1j*time*freq))
	W /= float(len(freq))
	return W

"""Testing fitting options besides scipy.optimize.minimize. Just other methods
and basinhopping for now, eventually partial/full linearization.
"""
def basin_fitting(post, verbose=True, minimizer_kwargs={'method':'Powell', 'options':dict(xtol=1e-8,maxiter=200,maxfev=100000)}): #options=dict(xtol=1e-8,maxiter=200,maxfev=100000):
	"""Maximum likelihood fitting, with an annealing method.

	Args:
        post (radvel.Posterior): Posterior object with initial guesses
        verbose (bool [optional]): Print messages and fitted values?
        method (string [optional]): Minimization method. See documentation for `scipy.optimize.minimize` for available
            options.

	Returns:
		radvel.Posterior: Posterior object with parameters
		updated to their maximum likelihood values.
	"""
	if verbose:
		print('Initial loglikelihood = %f' % post.logprob())
		print("Performing maximum likelihood fit...")

	res = scipy.optimize.basinhopping(post.neglogprob_array, post.get_vary_params(),
										minimizer_kwargs=minimizer_kwargs)
	synthparams = post.params.basis.to_synth(post.params, noVary = True)
	post.params.update(synthparams)

	if verbose:
		print("Final loglikelihood = %f" % post.logprob())
		print("Best-fit parameters:")
		print(post)

	return post

"""Series of functions for reading data from various sources into pandas dataframes.
"""
def read_from_csv(filename, verbose=True):
    data = pd.read_csv(filename)
    if 'tel' not in data.columns:
        if verbose:
            print('Instrument types not given.')
        data['tel'] = 'Inst.'
    return data

def read_from_arrs(t, mnvel, errvel, tel=None, verbose=True):
    data = pd.DataFrame()
    data['time'], data['mnvel'], data['errvel'] = t, mnvel, errvel
    if tel == None:
        if verbose:
            print('Instrument type not given.')
        data['tel'] = 'Inst.'
    else:
        data['tel'] = tel
    return data

def read_from_vst(filename, verbose=True):
    """This reads .vst files generated by the CPS pipeline, which
    means that it is only relevant for HIRES data.
    """
    b = io.read_vst(filename)
    data = pd.DataFrame()
    data['time'] = b.jd
    data['mnvel'] = b.mnvel
    data['errvel'] = b.errvel
    data['tel'] = 'HIRES'

    data.to_csv(filename[:-3]+'csv')
    return data

# Function for collecting results of searches in current directory.
def scrape(starlist, save=True):
	all_params = []

	for star in starlist:
		params = {}
		params['star'] = star
		post = radvel.posterior.load(star+'/post_final.pkl')
		if post.params.num_planets == 1:
			if post.params['k1'].value == 0.:
				num_planets = 0
			else:
				num_planets = 1
		else:
			num_planets = post.params.num_planets
		params['num_planets'] = num_planets

		for k in post.params.keys():
			params[k] = post.params[k].value
		all_params.append(params)

	dataframe = pd.DataFrame(all_params)
	if save:
		dataframe.to_csv('system_props.csv')
	return dataframe

# Test search-specific priors
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
