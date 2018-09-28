import numpy as np
import astropy.stats
import radvel
import radvel.fitting

VALID_TYPES = ['bic', 'ls']

class Periodogram(object):
    """
    Class to calculate and store periodograms.

    Args:
        posterior (radvel.Posterior): radvel.Posterior object
        minsearchp (float): minimum search period
        maxsearchp (float): maximum search period
        num_known_planets (int): Assume this many known planets in the system and search for one more
        num_freqs (int): (optional) number of frequencies to test
            [default = calculated via rvsearch.periodograms.freq_spacing]
    """
    #TO-DO: IN __INIT__, CHANGE POSTERIOR INPUT TO SEARCH CLASS INPUT.
    def __init__(self, search, minsearchp, maxsearchp, num_known_planets=0, num_freqs=None):
        self.minsearchP = minsearchp
        self.maxsearchP = maxsearchp
        self.num_freqs = num_freqs

        self.num_known_planets = num_known_planets

        self.post = search.post

        self.times = self.post.likelihood.x
        self.vel = self.post.likelihood.y
        self.errvel = self.post.likelihood.yerr

        if num_freqs is None:
            self.per_array = freq_spacing(self.times, self.minsearchP, self.maxsearchP)
        else:
            self.per_array = 1/np.linspace(1/self.maxsearchP, 1/self.minsearchP, self.num_freqs)

        self.freq_array = 1/self.per_array

        self.power = {key: None for key in VALID_TYPES}

def ics(self, post, base_bic, base_chi2, base_logp, Parr, planet_num, default_pdict):
    """Loop over Parr, calculate delta-BIC values

    Args:
        post (radvel Posteriors object): should have 'per{}'.format(planet_num) fixed.
        base_bic (float): The comparison BIC value
        Parr (array): List of periods to check
        planet_num (int): num_planets+1
        default_pdict (dict): default params to start each maxlike fit

    Returns:
        BICarr (array): List of delta bic values (or delta aic values)
    """
    BICarr = []
    chi2arr = []
    logparr = []
    bestfit = []

    for per in Parr:
    	#Reset post to default params:
    	post = ut.reset_to_default(post, default_pdict)

        #Set the period in the post object and perform maxlike fitting
        post.params['per{}'.format(planet_num)].value = per
        post = radvel.fitting.maxlike_fitting(post)

        bic = post.bic()
        delta_bic = base_bic - bic  #Should be positive since bic < base_bic
        BICarr += [delta_bic]

        chi2 = np.sum((post.likelihood.residuals()**2.)/(post.likelihood.errorbars()**2.))
        delta_chi2 = (base_chi2 - chi2) / base_chi2
        chi2arr += [delta_chi2]

        logp = post.logprob()
        delta_logp = logp - base_logp
        logparr += [delta_logp]

        #Save best fit params too
        best_params = {}
        for k in post.params.keys():
        	best_params[k] = post.params[k].value
        bestfit += [best_params]

    return BICarr, chi2arr, logparr, bestfit

    def per_ic(self, crit):
        #BJ's method, replacing with Lea's method
        """Compute delta-BIC periodogram. crit is BIC or AIC."""

        baseline_fit = radvel.fitting.maxlike_fitting(self.post, verbose=True)
        post = setup_posterior(self.post, self.num_known_planets)
        baseline_ic = baseline_fit.crit()

        power = np.zeros_like(self.per_array)
        for i, per in enumerate(self.per_array):
            perkey = 'per{}'.format(self.num_known_planets + 1)
            post.params[perkey].value = per
            post.params[perkey].vary = False

            fit = radvel.fitting.maxlike_fitting(post, verbose=False)

            power[i] = (fit.crit() - baseline_ic)

            print(i, per, power[i], fit.crit(), baseline_ic)
        self.power['bic'] = power

    def ls(self):
        """Astropy Lomb-Scargle periodogram"""

        print("Calculating Lomb-Scargle periodogram")

        power = astropy.stats.LombScargle(self.times, self.vel, self.errvel).power(self.freq_array)

        self.power['ls'] = power

    def save_per(self, ls=False):
        if ls==False:
            try:
                #FIX THIS; SPECIFY DIRECTORY/NAME, NUMBER OF PLANETS IN FILENAME, AND ARRAY ORDERING
                np.savetxt((self.per_array, self.power['bic']), filename='BIC_periodogram.csv')
            except:
                print('Have not generated a delta-BIC periodogram.')
        else:
            try:
                #FIX THIS, SPECIFY DIRECTORY/NAME, NUMBER OF PLANETS IN FILENAME, AND ARRAY ORDERING
                np.savetxt((self.per_array, self.power['LS']), filename='LS_periodogram.csv')
            except:
                print('Have not generated a Lomb-Scargle periodogram.')

    def plot_per(self):
        pass

    def eFAP_thresh(self, fap=0.01):
    	"""
        Calculate the threshold for significance based on BJ's eFAP algorithm
        From Lea's code. TO-DO: DEFINE BICARR, ETC. BICARR IS A BIC PERIODOGRAM. LOMB-S OPTION?
    	"""
    	#select out intermediate values of BIC
    	sBIC = np.sort(BICarr)
    	crop_BIC = sBIC[int(0.5*len(sBIC)):int(0.95*len(sBIC))] #select only median - 95% vals

    	#histogram
    	hist, edge = np.histogram(crop_BIC, bins=10)
    	cent = (edge[1:]+edge[:-1])/2.
    	norm = float(np.sum(hist))

    	nhist = hist/norm

    	func = np.poly1d(np.polyfit(cent, np.log10(nhist), 1))
    	xmod = np.linspace(np.min(BICarr[BICarr==BICarr]), 10.*np.max(BICarr), 10000)
    	lfit = 10.**(func(xmod))
    	fap_min = 10.**func(max(BICarr)) * len(BICarr)

    	thresh = xmod[np.where(np.abs(lfit-fap/len(BICarr)) == np.min(np.abs(lfit-fap/len(BICarr))))]
    	return thresh[0], fap_min


#TO-DO: MOVE THESE INTO CLASS STRUCTURE
def setup_posterior(post, num_known_planets):
    """Setup radvel.posterior.Posterior object

    Prepare posterior object for periodogram calculations. Fix values for previously-known planets.

    Args:
        post (radvel.posterior.Posterior): RadVel posterior object. Can be initialized from setup file or loaded
            from a RadVel fit.
        num_known_planets (int): Number of previously known planets. Parameters for these planets will be fixed.

    Returns:
        tuple: (radvel.posterior object used as baseline fit, radvel.posterior used in search)

    """
    basis_pars = post.likelihood.params.basis.name.split()

    for i in range(1, post.params.num_planets + 1):
        for par in basis_pars:
            parname = "{}{}".format(par, i)
            post.params[parname].vary = False

            if par == 'k':
                post.params[parname].value = 0.0
            elif par == 'logk':
                post.params[parname].value = -9

    return (base_post, search_post)

#TO-DO: MOVE INTO UTILS FILE?
def freq_spacing(times, minp, maxp, oversampling=1, verbose=True):
    """Get the number of sampled frequencies

    Condition for spacing: delta nu such that during the
    entire duration of observations, phase slip is no more than P/4

    Args:
        times (array): array of timestamps
        minp (float): minimum period
        maxp (float): maximum period
        oversampling (float): (optional) oversampling factor
        verbose (bool): (optional) print extra messages

    Returns:
        array: Array of test periods
    """

    fmin = 1 / maxp
    fmax = 1 / minp

    timlen = max(times) - min(times)
    dnu = 1. / (4. * timlen)
    numf = int((fmax - fmin) / dnu + 1)
    res = numf
    res *= oversampling

    if verbose:
        print("Number of test periods:", res)

    Farr = np.linspace(1 / maxp, 1 / minp, res)
    Parr = 1 / Farr

    return Parr
