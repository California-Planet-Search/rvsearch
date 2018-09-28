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

    def __init__(self, posterior, minsearchp, maxsearchp, num_known_planets=0, num_freqs=None):
        self.minsearchP = minsearchp
        self.maxsearchP = maxsearchp
        self.num_freqs = num_freqs

        self.num_known_planets = num_known_planets

        self.post = posterior

        self.times = self.post.likelihood.x
        self.vel = self.post.likelihood.y
        self.errvel = self.post.likelihood.yerr

        if num_freqs is None:
            self.per_array = freq_spacing(self.times, self.minsearchP, self.maxsearchP)
        else:
            self.per_array = 1/np.linspace(1/self.maxsearchP, 1/self.minsearchP, self.num_freqs)

        self.freq_array = 1/self.per_array

        self.power = {key: None for key in VALID_TYPES}

    def bic(self):
        """Compute delta-BIC periodogram"""

        baseline_fit = radvel.fitting.maxlike_fitting(self.post, verbose=True)
        post = setup_posterior(self.post, self.num_known_planets)
        baseline_bic = baseline_fit.bic()

        power = np.zeros_like(self.per_array)
        for i, per in enumerate(self.per_array):
            perkey = 'per{}'.format(self.num_known_planets + 1)
            post.params[perkey].value = per
            post.params[perkey].vary = False

            fit = radvel.fitting.maxlike_fitting(post, verbose=False)

            power[i] = (fit.bic() - baseline_bic)

            print(i, per, power[i], fit.bic(), baseline_bic)
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


#TO-DO: MOVE THIS INTO CLASS STRUCTURE
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
