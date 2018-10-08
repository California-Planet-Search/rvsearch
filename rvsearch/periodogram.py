import numpy as np
import astropy.stats
import radvel
import radvel.fitting
import matplotlib.pyplot as plt

import utils


class Periodogram:
    """Class to calculate and store periodograms.

    Args:
        posterior (radvel.Posterior): radvel.Posterior object
        minsearchp (float): minimum search period
        maxsearchp (float): maximum search period
        num_known_planets (int): Assume this many known planets in the system and search for one more
        num_freqs (int): (optional) number of frequencies to test
            [default = calculated via rvsearch.periodograms.freq_spacing]
    """

    def __init__(self, post, basebic=None, num_known_planets=0, minsearchp=3, maxsearchp=10000,
                 baseline=False, num_freqs=None, valid_types = ['bic', 'ls']):
        self.post = post
        self.basebic = basebic

        self.num_known_planets = num_known_planets

        self.times = self.post.likelihood.x
        self.vel = self.post.likelihood.y
        self.errvel = self.post.likelihood.yerr

        self.timelen = np.amax(self.times) - np.amin(self.times)

        self.minsearchP = minsearchp
        self.maxsearchP = maxsearchp
        self.baseline = baseline
        self.num_freqs = num_freqs

        if self.baseline == True:
            self.maxsearchP = 4. * self.timelen #SHOULD '4' BE VARIABLE?

        self.valid_types = valid_types
        self.power = {key: None for key in self.valid_types}

    def from_post(cls, post):
        return cls(post)

    def from_pandas(cls, data):
        post = utils.initialize_post(data)
        return cls(post)

    def from_csv(filename):
        data = utils.read_from_csv(filename)
        post = utils.initialize_post(data)
        return Periodogram(post)

    def freq_spacing(self, oversampling=1, verbose=True):
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

        fmin = 1. / self.maxsearchP
        fmax = 1. / self.minsearchP

        dnu = 1. / (4. * self.timlen)
        numf = int((fmax - fmin) / dnu + 1)
        res = numf
        res *= oversampling

        if verbose:
            print("Number of test periods:", res)

        Farr = np.linspace(1 / maxp, 1 / minp, res)
        Parr = 1 / Farr

        return Parr

    def make_per_grid(self):
        if num_freqs is None:
            self.pers = self.freq_spacing(self.times, self.minsearchP, self.maxsearchP)
        else:
            self.pers = 1/np.linspace(1/self.maxsearchP, 1/self.minsearchP, self.num_freqs)

        self.freqs = 1/self.pers

    def bics(self, post, base_bic, base_chi2, base_logp, planet_num):
        #Lea's method, rewrite this with desired inputs and outputs.
        #base_bic is the nth planet BIC array (we are calculating n+1th)
        """Loop over Parr, calculate delta-BIC values

        Args:
            post (radvel Posteriors object): should have 'per{}'.format(planet_num) fixed.
            base_bic (float): The comparison BIC value
            planet_num (int): num_planets+1
            default_pdict (dict): default params to start each maxlike fit

        Returns:
            BICs (array): List of delta bic values (or delta aic values)
        """
        BICs = []
        chi2arr = []
        logparr = []
        bestfit = []
        post = self.post

        for per in self.pers:
        	#Reset post to default params:
            post = utils.reset_params(post, self.search.default_pdict)

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

        self.power['bic'] = BICarr
        self.bestfit = bestfit
        #return BICarr, chi2arr, logparr, bestfit

    def per_bic(self):
        #BJ's method. Remove once final BIC/AIC method is established.
        """Compute delta-BIC periodogram. ADD: crit is BIC or AIC.
        """

        baseline_fit = radvel.fitting.maxlike_fitting(self.post, verbose=True)
        post = setup_posterior(self.post, self.num_known_planets)
        baseline_bic = baseline_fit.bic()

        power = np.zeros_like(self.per_array)
        for i, per in enumerate(self.per_array):
            perkey = 'per{}'.format(self.num_known_planets + 1)
            post.params[perkey].value = per
            post.params[perkey].vary = False

            fit = radvel.fitting.maxlike_fitting(post, verbose=False)

            power[i] = (fit.bic() - baseline_bic[i])

            print(i, per, power[i], fit.crit(), baseline_bic)
        self.power['bic'] = power

    def ls(self):
        """Astropy Lomb-Scargle periodogram
        """

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

    def plot_per(self, ls=False, save=True):
        #TO-DO: WORK IN AIC/BIC OPTION, INCLUDE IN PLOT TITLE
        peak = np.argmax(self.power['bic'])
        f_real = 1./p[peak]

        fig, ax = plt.subplots()
        ax.set_title('Planet ' + str(self.search.planet_num)) #TO-DO: FIGURE OUT WHERE PLANET_NUM IS
        ax.plot(self.pers, self.power['bic'])
        ax.scatter(self.pers[peak], self.power['bic'][peak], label='{} days'.format(
                   np.round(self.pers[peak], decimals=1)))
        ax.legend(loc=3)

        #Plot day, month, and year aliases.
        for alias in [1, 30, 365]:
            #Is this right? ASK BJ
            f_ap = 1/cad + f_real
            f_am = 1/cad - f_real
            ax.axvline(1/f_am, linestyle='--', label='Minus {} day alias'.format(cad))
            ax.axvline(1/f_ap, linestyle='--', label='Plus {} day alias'.format(cad))
        ax.set_xscale('log')
        ax.set_xlabel('Period (days)')
        ax.set_ylabel(r'$\Delta$BIC') #TO-DO: WORK IN AIC/BIC OPTION
        ax.set_title('Planet {}'.format(self.planet_num)) #TO-DO: FIGURE OUT WHERE PLANET_NUM IS

        #Store figure as object attribute, make separate saving functionality?
        self.fig = fig
        if save == True:
            #FINISH THIS
            fig.savefig('dbic.pdf')

    def eFAP_thresh(self, fap=0.01):
    	"""Calculate the threshold for significance based on BJ's eFAP algorithm
        From Lea's code. TO-DO: DEFINE BICARR, ETC. BICARR IS A BIC PERIODOGRAM. LOMB-S OPTION?
    	"""
    	#select out intermediate values of BIC
    	sBIC = np.sort(self.power['bic'])
    	crop_BIC = sBIC[int(0.5*len(sBIC)):int(0.95*len(sBIC))] #select only median - 95% vals

    	#histogram
    	hist, edge = np.histogram(crop_BIC, bins=10)
    	cent = (edge[1:]+edge[:-1])/2.
    	norm = float(np.sum(hist))

    	nhist = hist/norm

    	func = np.poly1d(np.polyfit(cent, np.log10(nhist), 1))
    	xmod = np.linspace(np.min(BICarr[BICarr==BICarr]), 10.*np.max(BICarr), 10000)
    	lfit = 10.**(func(xmod))
    	fap_min = 10.**func(max(self.power['bic'])) * self.num_freqs

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
