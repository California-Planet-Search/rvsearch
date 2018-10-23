import numpy as np
import astropy.stats
import radvel
import radvel.fitting
import matplotlib.pyplot as plt
import copy

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
                 baseline=True, num_freqs=None, search_pars = ['per'], valid_types = ['bic', 'aic', 'ls']):
        self.post = post
        self.default_pdict = {} #Default_pdict makes sense here, leave alone for now
        for k in post.params.keys():
            self.default_pdict[k] = post.params[k].value

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

        self.search_pars = search_pars
        self.valid_types = valid_types
        self.power = {key: None for key in self.valid_types}
        self.maxper = None

    @classmethod
    def from_pandas(cls, data):
        params = utils.initialize_default_pars(instnames=data.tel)
        post = utils.initialize_post(data, params=params)
        return cls(post)

    @classmethod
    def from_csv(cls, filename):
        data = utils.read_from_csv(filename)
        params = utils.initialize_default_pars(instnames=data.tel)
        post = utils.initialize_post(data, params=params)
        return cls(post)

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

        dnu       = 1. / (4. * self.timelen)
        num_freq  = int((fmax - fmin) / dnu + 1)
        num_freq *= oversampling

        if verbose:
            print("Number of test periods:", num_freq)

        freqs = np.linspace(fmin, fmax, num_freq)
        pers = 1 / freqs

        return pers

    def make_per_grid(self):
        if self.num_freqs is None:
            self.pers = self.freq_spacing()#(self.times, self.minsearchP, self.maxsearchP)
        else:
            self.pers = 1/np.linspace(1/self.maxsearchP, 1/self.minsearchP, self.num_freqs)

        self.freqs = 1/self.pers
        ''' This is what we need trend_test for, at the start of the search. Move to Search()
        if num_planets_known == 0:
    	post = trend_test(post)
        '''
    def trend_test(self):
        #Perform 0-planet baseline fit.
        post1 = copy.deepcopy(self.post)

        trend_curve_bic = self.post.likelihood.bic()
        dvdt_val = self.post.params['dvdt'].value
        curv_val = self.post.params['curv'].value

        #Test without curvature
        post1.params['curv'].value = 0.0
        post1.params['curv'].vary = False
        post1 = radvel.fitting.maxlike_fitting(post1)

        trend_bic = post1.likelihood.bic()

        #Test without trend or curvature
        post2 = copy.deepcopy(post1)

        post2.params['dvdt'].value = 0.0
        post2.params['dvdt'].vary = False
        post2 = radvel.fitting.maxlike_fitting(post2)

        flat_bic = post2.likelihood.bic()
        print('Flat:{}; Trend:{}; Curv:{}'.format(flat_bic, trend_bic, trend_curve_bic))

        if trend_bic < flat_bic - 5.:
    		#Flat model is excluded, check on curvature
    	    if trend_curve_bic < trend_bic - 5.:
    			#curvature model is preferred
    		    return self.post #t+c
    	    return post1 #trend only
        return post2 #flat

    def base_bic(self):
        base_post = self.trend_post()
        self.base_bic = base_post.likelihood.bic()

    def per_bic(self):
        #BJ's method. Remove once final BIC/AIC method is established.
        """Compute delta-BIC periodogram. ADD: crit is BIC or AIC.
        """

        """Can we track whether maxlike_fitting has been performed on a post?
        If so, we should do this, so we don't have to fit a posterior that
        has already been optimized.
        """
        #post = setup_posterior(self.post, self.num_known_planets)
        baseline_fit = radvel.fitting.maxlike_fitting(self.post, verbose=True)
        #This assumes nth planet parameters, and all periods, were locked in/
        baseline_bic = baseline_fit.likelihood.bic()
        #Run trend-post-test here

        #Allow amplitude and time offset to vary, fix eccentricity and period.
        self.post.params['k{}'.format(self.num_known_planets+1)].vary = True
        self.post.params['tc{}'.format(self.num_known_planets+1)].vary = True

        power = np.zeros_like(self.pers)
        for i, per in enumerate(self.pers):
            perkey = 'per{}'.format(self.num_known_planets+1)
            self.post.params[perkey].value = per
            self.post.params[perkey].vary = False

            fit = radvel.fitting.maxlike_fitting(self.post, verbose=False)
            power[i] = baseline_bic - fit.likelihood.bic()
            #print(i, per, power[i], fit.bic(), baseline_bic)
        self.power['bic'] = power
        self.maxper = np.amax(power)

    def ls(self):
        """Astropy Lomb-Scargle periodogram.
        """

        print("Calculating Lomb-Scargle periodogram")
        power = astropy.stats.LombScargle(self.times, self.vel, self.errvel).power(self.freq_array)
        self.power['ls'] = power

    def eFAP_thresh(self, fap=0.01):
        """Calculate the threshold for significance based on BJ's eFAP algorithm
        From Lea's code. LOMB-S OPTION?
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
        xmod = np.linspace(np.min(self.power['bic'][self.power['bic']==self.power['bic']]),
                           10.*np.max(self.power['bic']), 10000)
        lfit    = 10.**func(xmod)
        fap_min = 10.**func(sBIC[-1])*self.num_freqs #[-1] or [0]?

        #thresh = xmod[np.where(np.abs(lfit-fap/self.num_freqs) == np.min(np.abs(lfit-fap/self.num_freqs)))]
        thresh = xmod[np.argmin(np.abs(lfit - fap/self.num_freqs))]
        return thresh[0], fap_min

    def save_per(self, ls=False):
        if ls==False:
            try:
                #FIX THIS; SPECIFY DIRECTORY/NAME, NUMBER OF PLANETS IN FILENAME, AND ARRAY ORDERING
                np.savetxt((self.per_array, self.power['bic']), filename='BIC_periodogram.csv')
            except:
                print('Have not generated a delta-BIC periodogram.')
        else:
            try:
                np.savetxt((self.pers, self.power['LS']), filename='LS_periodogram.csv')
            except:
                print('Have not generated a Lomb-Scargle periodogram.')

    def plot_per(self, ls=False, save=True):
        #TO-DO: WORK IN AIC/BIC OPTION, INCLUDE IN PLOT TITLE
        peak = np.argmax(self.power['bic'])
        f_real = self.freqs[peak]

        fig, ax = plt.subplots()
        ax.set_title('Planet ' + str(self.num_known_planets+1)) #TO-DO: GET PLANET NUMBER RIGHT
        ax.plot(self.pers, self.power['bic'])
        ax.scatter(self.pers[peak], self.power['bic'][peak], label='{} days'.format(
                   np.round(self.pers[peak], decimals=1)))
        ax.legend(loc=1)

        #Plot day, month, and year aliases.
        colors = ['r', 'b', 'g']
        alias = [1, 30, 365]
        for i in np.arange(3):
            #Is this right? ASK BJ
            f_ap = 1./alias[i] + f_real
            f_am = 1./alias[i] - f_real
            ax.axvline(1./f_am, linestyle='--', c=colors[i], label='Minus {} day alias'.format(alias[i]))
            #ax.axvline(1./f_ap, linestyle='--', c=colors[i], label='Plus {} day alias'.format(alias[i]))
        ax.legend(loc=3)
        ax.set_xscale('log')
        ax.set_xlabel('Period (days)')
        ax.set_ylabel(r'$\Delta$BIC') #TO-DO: WORK IN AIC/BIC OPTION
        ax.set_title('Planet {}'.format(self.num_known_planets+1)) #TO-DO: FIGURE OUT WHERE PLANET_NUM IS

        #Store figure as object attribute, make separate saving functionality?
        self.fig = fig
        if save == True:
            #FINISH THIS
            fig.savefig('dbic.pdf')


#TO-DO: MOVE THESE INTO CLASS STRUCTURE, OR REMOVE IF UNNECESSARY
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

    #return (base_post, search_post)
    return search_post
