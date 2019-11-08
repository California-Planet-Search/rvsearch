"""Search class"""

import os
import copy
import pdb
import pickle

import numpy as np
import matplotlib.pyplot as pl
import corner
#from astroML.time_series import \
#    lomb_scargle, lomb_scargle_BIC, lomb_scargle_bootstrap
from astropy.timeseries import LombScargle
import radvel
import radvel.fitting
from radvel.plot import orbit_plots

import rvsearch.periodogram as periodogram
import rvsearch.utils as utils


class Search(object):
    """Class to initialize and modify posteriors as planet search runs.

    Args:
        data (DataFrame): pandas dataframe containing times, vel, err, and insts.
        post (radvel.Posterior): Optional posterior with known planet params.
        starname (str): String, used to name the output directory.
        max_planets (int): Integer, limit on iterative planet search.
        priors (list): List of radvel prior objects to use.
        crit (str): Either 'bic' or 'aic', depending on which criterion to use.
        fap (float): False-alarm-probability to pass to the periodogram object.
        min_per (float): Minimum search period, to pass to the periodogram object.
        trend (bool): Whether to perform a DBIC test to select a trend model.
        linear(bool): Wether to linearly optimize gamma offsets.
        fix (bool): Whether to fix known planet parameters during search.
        polish (bool): Whether to create finer period grid after planet is found.
        verbose (bool):
        save_outputs (bool): Save output plots and files? [default = True]
        mstar (tuple): (optional) stellar mass and uncertainty in solar units

    """
    def __init__(self, data, post=None, starname='star', max_planets=8,
                priors=[], crit='bic', fap=0.001, min_per=3, max_per=10000,
                manual_grid=None, oversampling=1., trend=False, linear=True,
                eccentric=False, fix=False, polish=True, baseline=True,
                mcmc=True, workers=1, verbose=True, save_outputs=True, mstar=None):

        if {'time', 'mnvel', 'errvel', 'tel'}.issubset(data.columns):
            self.data = data
            self.tels = np.unique(self.data['tel'].values)
        elif {'jd', 'mnvel', 'errvel', 'tel'}.issubset(data.columns):
            self.data = data
            self.data.time = self.data.jd
            self.tels = np.unique(self.data['tel'].values)
        else:
            raise ValueError('Incorrect data input.')

        self.starname = starname
        self.linear   = linear
        if mstar is not None:
            self.mstar     = mstar[0]
            self.mstar_err = mstar[1]
        else:
            self.mstar     = None
            self.mstar_err = None

        if post == None:
            self.priors = priors
            self.params = utils.initialize_default_pars(instnames=self.tels,
                                                        times=data.time,
                                                        linear=self.linear)
            self.post   = utils.initialize_post(data, params=self.params,
                                                priors=self.priors,
                                                linear=self.linear)
            self.setup  = False
            self.setup_planets = -1
        else:
            self.post   = post
            self.setup  = True
            self.setup_planets = self.post.params.num_planets

        self.all_params = []

        self.max_planets = max_planets
        if self.post.params.num_planets == 1 and self.post.params['k1'].value == 0.:
            self.num_planets = 0
        else:
            self.num_planets = self.post.params.num_planets

        self.crit = crit

        # # Play with calling __name__ of method
        # if crit=='bic':
        #     self.crit = radvel.posterior.bic()
        # eif crit=='aic':
        #     self.crit = radvel.posterior.aic()
        # self.critname = self.crit.__string__
        # else:
        #     raise ValueError('Invalid information criterion.')

        self.fap = fap
        self.min_per = min_per
        self.max_per = max_per

        self.trend     = trend
        self.eccentric = eccentric
        self.fix       = fix
        self.polish    = polish
        self.baseline  = baseline
        self.mcmc      = mcmc

        self.manual_grid  = manual_grid
        self.oversampling = oversampling
        self.workers      = workers
        self.verbose      = verbose
        self.save_outputs = save_outputs

        self.basebic = None
        '''
        if post == None:
            self.basebic = None
        else:
            self.basebic = post.likelihood.bic()
        '''

        self.pers = None
        self.periodograms = dict()
        self.bic_threshes = dict()
        self.best_bics = dict()
        self.eFAPs = dict()

    def trend_test(self):
        """Perform zero-planet baseline fit, test for significant trend.

        """
        post1 = copy.deepcopy(self.post)
        # Fix all Keplerian parameters. K is zero, equivalent to no planet.
        post1.params['k1'].vary      = False
        post1.params['tc1'].vary     = False
        post1.params['per1'].vary    = False
        post1.params['secosw1'].vary = False
        post1.params['sesinw1'].vary = False
        post1 = radvel.fitting.maxlike_fitting(post1, verbose=False)

        trend_curve_bic = post1.likelihood.bic()

        # Test without curvature
        post2 = copy.deepcopy(post1)
        post2.params['curv'].value = 0.0
        post2.params['curv'].vary  = False
        post2 = radvel.fitting.maxlike_fitting(post2, verbose=False)

        trend_bic = post2.likelihood.bic()

        # Test without trend or curvature
        post3 = copy.deepcopy(post2)
        post3.params['dvdt'].value = 0.0
        post3.params['dvdt'].vary  = False
        post3.params['curv'].value = 0.0
        post3.params['curv'].vary  = False

        flat_bic = post3.likelihood.bic()
        '''
        if trend_curve_bic < trend_bic - 5:
            # Quadratic
            self.post.params['dvdt'].value = post1.params['dvdt'].value
            self.post.params['curv'].value = post1.params['curv'].value

        elif trend_bic < flat_bic - 5:
            # Linear
            self.post.params['curv'].value = 0
            self.post.params['dvdt'].value = post2.params['dvdt'].value
            self.post.params['curv'].vary  = False

        else:
            # Flat
            self.post.params['dvdt'].value = 0
            self.post.params['curv'].value = 0
            self.post.params['dvdt'].vary  = False
            self.post.params['curv'].vary  = False
        '''
        if trend_bic < flat_bic - 5:
            if trend_curve_bic < trend_bic - 5:
                # Quadratic
                self.post.params['dvdt'].value = post1.params['dvdt'].value
                self.post.params['curv'].value = post1.params['curv'].value
            else:
                # Linear
                self.post.params['dvdt'].value = post2.params['dvdt'].value
                self.post.params['curv'].value = 0
                self.post.params['curv'].vary  = False
        else:
            # Flat
            self.post.params['dvdt'].value = 0
            self.post.params['curv'].value = 0
            self.post.params['dvdt'].vary  = False
            self.post.params['curv'].vary  = False


    def add_planet(self):
        """Add parameters for one more planet to posterior.

        """
        current_num_planets = self.post.params.num_planets
        fitting_basis = self.post.params.basis.name
        param_list = fitting_basis.split()

        new_num_planets = current_num_planets + 1

        default_pars = utils.initialize_default_pars(instnames=self.tels,
                                                     fitting_basis=fitting_basis)
        new_params = radvel.Parameters(new_num_planets, basis=fitting_basis)

        for planet in np.arange(1, new_num_planets):
            for par in param_list:
                parkey = par + str(planet)
                new_params[parkey] = self.post.params[parkey]

        for par in self.post.likelihood.extra_params:
            new_params[par] = self.post.params[par]  # For gamma and jitter

        # Set default parameters for n+1th planet
        default_params = utils.initialize_default_pars(self.tels,
                                                       fitting_basis=fitting_basis)
        for par in param_list:
            parkey = par + str(new_num_planets)
            onepar = par + '1'  # MESSY, FIX THIS 10/22/18
            new_params[parkey] = default_params[onepar]

        new_params['dvdt'] = self.post.params['dvdt']
        new_params['curv'] = self.post.params['curv']

        if not self.post.params['dvdt'].vary:
            new_params['dvdt'].vary = False
        if not self.post.params['curv'].vary:
            new_params['curv'].vary = False

        new_params['per{}'.format(new_num_planets)].vary = False
        if not self.eccentric:
            new_params['secosw{}'.format(new_num_planets)].vary = False
            new_params['sesinw{}'.format(new_num_planets)].vary = False

        new_params.num_planets = new_num_planets

        priors = []
        priors.append(radvel.prior.PositiveKPrior(new_num_planets))
        priors.append(radvel.prior.EccentricityPrior(new_num_planets))
        new_post = utils.initialize_post(self.data, new_params, priors)
        self.post = new_post


    def sub_planet(self):
        """Remove parameters for one  planet from posterior.

        """
        current_num_planets = self.post.params.num_planets
        fitting_basis = self.post.params.basis.name
        param_list = fitting_basis.split()

        new_num_planets = current_num_planets - 1

        default_pars = utils.initialize_default_pars(instnames=self.tels)
        new_params = radvel.Parameters(new_num_planets, basis=fitting_basis)

        for planet in np.arange(1, new_num_planets+1):
            for par in param_list:
                parkey = par + str(planet)
                new_params[parkey] = self.post.params[parkey]

        # Add gamma and jitter params to the dictionary.
        for par in self.post.likelihood.extra_params:
            new_params[par] = self.post.params[par]

        new_params['dvdt'] = self.post.params['dvdt']
        new_params['curv'] = self.post.params['curv']

        if not self.post.params['dvdt'].vary:
            new_params['dvdt'].vary = False
        if not self.post.params['curv'].vary:
            new_params['curv'].vary = False

        priors = []
        priors.append(radvel.prior.PositiveKPrior(new_num_planets))
        priors.append(radvel.prior.EccentricityPrior(new_num_planets))

        new_post = utils.initialize_post(self.data, new_params, priors)
        self.post = new_post


    def fit_orbit(self):
        """Perform a max-likelihood fit with all parameters free.

        """
        for n in np.arange(1, self.num_planets+1):
            self.post.params['per{}'.format(n)].vary = True
            self.post.params['k{}'.format(n)].vary = True
            self.post.params['tc{}'.format(n)].vary = True
            self.post.params['secosw{}'.format(n)].vary = True
            self.post.params['sesinw{}'.format(n)].vary = True

        if self.polish:
            # Make a finer, narrow period grid, and search with eccentricity.
            self.post.params['per{}'.format(self.num_planets)].vary = False
            default_pdict = {}
            for k in self.post.params.keys():
                default_pdict[k] = self.post.params[k].value
            polish_params = []
            polish_bics = []
            peak = np.argmax(self.periodograms[self.num_planets-1])
            if self.manual_grid is not None:
                # Polish around 1% of period value if manual grid specified
                # especially useful in the case that len(manual_grid) == 1
                subgrid = np.linspace(0.99*self.manual_grid[peak], 1.01*self.manual_grid[peak], 9)
            elif peak == len(self.periodograms[self.num_planets-1]) - 1:
                subgrid = np.linspace(self.pers[peak-1], 2*self.pers[peak] - self.pers[peak-1], 9)
            else:  # TO-DO: JUSTIFY 9 GRID POINTS, OR TAKE AS ARGUMENT
                subgrid = np.linspace(self.pers[peak-1], self.pers[peak+1], 9)

            fit_params = []
            power = []
            for per in subgrid:
                for k in default_pdict.keys():
                    self.post.params[k].value = default_pdict[k]
                perkey = 'per{}'.format(self.num_planets)
                self.post.params[perkey].value = per

                fit = radvel.fitting.maxlike_fitting(self.post, verbose=False)
                power.append(-fit.likelihood.bic())

                best_params = {}
                for k in fit.params.keys():
                    best_params[k] = fit.params[k].value
                fit_params.append(best_params)

            fit_index = np.argmax(power)
            bestfit_params = fit_params[fit_index]
            for k in self.post.params.keys():
                self.post.params[k].value = bestfit_params[k]
            self.post.params['per{}'.format(self.num_planets)].vary = True

        self.post = radvel.fitting.maxlike_fitting(self.post, verbose=False)

        if self.fix:
            for n in np.arange(1, self.num_planets+1):
                self.post.params['per{}'.format(n)].vary = False
                self.post.params['k{}'.format(n)].vary = False
                self.post.params['tc{}'.format(n)].vary = False
                self.post.params['secosw{}'.format(n)].vary = False
                self.post.params['sesinw{}'.format(n)].vary = False

    def save(self, filename='post_final.pkl'):
        """Pickle current posterior.

        """
        self.post.writeto(filename)

    def running_per(self):
        """Generate running BIC periodograms for each planet/signal.

        """
        nobs = len(self.post.likelihood.x)
        # Sort times, RVs, and RV errors chronologically.
        indices = np.argsort(self.post.likelihood.x)
        x    = self.post.likelihood.x[indices]
        y    = self.post.likelihood.y[indices]
        yerr = self.post.likelihood.yerr[indices]
        tels = self.post.likelihood.telvec[indices]
        # Generalized Lomb-Scargle version; functional, but seems iffy.
        # Subtract off gammas and trend terms.
        for tel in self.tels:
            y[np.where(tels == tel)] -= self.post.params['gamma_{}'.format(tel)].value

        if self.post.params['dvdt'].vary == True:
            y -= self.post.params['dvdt'].value * (x - self.post.likelihood.model.time_base)

        if self.post.params['curv'].vary == True:
            y -= self.post.params['curv'].value * (x - self.post.likelihood.model.time_base)**2

        # Instantiate a list to populate with running periodograms.
        runners = []
        # Iterate over the planets/signals.
        for n in np.arange(1, self.num_planets+1):
            runner = []
            planets = np.arange(1, self.num_planets+1)
            yres = copy.deepcopy(y)
            for p in planets[planets != n]:
                orbel = [self.post.params['per{}'.format(p)].value,
                         self.post.params['tp{}'.format(p)].value,
                         self.post.params['e{}'.format(p)].value,
                         self.post.params['w{}'.format(p)].value,
                         self.post.params['k{}'.format(p)].value]
                yres -= radvel.kepler.rv_drive(x, orbel)

            for i in np.arange(12, nobs+1):
                freq  = 1. / self.post.params['per{}'.format(n)].value
                runner.append(LombScargle(x[:i], yres[:i], yerr[:i],
                              normalization='psd').power(freq))

            runners.append(runner)
        self.runners = runners

    def run_search(self, fixed_threshold=None, outdir=None, mkoutdir=True):
        """Run an iterative search for planets not given in posterior.

        Args:
            fixed_threshold (float): (optional) use a fixed delta BIC threshold
            mkoutdir (bool): create the output directory?
        """
        if outdir is None:
            outdir = os.path.join(os.getcwd(), self.starname)
        if mkoutdir and not os.path.exists(outdir):
            os.mkdir(outdir)

        if self.trend:
            self.trend_test()
            #self.post.params['dvdt'].vary = True
            #self.post.params['curv'].vary = True
        else:
            self.post.params['dvdt'].vary = False
            self.post.params['curv'].vary = False

        run = True
        while run:
            if self.num_planets != 0:
                self.add_planet()

            perioder = periodogram.Periodogram(self.post, basebic=self.basebic,
                                               minsearchp=self.min_per,
                                               maxsearchp=self.max_per,
                                               fap=self.fap,
                                               manual_grid=self.manual_grid,
                                               oversampling=self.oversampling,
                                               baseline=self.baseline,
                                               eccentric=self.eccentric,
                                               workers=self.workers,
                                               verbose=self.verbose)
            # Run the periodogram, store arrays and threshold (if computed).
            perioder.per_bic()
            self.periodograms[self.num_planets] = perioder.power[self.crit]
            if self.num_planets == 0 or self.pers is None:
                self.pers = perioder.pers

            if fixed_threshold is None:
                perioder.eFAP()
                self.eFAPs[self.num_planets] = perioder.fap_min
            else:
                perioder.bic_thresh = fixed_threshold
            self.bic_threshes[self.num_planets] = perioder.bic_thresh
            self.best_bics[self.num_planets] = perioder.best_bic

            if self.save_outputs:
                perioder.plot_per()
                perioder.fig.savefig(outdir+'/dbic{}.pdf'.format(
                                     self.num_planets+1))

            # Check whether there is a detection. If so, fit free and proceed.
            if perioder.best_bic > perioder.bic_thresh:
                self.num_planets += 1
                for k in self.post.params.keys():
                    self.post.params[k].value = perioder.bestfit_params[k]

                # Generalize tc reset to each new discovery.
                tckey = 'tc{}'.format(self.num_planets)
                if self.post.params[tckey].value < np.amin(self.data.time):
                    self.post.params[tckey].value = np.median(self.data.time)
                    for n in np.arange(1, self.num_planets+1):
                        self.post.params['k{}'.format(n)].vary = False
                        self.post.params['per{}'.format(n)].vary = False
                        self.post.params['secosw{}'.format(n)].vary = False
                        self.post.params['secosw{}'.format(n)].vary = False
                        if n != self.num_planets:
                            self.post.params['tc{}'.format(n)].vary = False

                    self.post = radvel.fitting.maxlike_fitting(self.post,
                                                               verbose=False)

                    for n in np.arange(1, self.num_planets+1):
                        self.post.params['k{}'.format(n)].vary = True
                        self.post.params['per{}'.format(n)].vary = True
                        self.post.params['secosw{}'.format(n)].vary = True
                        self.post.params['secosw{}'.format(n)].vary = True
                        self.post.params['tc{}'.format(n)].vary = True

                self.fit_orbit()
                self.all_params.append(self.post.params)
                self.basebic = self.post.likelihood.bic()
            else:
                self.sub_planet()
                # 8/3: Update the basebic anyway, for injections.
                self.basebic = self.post.likelihood.bic()
                run = False
            if self.num_planets >= self.max_planets:
                run = False

            # If any jitter values are negative, flip them.
            for key in self.post.params.keys():
                if 'jit' in key:
                    if self.post.params[key].value < 0:
                        self.post.params[key].value = -self.post.params[key].value

            # Generate an orbit plot.
            if self.save_outputs:
                rvplot = orbit_plots.MultipanelPlot(self.post, saveplot=outdir +
                                                    '/orbit_plot{}.pdf'.format(
                                                    self.num_planets))
                multiplot_fig, ax_list = rvplot.plot_multipanel()
                multiplot_fig.savefig(outdir+'/orbit_plot{}.pdf'.format(
                                                        self.num_planets))

        # Generate running periodograms.
        self.running_per()

        # Run MCMC on final posterior, save new parameters and uncertainties.
        if self.mcmc == True and (self.num_planets != 0 or
                                  self.post.params['dvdt'].vary == True):
            self.post.uparams   = {}
            self.post.medparams = {}
            self.post.maxparams = {}
            # Use recommended parameters for mcmc.
            nensembles = np.min([self.workers, 16])
            if os.cpu_count() < nensembles:
                nensembles = os.cpu_count()
            # Set custom mcmc scales for e/w parameters.
            for n in np.arange(1, self.num_planets+1):
                self.post.params['secosw{}'.format(n)].mcmcscale = 0.005
                self.post.params['sesinw{}'.format(n)].mcmcscale = 0.005

            # Sample in log-period space.
            '''
            logparams = self.post.params.basis.to_any_basis(
                        self.post.params, 'logper tc secosw sesinw k')
            logpost = copy.deepcopy(self.post)
            logpost.params = logparams
            '''
            # Run MCMC.
            chains = radvel.mcmc(self.post, nwalkers=50, nrun=25000,
                                 burnGR=1.03, maxGR=1.0085, minTz=2000,
                                 minsteps=10000, minpercent=33,
                                 thin=5, ensembles=nensembles)

            # Convert chains to per, e, w basis.
            for par in self.post.params.keys():
                if not self.post.params[par].vary:
                    chains[par] = self.post.params[par].value
            synthchains = self.post.params.basis.to_synth(chains)

            quants = chains.quantile([0.159, 0.5, 0.841])
            synthquants = synthchains.quantile([0.159, 0.5, 0.841])

            # Compress, thin, and save chain, in fitting basis.
            csvfn = outdir + '/chains.csv.tar.bz2'
            synthchains.to_csv(csvfn, compression='bz2')

            # Retrieve e and w medians & uncertainties from synthetic chains.
            for n in np.arange(1, self.num_planets+1):
                e_key = 'e{}'.format(n)
                w_key = 'w{}'.format(n)
                # Add period if it's a synthetic parameter.
                '''
                per_key = 'per{}'.format(n)
                logper_key = 'logper{}'.format(n)

                med_per  = synthquants[per_key][0.5]
                high_per = synthquants[per_key][0.841] - med_per
                low_per  = med_per - synthquants[per_key][0.159]
                err_per  = np.mean([high_per,low_per])
                err_per  = radvel.utils.round_sig(err_per)
                med_per, err_per, errhigh_per = radvel.utils.sigfig(med_per,
                                                                    err_per)
                max_per, err_per, errhigh_per = radvel.utils.sigfig(
                               np.exp(self.post.params[logper_key].value), err_per)
                '''

                med_e  = synthquants[e_key][0.5]
                high_e = synthquants[e_key][0.841] - med_e
                low_e  = med_e - synthquants[e_key][0.159]
                err_e  = np.mean([high_e,low_e])
                err_e  = radvel.utils.round_sig(err_e)
                med_e, err_e, errhigh_e = radvel.utils.sigfig(med_e, err_e)
                max_e, err_e, errhigh_e = radvel.utils.sigfig(
                                          self.post.params[e_key].value, err_e)

                med_w  = synthquants[w_key][0.5]
                high_w = synthquants[w_key][0.841] - med_w
                low_w  = med_w - synthquants[w_key][0.159]
                err_w  = np.mean([high_w,low_w])
                err_w  = radvel.utils.round_sig(err_w)
                med_w, err_w, errhigh_w = radvel.utils.sigfig(med_w, err_w)
                max_w, err_w, errhigh_w = radvel.utils.sigfig(
                                          self.post.params[w_key].value, err_w)
                '''
                self.post.uparams[per_key]   = err_per
                self.post.medparams[per_key] = med_per
                self.post.maxparams[per_key] = max_per
                '''
                self.post.uparams[e_key]   = err_e
                self.post.uparams[w_key]   = err_w
                self.post.medparams[e_key] = med_e
                self.post.medparams[w_key] = med_w
                self.post.maxparams[e_key] = max_e
                self.post.maxparams[w_key] = max_w

            # Retrieve medians & uncertainties for the fitting basis parameters.
            for par in self.post.params.keys():
                if self.post.params[par].vary:
                    med = quants[par][0.5]
                    high = quants[par][0.841] - med
                    low = med - quants[par][0.159]
                    err = np.mean([high,low])
                    err = radvel.utils.round_sig(err)
                    med, err, errhigh = radvel.utils.sigfig(med, err)
                    max, err, errhigh = radvel.utils.sigfig(
                                        self.post.params[par].value, err)

                    self.post.uparams[par] = err
                    self.post.medparams[par] = med
                    self.post.maxparams[par] = max

            # Add uncertainties on derived parameters, if mass is provided.
            if self.mstar is not None:
                self.post = utils.derive(self.post, synthchains,
                                         self.mstar, self.mstar_err)

            if self.save_outputs:
                # Generate a corner plot, sans nuisance parameters.
                labels = []
                for n in np.arange(1, self.num_planets+1):
                    labels.append('per{}'.format(n))
                    labels.append('tc{}'.format(n))
                    labels.append('k{}'.format(n))
                    labels.append('secosw{}'.format(n))
                    labels.append('sesinw{}'.format(n))
                if self.post.params['dvdt'].vary == True:
                    labels.append('dvdt')
                if self.post.params['curv'].vary == True:
                    labels.append('curv')
                texlabels = [self.post.params.tex_labels().get(l, l)
                             for l in labels]

                plot = corner.corner(synthchains[labels], labels=texlabels,
                                     label_kwargs={"fontsize": 14},
                                     plot_datapoints=False, bins=30,
                                     quantiles=[0.16, 0.5, 0.84],
                                     title_kwargs={"fontsize": 14},
                                     show_titles=True, smooth=True)
                pl.savefig(outdir+'/{}_corner_plot.pdf'.format(self.starname))

                # Generate an orbit plot wth median parameters and uncertainties.
                rvplot = orbit_plots.MultipanelPlot(self.post,saveplot=
                                outdir+'/orbit_plot_mc_{}.pdf'.format(
                                self.starname), uparams=self.post.uparams)
                multiplot_fig, ax_list = rvplot.plot_multipanel()
                multiplot_fig.savefig(outdir+'/orbit_plot_mc_{}.pdf'.format(
                                      self.starname))

        if self.save_outputs:
            self.save(filename=outdir+'/post_final.pkl')
            pickle_out = open(outdir+'/search.pkl','wb')
            pickle.dump(self, pickle_out)
            pickle_out.close()

            periodograms_plus_pers = np.append([self.pers], list(self.periodograms.values()), axis=0).T
            np.savetxt(outdir+'/pers_periodograms.csv', periodograms_plus_pers,
                       header='period  BIC_array')

            threshs_bics_faps = np.append([list(self.bic_threshes.values())],
                                          [list(self.best_bics.values()),
                                           list(self.eFAPs.values())], axis=0).T

            np.savetxt(outdir+'/thresholds_bics_faps.csv', threshs_bics_faps,
                       header='threshold  best_bic  fap')

    def continue_search(self, fixed_threshold=True):
        """Continue a search by trying to add one more planet

        Args:
            fixed_threshold (bool): fix the BIC threshold at the last threshold, or re-derive for each periodogram
        """
        if self.num_planets == 0:
            self.add_planet()
        last_thresh = max(self.bic_threshes.keys())
        if fixed_threshold:
            thresh = self.bic_threshes[last_thresh]
        else:
            thresh = None

        self.run_search(fixed_threshold=thresh, mkoutdir=False)

    def inject_recover(self, injected_orbel, num_cpus=None, full_grid=False):
        """Inject and recover
        Inject and attempt to recover a synthetic planet signal
        Args:
            injected_orbel (array): array of orbital elements sent to radvel.kepler.rv_drive
            num_cpus (int): Number of CPUs to utilize. Will default to self.workers
            full_grid (bool): if True calculate periodogram on full grid, if False only calculate
                at single period
        Returns:
            tuple: (recovered? (T/F), recovered_orbel)
        """

        if num_cpus is not None:
            self.workers = int(num_cpus)

        self.max_planets = self.num_planets + 1
        self.mcmc = False
        self.save_outputs = False
        self.verbose = False
        # 8/2: Trying to fix injections, possibly basebic error.
        self.basebic = None
        if not full_grid:
            self.manual_grid = [injected_orbel[0]]
            fixed_threshold = True
        else:
            fixed_threshold = False
            # self.manual_grid = self.pers[::4]

        mod = radvel.kepler.rv_drive(self.data['time'].values, injected_orbel)

        self.data['mnvel'] += mod

        self.continue_search(fixed_threshold)

        # Determine successful recovery
        last_planet = self.num_planets
        pl = str(last_planet)
        if last_planet >= self.max_planets:
            synth_params = self.post.params.basis.to_synth(self.post.params)
            recovered_orbel = [synth_params['per'+pl].value,
                               synth_params['tp'+pl].value,
                               synth_params['e'+pl].value,
                               synth_params['w'+pl].value,
                               synth_params['k'+pl].value]
            per, tp, e, w, k = recovered_orbel
            iper, itp, ie, iw, ik = injected_orbel

            # calculate output model to check for phase mismatch
            # probably not most efficient way to do this
            xmod = np.linspace(tp, tp+iper, 100)
            inmod = radvel.kepler.rv_drive(xmod, injected_orbel)
            outmod = self.post.likelihood.model(xmod)
            xph1 = np.mod(xmod - itp, iper)
            xph1 /= iper
            xph2 = np.mod(xmod - tp, per)
            xph2 /= per
            inmin = xph1[np.argmin(inmod)]
            outmin = xph2[np.argmin(outmod)]
            inmax = xph1[np.argmax(inmod)]
            outmax = xph2[np.argmax(outmod)]
            phdiff = np.min([abs(inmin - outmin), abs(outmax - inmax)])

            dthresh = 0.25                                 # recover parameters to 25%
            criteria = [last_planet >= self.max_planets,   # check detected
                        np.abs(per-iper)/iper <= dthresh,  # check periods match
                        phdiff <= np.pi / 6,               # check that phase is right
                        np.abs(k - ik)/ik <= dthresh]      # check that K is right

            criteria = np.array(criteria, dtype=bool)
            if criteria.all():
                recovered = True
            else:
                recovered = False
        else:
            recovered = False
            recovered_orbel = [np.nan for i in range(5)]

        return recovered, recovered_orbel
