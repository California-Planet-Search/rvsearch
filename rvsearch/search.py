#Search class.
import copy
import time
import pdb

import numpy as np
import pandas
import radvel
import radvel.fitting

import periodogram
import utils
#from .periodogram import *
#from .utils import *
#IS THIS GOOD PRACTICE?

class Search(object):
    """Class to initialize and modify posteriors as planet search runs,
    send to Periodogram class for periodogram and IC-threshold calculations.

    Args:
        data: pandas dataframe containing times, velocities,  errors, and instrument names.
        params: List of radvel parameter objects.
        priors: List of radvel prior objects.
        aic: if True, use Akaike information criterion instead of BIC. STILL WORKING ON THIS
    """

    def __init__(self, data, starname='', max_planets=3, priors=[], crit='bic', verbose=True):
        if {'time', 'mnvel', 'errvel', 'tel'}.issubset(data.columns):
            self.data = data
            self.tels = np.unique(self.data['tel'].values)
        else:
            raise ValueError('Incorrect data input.')

        self.starname = starname
        self.params = utils.initialize_default_pars(instnames=self.tels)
        self.priors = priors

        self.all_posts = []
        self.post = utils.initialize_post(self.data, self.params)

        self.max_planets = max_planets
        self.num_planets = 0
        '''
        #Play with calling __name__ of method
        if crit=='bic':
            self.crit = radvel.posterior.bic()
        eif crit=='aic':
            self.crit = radvel.posterior.aic()
        self.critname = self.crit.__string__
        else:
            raise ValueError('Invalid information criterion.')
        '''

    def add_planet(self):

        current_num_planets = self.post.params.num_planets
        fitting_basis = self.post.params.basis.name
        param_list = fitting_basis.split()

        new_num_planets = current_num_planets + 1

        default_pars = utils.initialize_default_pars(instnames=self.tels)
        new_params = radvel.Parameters(new_num_planets, basis=fitting_basis)

        for planet in np.arange(1, new_num_planets):
            for par in param_list:
                parkey = par + str(planet)
                new_params[parkey] = self.post.params[parkey]

        for par in self.post.likelihood.extra_params:
            new_params[par] = self.post.params[par] #For gamma and jitter
        '''
        for k in self.post.params.keys():
            new_params[k] = self.post.params[k].value
        '''
        #Set default parameters for n+1th planet
        default_params = utils.initialize_default_pars(self.tels)
        for par in param_list:
            parkey = par + str(new_num_planets)
            onepar = par + '1' #MESSY, FIX THIS 10/22/18
            new_params[parkey] = default_params[onepar]

        new_params['dvdt'] = self.post.params['dvdt']
        new_params['curv'] = self.post.params['curv']

        if self.post.params['dvdt'].vary == False:
        	new_params['dvdt'].vary = False
        if self.post.params['curv'].vary == False:
        	new_params['curv'].vary = False

        new_params['per{}'.format(new_num_planets)].vary = False
        new_params['secosw{}'.format(new_num_planets)].vary = False
        new_params['sesinw{}'.format(new_num_planets)].vary = False

        new_params.num_planets = new_num_planets

        priors = [radvel.prior.HardBounds('jit_'+inst, 0.0, 20.0) for inst in self.tels]
        priors.append(radvel.prior.PositiveKPrior(new_num_planets))
        priors.append(radvel.prior.EccentricityPrior(new_num_planets))

        new_post = utils.initialize_post(self.data, new_params, priors)
        self.post = new_post

        '''
        1. Get default values for new planet parameters !
        2. Initialize new radvel Parameter object, new_param, with n+1 planets !
        3. TO COMPLETE Set values of 1st - nth planet in new_param !
        4. Set curvature fit parameters, check locked or unlocked !

        5. Put some kinds of priors on the 1st-nth planet parameters (period, phase)
            Allow phase, period to vary within ~5-10% of original value, ask Andrew.
            Or allow *all* params (except curv, dvdt)to be totally free. This is while making per_bics

            Make 1 flag each for dvdt, curv at the start of the search. In search object
                Force on, force off, or auto for 0-1 model. Off for Legacy

        6. Set number of planets in new_param? Not already done? !
        7. Add positive amp. & ecc. priors, set self.post to new posterior !
        8. Save old posterior to list containing history of posterior
        '''

    def sub_planet(self):
        pass

    def reset_priors(self):
        pass

    def fit_orbit(self):
        #REWRITE TO ITERATE OVER ALL PARAM KEYS? INCLUDING DVDT AND CURV? 10/26/18
        for planet in np.arange(1, self.num_planets+1):
            self.post.params['per{}'.format(planet)].vary = True
            self.post.params['k{}'.format(planet)].vary = True
            self.post.params['tc{}'.format(planet)].vary = True
            self.post.params['secosw{}'.format(planet)].vary = True
            self.post.params['sesinw{}'.format(planet)].vary = True
        fit = radvel.fitting.maxlike_fitting(self.post, verbose=False)
        for planet in np.arange(1, self.num_planets+1):
            self.post.params['per{}'.format(planet)].vary = False
            self.post.params['k{}'.format(planet)].vary = False
            self.post.params['tc{}'.format(planet)].vary = False
            self.post.params['secosw{}'.format(planet)].vary = False
            self.post.params['sesinw{}'.format(planet)].vary = False
        self.post = fit

    def add_gp(self):
        pass

    def sub_gp(self, num_gps=1):
        try:
            pass
        except:
            raise RuntimeError('Model contains fewer than {} Gaussian processes.'.format(num_gps))

    def save(self, post, filename=None):
        if filename != None:
            post.writeto(filename)
        else:
            post.writeto('post_final.pkl')
        #Write this so that it can be iteratively applied with each planet addition.

    def plot_model(self, post):
        pass

    def save_all_posts(self):
        #Pickle the list of posteriors for each nth planet model
        pass

    def run_search(self):
        t1 = time.process_time()
        #Use all of the above routines to run a search. TO-DO: KNOW WHEN TO FIX, FREE PARAMS 10/15/18
        run = True
        while run == True:
            if self.num_planets != 0:
                self.add_planet()
            perioder = periodogram.Periodogram(self.post, num_known_planets=self.num_planets)

            pdb.set_trace()
            t3 = time.process_time()
            perioder.per_bic()
            t4 = time.process_time()
            print('Time = {}'.format(t4 - t3))

            perioder.eFAP_thresh()
            if perioder.best_bic > perioder.bic_thresh:
                perioder.plot_per()
                self.num_planets += 1
                perkey = 'per{}'.format(self.num_planets)
                self.post.params[perkey].vary = False
                self.post.params[perkey].value = perioder.best_per
                self.post.params['k{}'.format(self.num_planets)].value = perioder.best_k
                self.post.params['tc{}'.format(self.num_planets)].value = perioder.best_tc
                self.fit_orbit()
            else:
                self.sub_planet() #FINISH SUB_PLANET() 10/24/18
                run = False
            if self.num_planets >= self.max_planets:
                run = False
        t2 = time.process_time()
        print('Time = {}'.format(t2 - t1))
