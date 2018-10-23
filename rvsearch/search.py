#Search class.
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

    def __init__(self, data, starname='', max_planets=5, priors=[], crit='bic'):
        if {'time', 'mnvel', 'errvel', 'tel'}.issubset(data.columns):
            self.data = data
            self.tels = np.unique(self.data['tel'].values)
            #self.tels = set(self.data.tel) #Unique list of telescopes.
        else:
            raise ValueError('Incorrect data input.')

        self.starname = starname
        self.params = utils.initialize_default_pars(instnames=self.tels)
        self.priors = priors

        #Default pdict can contain known planet parameters, contains nplanets
            #Change it to an rvparams object, includes functionality of param objects. *init_params*
        #self.default_pdict = default_pdict
        #self.default_params = HOW DO WE DO THIS? 10/18

        #self.all_posts = []

        self.post = utils.initialize_post(self.data, self.params)

        #TRYING TO GENERALIZE INFORMATION CRITERION TO AIC OR BIC.
        '''
        if crit=='bic':
            self.crit = radvel.posterior.bic()
        eif crit=='aic':
            self.crit = radvel.posterior.aic()
        self.critname = self.crit.__string__
        else:
            raise ValueError('Invalid information criterion.')
        #Play with calling __name__ of method
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

        #for par in self.post.likelihood.extra_params: #WHAT DOES THIS MEAN? ASK LEA
        #    new_params[par] = radvel.Parameter(value=self.default_pdict[par])
        '''
        for k in self.post.params.keys():
            new_params[k] = self.post.params[k].value
        '''
        #Set default parameters for n+1th planet
        default_params = utils.initialize_default_pars(self.tels)#FIX INSTRUMENT_NAME PROBLEM 10/22/18
        for par in param_list:
            parkey = par + str(new_num_planets)
            onepar = par + '1' #MESSY, FIX THIS 10/22/18
            new_params[parkey] = default_params[onepar]

        new_params['dvdt'] = radvel.Parameter(value=self.post.params['dvdt'])
        new_params['curv'] = radvel.Parameter(value=self.post.params['curv'])

        if self.post.params['dvdt'].vary == False:
        	new_params['dvdt'].vary = False
        if self.post.params['curv'].vary == False:
        	new_params['curv'].vary = False

        new_params['per{}'.format(new_num_planets)].vary = False
        new_params['secosw{}'.format(new_num_planets)].vary = False
        new_params['sesinw{}'.format(new_num_planets)].vary = False
        '''
        for inst in self.tels:
            new_params['gamma_'+inst] = self.post.params['gamma_'+inst]
            new_params['jit_'+inst] = self.post.params['jit_'+inst]
        '''
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
            Allow phase, period to vary within ~5-10% of original value, ask Andrew
            Or allow *all* params (except curv, dvdt)to be totally free. This is while making per_bics
            Make 1 flag each for dvdt, curv at the start of the search. In search object
                Force on, force off, or auto for 0-1 model. Off for Legacy

        6. Set number of planets in new_param? Not already done? !
        7. Add positive amp. & ecc. priors, set self.post to new posterior !
        8. Save old posterior to list containing history of posterior
        '''
        pass

    def sub_planet(self):
        #self.posts = self.posts[:-1] Not quite right
        pass

    def fit_orbit(self):
        #Redundant with add_planet (current_planets, fitting_basis)? Make class properties?
        #current_planets = self.post.params.num_planets
        fit = radvel.fitting.maxlike_fitting(self.post, verbose=False)
        
        self.post = fit

    '''
    def add_gp(self):
        pass

    def sub_gp(self):
        try:
            sub_gp
        except:
            raise RuntimeError('Model does not contain a Gaussian process.')
    '''

    def save(self, post, filename=None):
        if filename != None:
            post.writeto(filename)
        else:
            post.writeto('post_final.pkl')
        #Write this so that it can be iteratively applied with each planet addition.

    def plot_model(self, post):
        pass

    def save_all_posts(self):
        #Return list of posteriors for each nth planet model
        #self.all_posts
        pass

    def run_search(self):
        #Use all of the above routines to run a search.
        pass
