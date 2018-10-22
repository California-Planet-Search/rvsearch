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
        fitting_basis = self.post.params.basis
        param_list = fittin_basis.split()

        new_num_planets = current_num_planets + 1

        default_pars = utils.initialize_default_pars(instnames=self.tels)
        new_params = radvel.Parameters(new_num_planets, basis=fitting_basis)

        #THIS IS WRONG, DOESN'T SET 1-NTH PLANET PARAMETERS PROPERLY. ASK BJ
        for planet in np.arange(1, new_num_planets):
            for par in param_list:
                parkey = par + str(planet)
                new_params[parkey] = self.post.params[parkey]
                '''
                if parkey in self.default_pdict.keys():
                    val = radvel.Parameter(value=self.default_pdict[parkey])
                else:
                    parkey1 = parkey[:-1] + '1' #WHAT DOES THIS MEAN?
                    val = radvel.Parameter(value=default_pars[parkey1].value)
                new_params[parkey] = val
                '''
        for par in self.post.likelihood.extra_params: #WHAT DOES THIS MEAN
            new_params[par] = radvel.Parameter(value=self.default_pdict[par])

        #Set default parameters for n+1th planet
        default_params = utils.initialize_default_pars()#FIX INSTRUMENT_NAME PROBLEM
        for par in param_list:
            parkey = par + str(new_num_planets)
            new_params[parkey] = default_params[parkey]

        new_params['dvdt'] = radvel.Parameter(value=default_pdict['dvdt'])
        new_params['curv'] = radvel.Parameter(value=default_pdict['curv'])

        if self.post.params['dvdt'].vary == False:
        	new_params['dvdt'].vary = False
        if self.post.params['curv'].vary == False:
        	new_params['curv'].vary = False

        new_params['per{}'.format(new_num_planets)].vary = False
        new_params['secosw{}'.format(new_num_planets)].vary = False
        new_params['sesinw{}'.format(new_num_planets)].vary = False

        new_params.num_planets = new_num_planets

        priors = [radvel.prior.HardBounds('jit'+inst, 0.0, 20.0) for inst in self.tels]
        priors.append(radvel.prior.PositiveKPrior(new_num_planets))
        priors.append(radvel.prior.EccentricityPrior(new_num_planets))

        new_post = utils.initialize_post(new_params, self.data, priors)
        self.post = new_post

        '''
        1. Get default values for new planet parameters
        2. Initialize new radvel Parameter object, new_param, with n+1 planets !
        3. TO COMPLETE Set values of 1st - nth planet in new_param TO COMPLETE
        4. Set curvature fit parameters, check locked or unlocked

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
        current_planets = self.post.params.num_planets
        fitting_basis = self.post.params.basis.name
        param_list = fitting_basis.split()

        self.post.params['k{}'.format(current_planets)].vary = True #to initialize 1 planet bic
        self.post.params['tc{}'.format(current_planets)].vary = True
        self.post.params['secosw{}'.format(current_planets)].vary = True
        self.post.params['sesinw{}'.format(current_planets)].vary = True

        fit = radvel.fitting.maxlike_fitting(self.post, verbose=False)

        self.post.params['k{}'.format(current_planets)].vary = False #to initialize 1 planet bic
        self.post.params['tc{}'.format(current_planets)].vary = False
        self.post.params['secosw{}'.format(current_planets)].vary = False
        self.post.params['sesinw{}'.format(current_planets)].vary = False

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

    '''
    def nominal_model(self, data, starname, fitting_basis='per tc secosw sesinw k'):
    	"""Define the default nominal model. If binary orbit, read from table. Otherwise,
    		define from default pars.
    	"""
    	binary_table = pd.read_csv('binary_orbits.csv',
    								dtype={'star':str,'per':np.float64,'tc':np.float64,
    										'e':np.float64,'w':np.float64,'k':np.float64})
    	binlist = binary_table['star'].values

    	instnames = np.unique(data['tel'].values)
    	priors = [radvel.prior.HardBounds('jit_'+inst, 0.0, 20.0) for inst in instnames]
    	priors.append(radvel.prior.PositiveKPrior( 1 ))

    	#pdb.set_trace()
    	bin_orb = binary_table[binary_table['star']==starname]

    	#pdb.set_trace()

    	if len(bin_orb) > 0:
    		anybasis_params = radvel.Parameters(num_planets=1, basis='per tc e w k')
    		anybasis_params['tc1'] = radvel.Parameter(value=bin_orb['tc'].values[0])
    		anybasis_params['w1'] = radvel.Parameter(value=bin_orb['w'].values[0])
    		anybasis_params['k1'] = radvel.Parameter(value=bin_orb['k'].values[0])
    		anybasis_params['e1'] = radvel.Parameter(value=bin_orb['e'].values[0])
    		anybasis_params['per1'] = radvel.Parameter(value=bin_orb['per'].values[0])

    		anybasis_params['dvdt'] = radvel.Parameter(value=0.0)
    		anybasis_params['curv'] = radvel.Parameter(value=0.0)

    		for inst in instnames:
    			anybasis_params['gamma_'+inst] = radvel.Parameter(value=0.0)
    			anybasis_params['jit_'+inst] = radvel.Parameter(value=2.0)
    		#pdb.set_trace()
    		params = anybasis_params.basis.to_any_basis(anybasis_params, fitting_basis)
    		params['dvdt'].vary = False
    		params['curv'].vary = False
    		planet_num = 2

    		priors.append(radvel.prior.EccentricityPrior( 1 ))
    		post = ut.initialize_post(params, data, priors)
    		#pdb.set_trace()
    		post = radvel.fitting.maxlike_fitting(post) #Fit the best params for binary

    		default_pdict = {} #Save the binary orbital params
    		for k in post.params.keys():
    			default_pdict[k] = post.params[k].value

    		#Now add the planet onto the binary, with K==0 and no new free params
    		post = ut.add_planet(post, default_pdict, data)

    	else:
    		planet_num = 1
    		params = ut.initialize_default_pars(instnames)

    		priors.append(radvel.prior.EccentricityPrior( 1 ))
    		post = ut.initialize_post(params, data, priors)
    		post = radvel.fitting.maxlike_fitting(post, verbose=False)

    	return post, priors, planet_num
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
