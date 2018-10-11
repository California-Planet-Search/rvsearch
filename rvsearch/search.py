#Search class.
import numpy as np
import pandas
import radvel
import radvel.fitting

import utils

class Search(object):
    """Class to initialize and modify posteriors as planet search runs,
    send to Periodogram class for periodogram and IC-threshold calculations.

    Args:
        data: pandas dataframe containing times, velocities, velocity errors, and telescope types.
        params: List of radvel parameter objects.
        priors: List of radvel prior objects.
        aic: if True, use Akaike information criterion instead of BIC. STILL WORKING ON THIS
    """

    def __init__(self, data, starname, max_planets=5,
                 params=[], priors=[], default_pdict=[], aic=False):
        if {'time', 'mnvel', 'errvel', 'tel'}.issubset(data.columns):
            self.data = data
            self.tels = set(self.data.tel) #Unique list of telescopes.
        else:
            raise ValueError('Incorrect data input.')

        self.starname = starname
        #self.params = params
        #self.params = radvel.Parameters(1, basis='per tc secosw sesinw logk')
        self.params = utils.initialize_default_pars(instnames=self.tels)
        #Might not need priors?
        self.priors = priors
        self.default_pdict = default_pdict
        self.all_posts = []

        #TRYING TO GENERALIZE INFORMATION CRITERION TO AIC OR BIC.
        '''
        if aic==False:
            self.crit = radvel.posterior.bic()
        else:
            self.crit = radvel.posterior.aic()
        self.critname = self.crit.__string__
        '''

    def initialize_post(self):
        #TO-DO: DEFINE 'DATA' INPUT, FIGURE OUT WHICH DATAFRAME FORMAT
        """Initialize a posterior object with data, params, and priors

        Args:

        Returns:
            post (radvel Posterior object)
        """

        iparams = radvel.basis._copy_params(self.params)

        #initialize RVModel
        time_base = np.mean([self.data['time'].max(), self.data['time'].min()])
        mod = radvel.RVModel(self.params, time_base=time_base)

        #initialize Likelihood objects for each instrument
        telgrps = self.data.groupby('tel').groups
        likes = {}

        for inst in telgrps.keys():
            likes[inst] = radvel.likelihood.RVLikelihood(
                mod, self.data.iloc[telgrps[inst]].time, self.data.iloc[telgrps[inst]].mnvel,
                self.data.iloc[telgrps[inst]].errvel, suffix='_'+inst)

            likes[inst].params['gamma_'+inst] = iparams['gamma_'+inst]
            likes[inst].params['jit_'+inst] = iparams['jit_'+inst]

        like = radvel.likelihood.CompositeLikelihood(list(likes.values()))

        post = radvel.posterior.Posterior(like)
        post.priors = self.priors

        self.post = post
        #return post

    def add_planet(self):
        current_planets = self.post.params.num_planets
        fitting_basis = self.post.params.basis.name
        param_list = fitting_basis.split()

        new_planet_index = current_planets + 1

        #Default values for new planet
        def_pars = utils.initialize_default_pars([], fitting_basis=fitting_basis)
        new_params = radvel.Parameters(new_planet_index, basis=fitting_basis)

        for pl in range(1, new_planet_index+1):
            for par in param_list:
                parkey = par + str(pl)

                if parkey in default_pdict.keys():
                    val = radvel.Parameter(value=default_pdict[parkey])
                else:
           		    parkey1 = parkey[:-1] + '1'
           		    val = radvel.Parameter(value=def_pars[parkey1].value)

                new_params[parkey] = val

        for par in post.likelihood.extra_params:
            new_params[par] = radvel.Parameter(value=default_pdict[par])

        new_params['dvdt'] = radvel.Parameter(value=default_pdict['dvdt'])
        new_params['curv'] = radvel.Parameter(value=default_pdict['curv'])

        if post.params['dvdt'].vary == False:
        	new_params['dvdt'].vary = False
        if post.params['curv'].vary == False:
        	new_params['curv'].vary = False

        new_params['k{}'.format(new_planet_index)].vary = False #to initialize 1 planet bic
        new_params['tc{}'.format(new_planet_index)].vary = False
        new_params['per{}'.format(new_planet_index)].vary = False
        new_params['secosw{}'.format(new_planet_index)].vary = False
        new_params['sesinw{}'.format(new_planet_index)].vary = False

        new_params.num_planets = new_planet_index

        instnames = np.unique(data['tel'].values)
        priors = [radvel.prior.HardBounds('jit_'+inst, 0.0, 20.0) for inst in instnames]
        priors.append(radvel.prior.PositiveKPrior( new_planet_index ))
        priors.append(radvel.prior.EccentricityPrior( new_planet_index ))

        #FIX THIS, NO 'SETUP_POST IN OUR CODE YET'
        new_post = self.setup_post(new_params, data, priors)

        self.post = new_post
        #return new_post

    def sub_planet(self):
        self.posts = self.posts[:-1]

    def fit_orbit(self):
        #Redundant with add_planet (current_planets, fitting_basis), make class properties?
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
