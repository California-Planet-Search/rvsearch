    #Search class.

class Search(object):

    def __init__(self, t, v, verr, aic=False):
        self.t, self.v, self.verr = t, v, verr
        '''
        if aic==False:
            self.crit = radvel.posterior.bic()
        else:
            self.crit = radvel.posterior.aic()
        self.critname = self.crit.__string__
        '''

    def initialize_post(self, params, data, priors):
        """Initialize a posterior object with data, params, and priors

        Args:
            params (radvel Parameters object)
            data (DataFrame): Must have column names 'time', 'mnvel', 'errve', 'tel'
            priors (list): radvel Priors objects

        Returns:
            post (radvel Posterior object)
        """

        iparams = radvel.basis._copy_params(params)

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

        like = radvel.likelihood.CompositeLikelihood(list(likes.values()))

        post = radvel.posterior.Posterior(like)
        post.priors = priors

        self.post = post
        #return post

    def add_planet(self, post, default_pdict, data):
        current_planets = post.params.num_planets
        fitting_basis = post.params.basis.name
        param_list = fitting_basis.split()

        new_planet_index = current_planets + 1

        #Default values for new planet
        def_pars = initialize_default_pars([], fitting_basis=fitting_basis)

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
        #pdb.set_trace()

        new_post = self.setup_post(new_params, data, priors)

        self.post = new_post
        #return new_post

    def sub_planet(self, post):
        pass

    def save(self, post, filename=None):
        if filename != None:
            post.writeto(filename)
        else:
            post.writeto('post.pkl')
        #Write this so that it can be iteratively applied with each planet addition.

    def plot_model(self, post):
        pass

    def all_posts(self):
        #Return list of posteriors for each nth planet model
        pass
