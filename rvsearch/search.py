    #Search class.

class Search(object):

    def __init__(self, t, v, verr, aic=False):
        self.t, self.v, self.verr = t, v, verr
        if aic==False:
            self.crit = radvel.Posterior.bic()
        else:
            self.crit = radvel.Posterior.aic()
        self.critname = self.crit.__string__

    def setup_post(self):
        #TO-DO: FINISH, NEEDED TO PASS TO PERIODOGRAM
        self.post = post

    def add_planet(self, post, default_pdict, data):
        #Attempting to modify Lea's add_planet function to fit this class.
        pass

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
