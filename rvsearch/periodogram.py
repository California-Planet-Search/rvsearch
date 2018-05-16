import numpy as np


class Periodogram(object):
    """
    Class to calculate and store periodograms.

    Args:
        posterior (radvel.Posterior): radvel.Posterior object
        minsearchP (float): minimum search period
        maxsearchP (float): maximum search period
        num_known_planets (int): Assume this many known planets in the system and search for one more
        num_freqs (int): (optional) number of frequencies to test [default = calculated via rvsearch.periodograms.freq_spacing]
    """

    def __init__(self, posterior, minsearchP, maxsearchP, num_known_planets=0, num_freqs=None):
        self.minsearchP = minsearchP
        self.maxsearchP = maxsearchP
        self.num_freqs = num_freqs

        self.num_known_planets = num_known_planets

        self.post = self.setup_posterior(posterior, num_known_planets)

        if num_freqs is None:
            self.per_array = freq_spacing(self.post.time, self.minsearchP, self.maxsearchP)
        else:
            self.per_array = 1/np.linspace(1/self.maxsearchP, 1/self.minsearchP, self.num_freqs)

        self.freq_array = 1/self.per_array

    def setup_posterior(self, post, num_known_planets):
        """Setup radvel.posterior.Posterior object

        Prepare posterior object for periodogram caclulations. Fix values for previously-known planets.

        Args:
            post (radvel.posterior.Posterior): RadVel posterior object. Can be initialized from setup file or loaded
                from a RadVel fit.
            num_known_planets (int): Number of previously known planets. Parameters for these planets will be fixed.

        Returns:
            radvel.posterior.Posterior: augmented posterior object

        """

        basis_name = post.params.basis.name
        basis_pars = basis_name.split()
        for i in range(1, num_known_planets+1):
            for par in basis_pars:
                parname = "{}{}".format(par, i)
                post.params[parname].vary = False

        # Needed for pyTiming periodograms
        post.time = post.likelihood.x
        post.flux = post.likelihood.y
        post.error = post.likelihood.yerr

        print(post)

        return post

    def bic(self):
        """Compute delta-BIC periodogram"""
        pass

    def gls(self):
        print("Calculating GLS periodogram")
        gls = pyTiming.pyPeriod.Gls(self.post, freq=self.freq_array)

        self.power = gls.power

def freq_spacing(times, minp, maxp, oversampling=1, verbose=True):
    """Get the number of sampled frequencies

    Condition for spacing: delta nu such that during the
    entire duration of observations, phase slip is no more than P/4

    Args:
        times (array): array of timestamps
        minP (float): minimum period
        maxP (float): maximum period
        oversampling (float): (optional) oversampling factor
        verbose (bool): (optional) print extra messages

    Returns:
        array: Array of test periods
    """

    fmin = 1 / maxp
    fmax = 1 / minp

    timlen = max(times) - min(times)
    dnu = 1. / (4. * timlen)
    numf = int((fmax - fmin) / dnu + 1)
    res = numf
    res *= oversampling

    if verbose:
        print("Number of test periods:", res)

    Farr = np.linspace(1 / maxp, 1 / minp, res)
    Parr = 1 / Farr

    return Parr



