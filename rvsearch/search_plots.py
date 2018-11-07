import numpy as np
from matplotlib import rcParams, gridspec
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from astropy.time import Time
import radvel
from radvel import plot
from radvel.utils import t_to_phase, fastbin, sigfig

class PeriodModelPlot(object):
    """Class to jointly plot the periodograms and best models
    for each search iteration. Based on radvel.orbit_plots.MultipanelPlot.

    Args:
        search (rvsearch.Search): rvsearch.Search object.
            This includes the periodograms and best-fit RadVel
            posteriors for each added planet.

    """
    def __init__(self, search, saveplot=None, epoch=2450000):

        self.search = search
        self.num_known_planets = self.search.num_planets
        self.pers = self.search.pers
        self.periodograms = self.search.periodograms
        self.saveplot = saveplot
        self.epoch = epoch

    def plot_timeseries(self):
        pass

    def plot_phasefold(self, pltletter, pnum=0):
        pass

    def plot_periodogram(self, pnum=0):
        # TO-DO: WORK IN AIC/BIC OPTION, INCLUDE IN PLOT TITLE
        peak = np.argmax(self.periodograms[pnum])
        f_real = 1/self.pers[peak]

        fig, ax = plt.subplots()
        ax.plot(self.pers, self.periodograms[pnum])
        ax.scatter(self.pers[peak], self.periodograms[pnum][peak],
                   label='{} days'.format(np.round(self.pers[peak], decimals=1)))

        # If D-BIC threshold has been calculated, plot.
        if self.bic_thresh[pnum] is not None:
            ax.axhline(self.bic_thresh[pnum], ls=':', c='y', label=r'$\Delta$BIC threshold')
            upper = 1.05*max(np.amax(self.periodograms[pnum]), self.bic_thresh)
            ax.set_ylim([np.amin(self.periodograms[pnum]), upper])
        else:
            ax.set_ylim([np.amin(self.periodograms[pnum]), 1.05*np.amax(self.periodograms[pnum])])
        ax.set_xlim([self.pers[0], self.pers[-1]])

        if alias:
            # Plot sidereal day, lunation period, and sidereal year aliases.
            colors = ['r', 'b', 'g']
            alias = [0.997, 29.531, 365.256]
            for i in np.arange(3):
                f_ap = 1./alias[i] + f_real
                f_am = 1./alias[i] - f_real
                ax.axvline(1./f_am, linestyle='--', c=colors[i], alpha=0.5,
                           label='{} day alias'.format(np.round(alias[i], decimals=1)))
                ax.axvline(1./f_ap, linestyle='--', c=colors[i], alpha=0.5)

        ax.legend(loc=0)
        ax.set_xscale('log')
        ax.set_xlabel('Period (days)')
        ax.set_ylabel(r'$\Delta$BIC')  # TO-DO: WORK IN AIC/BIC OPTION
        ax.set_title('Planet {} vs. planet {}'.format(self.num_known_planets+1, self.num_known_planets))

        # Store figure as object attribute, make separate saving functionality?
        self.fig = fig
        if save:
            # FINISH THIS, WRITE NAMING PROCEDURE
            fig.savefig('dbic{}.pdf'.format(self.num_known_planets+1))

    def plot_periodograms_orbits(self):
        """Call everything above to construct a multipanel plot.
        """
        fig = plt.figure()
