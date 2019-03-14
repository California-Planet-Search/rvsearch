import numpy as np
import pylab as pl
from matplotlib import pyplot as plt
import radvel


class PeriodModelPlot(object):
    """Class to jointly plot the periodograms, best model phaseplots, and
        window function for each search iteration.

    Args:
        search (rvsearch.Search): rvsearch.Search object.
            This includes the periodograms and best-fit RadVel
            posteriors for each added planet.

    """
    def __init__(self, search, saveplot=None, epoch=2450000, phase_nrows=None,
                 yscale_auto=False, yscale_sigma=3.0, phase_ncols=None,
                 uparams=None, telfmts={}, legend=True, nobin=False,
                 phasetext_size='large', rv_phase_space=0.08, figwidth=7.5,
                 fit_linewidth=2.0, set_xlim=None, text_size=9,
                 legend_kwards=dict(loc='best')):

        self.search = search
        self.post = self.search.post
        self.num_known_planets = self.search.num_planets
        self.pers = self.search.pers
        self.periodograms = self.search.periodograms
        self.bic_threshes = self.search.bic_threshes

        self.saveplot = saveplot
        self.epoch = epoch
        self.phase_nrows = phase_nrows
        self.phase_ncols = phase_ncols
        self.uparams = None
        self.telfmts = telfmts
        self.legend = legend
        self.nobin = nobin
        self.phasetext_size = phasetext_size
        self.rv_phase_space = rv_phase_space
        self.figwidth = figwidth
        self.fit_linewidth = fit_linewidth
        self.set_lim = set_xlim
        self.text_size =etext_size
        self.legend_kwargs = legend_kwargs

        if isinstance(self.post.likelihood, radvel,likelihood.CompositeLikelihood):
            self.like_list = self.post.likelihood.like_list
        else:
            self.like_list = [ self.post.likelihood ]

        # FIGURE PROVISIONING
        self.ax_rv_height = self.figwidth * 0.6
        self.ax_phase_height = self.ax_rv_height / 1.4

        # convert params to synth basis
        synthparams = self.post.params.basis.to_synth(self.post.params)
        self.post.params.update(synthparams)

        self.model = self.post.likelihood.model
        self.rvtimes = self.post.likelihood.x
        self.rverr = self.post.likelihood.errorbars()
        self.num_planets = self.model.num_planets

        self.rawresid = self.post.likelihood.residuals()

        self.resid = (
            self.rawresid + self.post.params['dvdt'].value*(self.rvtimes-self.model.time_base)
            + self.post.params['curv'].value*(self.rvtimes-self.model.time_base)**2
        )

        if self.saveplot is not None:
            resolution = 10000
        else:
            resolution = 2000

        periods = []
        for i in range(self.num_planets):
            periods.append(synthparams['per%d' % (i+1)].value)
        if len(periods) > 0:
            longp = max(periods)
        else:
            longp = max(self.post.likelihood.x) - min(self.post.likelihood.x)

        self.dt = max(self.rvtimes) - min(self.rvtimes)
        self.rvmodt = np.linspace(
            min(self.rvtimes) - 0.05 * self.dt, max(self.rvtimes) + 0.05 * self.dt + longp,
            int(resolution)
        )

        self.orbit_model = self.model(self.rvmodt)
        self.rvmod = self.model(self.rvtimes)

        if ((self.rvtimes - self.epoch) < -2.4e6).any():
            self.plttimes = self.rvtimes
            self.mplttimes = self.rvmodt
        elif self.epoch == 0:
            self.epoch = 2450000
            self.plttimes = self.rvtimes - self.epoch
            self.mplttimes = self.rvmodt - self.epoch
        else:
           self.plttimes = self.rvtimes - self.epoch
           self.mplttimes = self.rvmodt - self.epoch


        self.slope = (
            self.post.params['dvdt'].value * (self.rvmodt-self.model.time_base)
            + self.post.params['curv'].value * (self.rvmodt-self.model.time_base)**2
        )
        self.slope_low = (
            self.post.params['dvdt'].value * (self.rvtimes-self.model.time_base)
            + self.post.params['curv'].value * (self.rvtimes-self.model.time_base)**2
        )

        # list for Axes objects
        self.ax_list = []

    def plot_timeseries(self):
        """Make a plot of the RV data and model in the current Axes.
        """

        ax = plt.gca()
        ax.hline(0, color='0.5', linestyle='--')

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
            ax.axhline(self.bic_thresh[pnum ], ls=':', c='y', label=r'$\Delta$BIC threshold')
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


class CompletenessPlots(object):
    """Class to plot results of injection/recovery tests

    Args:
        completeness (inject.Completeness): completeness object

    """
    def __init__(self, completeness):
        self.comp = completeness

    @staticmethod
    def completeness_plot(xgrid, ygrid, comp_array, title='', xlabel='', ylabel=''):
        """Plot completeness contours

        Args:
            xgrid (array): grid of x points to plot at
            ygrid (array): grid of y points to plot at
            comp_array (array): array of shape (len(xgrid) x len(ygrid)) with completeness value
                at each combination of xgrid and ygrid
            title (string): (optional) plot title
            xlabel (string): (optional) x-axis label
            ylabel (string): (optional) y-axis label
        """
        CS = pl.contourf(xgrid, ygrid, comp_array, 10, cmap=pl.cm.Reds_r, vmax=0.9)
        ax = pl.gca()
        ax.set_xscale('log')
        ax.set_yscale('log')

        xticks = pl.xticks()[0]
        pl.xticks(xticks, xticks)

        yticks = pl.yticks()[0]
        pl.yticks(yticks, yticks)

        pl.xlim(xgrid.min(), xgrid.max())
        pl.ylim(ygrid.min(), ygrid.max())

        pl.title(title)
        pl.xlabel(xlabel)
        pl.ylabel(ylabel)

        pl.grid(True)

        return CS
