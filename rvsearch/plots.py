import pdb
import numpy as np
import pylab as pl
from matplotlib import pyplot as pl
from matplotlib import rcParams, gridspec
from matplotlib.ticker import MaxNLocator, LogFormatterSciNotation, FuncFormatter
from astropy.time import Time

import radvel
from radvel import plot
from radvel.utils import t_to_phase, fastbin, sigfig

import rvsearch.utils as utils
# IMPORTANT: AT SOME POINT, REDEFINE AS CLASS INHERITING FROM RADVEL MULTIPLOT.


class CustomTicker(LogFormatterSciNotation):
    def __call__(self, x, pos=None):
        if x not in [0.1, 1, 10, 100, 1000, 10000]:
            return LogFormatterSciNotation.__call__(self, x, pos=None)
        else:
            return "{x:g}".format(x=x)


class PeriodModelPlot(radvel.plot.orbit_plots.MultipanelPlot):
    """Class to jointly plot the periodograms, best model phaseplots, and
        window function for each search iteration.

    Args:
        search (rvsearch.Search): rvsearch.Search object.
            This includes the periodograms and best-fit RadVel
            posteriors for each added planet.
        saveplot (string): path to save plot
        epoch (int, optional): epoch to subtract off of all time measurements
        yscale_auto (bool, optional): Use matplotlib auto y-axis
             scaling (default: False)
        yscale_sigma (float, optional): Scale y-axis limits for all panels to be +/-
             yscale_sigma*(RMS of data plotted) if yscale_auto==False
        phase_nrows (int, optional): number of columns in the phase
            folded plots. Default is nplanets.
        phase_ncols (int, optional): number of columns in the phase
            folded plots. Default is 1.
        uparams (dict, optional): parameter uncertainties, must
           contain 'per', 'k', and 'e' keys.
        telfmts (dict, optional): dictionary of dictionaries mapping
            instrument suffix to plotting format code. Example:
                telfmts = {
                     'hires': dict(fmt='o',label='HIRES'),
                     'harps-n' dict(fmt='s')
                }
        legend (bool, optional): include legend on plot? Default: True.
        phase_limits (list, optional): two element list specifying
            pyplot.xlim bounds for phase-folded plots. Useful for
            partial orbits.
        nobin (bool, optional): If True do not show binned data on
            phase plots. Will default to True if total number of
            measurements is less then 20.
        phasetext_size (string, optional): fontsize for text in phase plots.
            Choice of {'xx-small', 'x-small', 'small', 'medium', 'large',
            'x-large', 'xx-large'}. Default: 'x-small'.
        rv_phase_space (float, optional): amount of space to leave between orbit/residual plot
            and phase plots.
        figwidth (float, optional): width of the figures to be produced.
            Default: 7.5 (spans a page with 0.5 in margins)
        fit_linewidth (float, optional): linewidth to use for orbit model lines in phase-folded
            plots and residuals plots.
        set_xlim (list of float): limits to use for x-axes of the timeseries and residuals plots, in
            JD - `epoch`. Ex: [7000., 70005.]
        text_size (int): set matplotlib.rcParams['font.size'] (default: 9)
        legend_kwargs (dict): dict of options to pass to legend (plotted in top panel)

    """
    def __init__(self, search, saveplot=None, epoch=2450000, yscale_auto=False,
                 yscale_sigma=3.0, phase_nrows=None, phase_ncols=None,
                 summary_ncols=2, uparams=None, telfmts=plot.telfmts_default,
                 legend=True, phase_limits=[], nobin=False, phasetext_size='small',
                 rv_phase_space=0.06, figwidth=9.5, fit_linewidth=2.0,
                 set_xlim=None, text_size=9, legend_kwargs=dict(loc='best')):

        self.search = search
        self.starname = self.search.starname
        self.post = self.search.post
        self.num_known_planets = self.search.num_planets
        self.pers = self.search.pers
        self.periodograms = self.search.periodograms
        self.bic_threshes = self.search.bic_threshes
        self.fap = self.search.fap

        self.saveplot = saveplot
        self.epoch = epoch
        self.yscale_auto = yscale_auto
        self.yscale_sigma = yscale_sigma
        if phase_nrows is None:
            self.phase_nrows = self.post.likelihood.model.num_planets
        if phase_ncols is None:
            self.phase_ncols = 1
        self.summary_ncols = summary_ncols #Number of columns for phas & pers
        if self.post.uparams is not None:
            self.uparams = self.post.uparams
        else:
            self.uparams = uparams
        self.telfmts = telfmts
        self.legend = legend
        self.phase_limits = phase_limits
        self.nobin = nobin
        self.phasetext_size = phasetext_size
        self.rv_phase_space = rv_phase_space
        self.figwidth = figwidth
        self.fit_linewidth = fit_linewidth
        self.set_xlim = set_xlim
        self.text_size = text_size
        self.legend_kwargs = legend_kwargs

        if isinstance(self.post.likelihood, radvel.likelihood.CompositeLikelihood):
            self.like_list = self.post.likelihood.like_list
        else:
            self.like_list = [ self.post.likelihood ]

        # FIGURE PROVISIONING
        self.ax_rv_height = self.figwidth * 0.6
        self.ax_phase_height = self.ax_rv_height / 1.4
        # Make shorter/wider panels for summary plot with periodograms.
        self.ax_summary_height = self.ax_rv_height / 2

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

    def plot_periodogram(self, pltletter, pnum=0, alias=True, floor=True):
        """Plot periodogram for a given search iteration.

        """
        ax = pl.gca()

        if pnum < self.num_known_planets:
            #Put axis and label on the right side, unless non-detection.
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position('right')

        plot.labelfig(pltletter)

        # TO-DO: WORK IN AIC/BIC OPTION, INCLUDE IN PLOT TITLE
        try:
            peak = np.argmax(self.periodograms[pnum])
        except KeyError:
            # No periodogram for this planet, assume it was previously-known.
            ax = pl.gca()
            ax.annotate('Pre-defined Orbit', xy=(0.5, 0.5),
                        xycoords='axes fraction', horizontalalignment='center',
                        verticalalignment='center', fontsize=self.text_size+8)
            return

        f_real = 1/self.pers[peak]

        # Plot periodogram, and maximum value.
        ax.plot(self.pers, self.periodograms[pnum], c='b')
        ax.scatter(self.pers[peak], self.periodograms[pnum][peak], c='black',
                   label='{} days'.format(np.round(self.pers[peak], decimals=1)))

        # Plot DBIC threshold, set floor periodogram floor
        if pnum == 0:
            fap_label = '{} FAP'.format(self.fap)
        else:
            fap_label = None
        ax.axhline(self.bic_threshes[pnum], ls=':', c='y', label=fap_label)
        upper = 1.1*(max(np.amax(self.periodograms[pnum]),
                         self.bic_threshes[pnum]))

        if floor:
            # Set periodogram plot floor according to circular-fit BIC min.
            lower = -2*np.log(len(self.rvtimes))
        else:
            lower = np.amin(self.periodograms[pnum])

        ax.set_ylim([lower, upper])
        ax.set_xlim([self.pers[0], self.pers[-1]])

        if alias:
            # Plot sidereal day, lunation period, and sidereal year aliases.
            if self.pers[0] <= 1:
                colors = ['r', 'g', 'b']
                alias_preset = [365.256, 29.531, 0.997]
            else:
                colors = ['r', 'g']
                alias_preset = [365.256, 29.531]
            for j in np.arange(len(alias_preset)):
                f_ap = 1./alias_preset[j] + f_real
                f_am = 1./alias_preset[j] - f_real
                if pnum == 0:
                    label = '{} day alias'.format(np.round(
                                                  alias_preset[j], decimals=1))
                else:
                    label = None
                ax.axvline(1./f_am, linestyle='--', c=colors[j], alpha=0.66,
                           label=label)
                ax.axvline(1./f_ap, linestyle='--', c=colors[j], alpha=0.66)
                # Annotate each alias with the associated timescale.
                #ax.text(0.66/f_ap, 0.5*(lower + upper), label, rotation=90,
                #        size=self.phasetext_size, weight='bold',
                #        verticalalignment='bottom') # 0.5*(lower + upper)

        ax.set_xscale('log')
        # TO-DO: WORK IN AIC/BIC OPTION
        ax.set_ylabel(r'$\Delta$BIC$_{}$'.format(pnum+1), fontweight='bold')
        ax.legend(loc=0, prop=dict(size=self.phasetext_size, weight='bold'))
        #          frameon=False, framealpha=0.8)

        # Set tick mark formatting based on gridspec location.
        if pnum < self.num_known_planets:
            ax.tick_params(axis='x', which='both', direction='in',
                           bottom='on', top='on', labelbottom='off')
        elif pnum == self.num_known_planets:
            # Print units and axis label at the bottom.
            ax.set_xlabel('Period [day]', fontweight='bold')
            ax.tick_params(axis='x', which='both', direction='out',
                           bottom='on', top='off', labelbottom='on')
            ax.xaxis.set_major_formatter(CustomTicker())

    def plot_window(self, pltletter):
        """Plot the window function of the data, for each instrument.
        """
        ax = pl.gca()

        #Put axis and label on the right side of the plot.
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position('right')

        ax.set_xlabel('Period [day]', fontweight='bold')
        ax.set_ylabel('Window function power', fontweight='bold')
        ax.set_xscale('log')
        ax.set_xlim([np.amin(self.pers), np.amax(self.pers)])
        plot.labelfig(pltletter)

        # Loop over all instruments, generate separate window function for each.
        for like in self.like_list:
            times    = like.x
            tel      = like.telvec[0]
            baseline = np.amax(times) - np.amin(times)
            window   = utils.window(times, np.flip(1/self.pers))
            if tel in self.telfmts.keys():
                tel = self.telfmts[tel]['label']

            min = np.amax([3, np.amin(self.pers)])
            max = baseline/2

            window_safe = window[np.where(np.logical_and(
                                          self.pers < max, self.pers > min))]
            pers_safe   = self.pers[np.where(np.logical_and(
                                             self.pers < max, self.pers > min))]

            # skip plotting if baseline < min search period
            if len(window_safe) == 0:
                continue

            ax.set_ylim([0, 1.1*np.amax(window_safe)])
            ax.plot(pers_safe, window_safe, alpha=0.75, label=tel)
        ax.xaxis.set_major_formatter(CustomTicker())
        ax.legend()
        #ax.axvspan(np.amin(self.pers), min, alpha=0.25, color='purple')
        #ax.axvspan(max, np.amax(self.pers), alpha=0.25, color='purple')

    def plot_summary(self, letter_labels=True):
        """Provision and plot a search summary plot

        Args:
            letter_labels (bool, optional): if True, include
                letter labels on orbit and residual plots.
                Default: True.

        Returns:
            tuple containing:
                - current matplotlib Figure object
                - list of Axes objects

        """
        scalefactor = self.phase_nrows + 1
        #figheight = self.ax_rv_height + self.ax_phase_height * scalefactor
        figheight = self.ax_rv_height + self.ax_summary_height * scalefactor

        # provision figure
        fig = pl.figure(figsize=(self.figwidth, figheight))
        right_edge = 0.90
        top_edge = 0.92
        bottom_edge = 0.05

        fig.subplots_adjust(left=0.12, right=right_edge)
        gs_rv = gridspec.GridSpec(2, 1, height_ratios=[1., 0.5])

        divide = 0.95 - self.ax_rv_height / figheight
        ipl = 5 - self.num_known_planets
        while ipl > 0:
            top_edge -= 0.01
            bottom_edge += 0.005
            divide += 0.015
            self.rv_phase_space += 0.012
            ipl -= 1
        gs_rv.update(left=0.12, right=right_edge, top=top_edge,
                     bottom=divide+self.rv_phase_space*0.5, hspace=0.)

        # orbit plot
        ax_rv = pl.subplot(gs_rv[0, 0])
        self.ax_list += [ax_rv]

        pl.sca(ax_rv)
        self.plot_timeseries()
        if letter_labels:
            pltletter = ord('a')
            plot.labelfig(pltletter)
            pltletter += 1

        # residuals
        ax_resid = pl.subplot(gs_rv[1, 0])
        self.ax_list += [ax_resid]

        pl.sca(ax_resid)
        self.plot_residuals()
        if letter_labels:
            plot.labelfig(pltletter)
            pltletter += 1

        # phase-folded plots and periodograms
        gs_phase = gridspec.GridSpec(self.phase_nrows+1, self.summary_ncols)

        if self.summary_ncols == 1:
            gs_phase.update(left=0.12, right=right_edge,
                            top=divide - self.rv_phase_space * 0.2,
                            bottom=bottom_edge, hspace=0.003)
        else:
            gs_phase.update(left=0.12, right=right_edge,
                            top=divide - self.rv_phase_space * 0.2,
                            bottom=bottom_edge, hspace=0.003, wspace=0.05)

        for i in range(self.num_planets):
            # Plot phase.
            # i_row = int(i / self.summary_ncols)
            i_row = i
            # i_col = int(i - i_row * self.summary_ncols)
            i_col = 0
            ax_phase = pl.subplot(gs_phase[i_row, i_col])
            self.ax_list += [ax_phase]

            pl.sca(ax_phase)
            self.plot_phasefold(pltletter, i+1)
            pltletter += 1

            # Plot periodogram.
            i_row = i
            i_col = 1
            ax_per = pl.subplot(gs_phase[i_row, i_col])
            self.ax_list += [ax_per]

            pl.sca(ax_per)
            self.plot_periodogram(pltletter, i)
            pltletter += 1

        # Plot final row, window function & non-detection.
        '''
        gs_phase.update(left=0.12, right=0.93,
                        top=divide - self.rv_phase_space * 0.2,
                        bottom=0.07, hspace=0.003, wspace=0.05)
        '''
        ax_non = pl.subplot(gs_phase[self.num_planets, 0])
        self.ax_list += [ax_non]
        pl.sca(ax_non)
        self.plot_periodogram(pltletter, self.num_planets)
        pltletter += 1

        ax_window = pl.subplot(gs_phase[self.num_planets, 1])
        self.ax_list += [ax_window]
        pl.sca(ax_window)
        self.plot_window(pltletter)
        pltletter += 1

        pl.suptitle(self.search.starname, fontsize=self.text_size+6, weight='bold')

        if self.saveplot is not None:
            pl.savefig(self.saveplot, dpi=150)
            print("Search summary plot saved to %s" % self.saveplot)

        return fig, self.ax_list


class CompletenessPlots(object):
    """Class to plot results of injection/recovery tests

    Args:
        completeness (inject.Completeness): completeness object

    """
    def __init__(self, completeness):
        self.comp = completeness

        self.xlim = (min(completeness.recoveries[completeness.xcol]),
                     max(completeness.recoveries[completeness.xcol]))

        self.ylim = (min(completeness.recoveries[completeness.ycol]),
                     max(completeness.recoveries[completeness.ycol]))

        self.xgrid, self.ygrid, self.comp_array = completeness.completeness_grid(self.xlim, self.ylim)

    def completeness_plot(self, title='', xlabel='', ylabel=''):
        """Plot completeness contours

        Args:
            title (string): (optional) plot title
            xlabel (string): (optional) x-axis label
            ylabel (string): (optional) y-axis label
        """
        good = self.comp.recoveries.query('recovered == True')
        bad = self.comp.recoveries.query('recovered == False')

        fig = pl.figure(figsize=(5, 3.5))
        pl.subplots_adjust(bottom=0.18, left=0.22, right=0.95)

        CS = pl.contourf(self.xgrid, self.ygrid, self.comp_array, 10, cmap=pl.cm.Reds_r, vmax=0.9)
        pl.plot(good[self.comp.xcol], good[self.comp.ycol], 'b.', alpha=0.3, label='recovered')
        pl.plot(bad[self.comp.xcol], bad[self.comp.ycol], 'r.', alpha=0.3, label='missed')
        ax = pl.gca()
        ax.set_xscale('log')
        ax.set_yscale('log')

        xticks = pl.xticks()[0]
        pl.xticks(xticks, xticks)

        yticks = pl.yticks()[0]
        pl.yticks(yticks, yticks)

        pl.xlim(self.xlim[0], self.xlim[1])
        pl.ylim(self.ylim[0], self.ylim[1])

        pl.title(title)
        pl.xlabel(xlabel)
        pl.ylabel(ylabel)

        pl.grid(True)

        fig = pl.gcf()

        return fig
