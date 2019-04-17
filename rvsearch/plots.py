import pdb
import numpy as np
import pylab as pl
from matplotlib import pyplot as pl
from matplotlib import rcParams, gridspec
from matplotlib.ticker import MaxNLocator
from astropy.time import Time

import radvel
from radvel import plot
from radvel.utils import t_to_phase, fastbin, sigfig

import rvsearch.utils as utils
# IMPORTANT: AT SOME POINT, REDEFINE AS CLASS INHERITING FROM RADVEL MULTIPLOT.

class PeriodModelPlot(object):
    """Class to jointly plot the periodograms, best model phaseplots, and
        window function for each search iteration.

    Args:
        search (rvsearch.Search): rvsearch.Search object.
            This includes the periodograms and best-fit RadVel
            posteriors for each added planet.

    """
    def __init__(self, search, saveplot=None, epoch=2450000, yscale_auto=False,
                 yscale_sigma=3.0, phase_nrows=None, phase_ncols=None,
                 summary_ncols=2, uparams=None, telfmts={}, legend=True,
                 phase_limits=[], nobin=False, phasetext_size='small',
                 rv_phase_space=0.08, figwidth=9.5, fit_linewidth=2.0,
                 set_xlim=None, text_size=9, legend_kwargs=dict(loc='best')):
        '''
        self, search, saveplot=None, epoch=2450000, phase_nrows=None,
        yscale_auto=False, yscale_sigma=3.0, phase_ncols=None,
        uparams=None, telfmts={}, legend=True, nobin=False,
        phasetext_size='large', rv_phase_space=0.08, figwidth=7.5,
        summary_ncols=2, fit_linewidth=2.0, set_xlim=None, text_size=9,
        legend_kwargs=dict(loc='best')):
       '''
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

    def plot_timeseries(self):
        """Make a plot of the RV data and model in the current Axes.

        """
        ax = pl.gca()

        ax.axhline(0, color='0.5', linestyle='--')

        # plot orbit model
        ax.plot(self.mplttimes, self.orbit_model, 'b-', rasterized=False, lw=self.fit_linewidth)

        # plot data
        plot.mtelplot(
            # data = residuals + model
            self.plttimes, self.rawresid+self.rvmod, self.rverr, self.post.likelihood.telvec, ax, telfmts=self.telfmts
        )

        if self.set_xlim is not None:
            ax.set_xlim(self.set_xlim)
        else:
            ax.set_xlim(min(self.plttimes)-0.01*self.dt, max(self.plttimes)+0.01*self.dt)
        pl.setp(ax.get_xticklabels(), visible=False)

        # legend
        if self.legend:
            ax.legend(numpoints=1, **self.legend_kwargs)

        # years on upper axis
        axyrs = ax.twiny()
        xl = np.array(list(ax.get_xlim())) + self.epoch
        decimalyear = Time(xl, format='jd', scale='utc').decimalyear
        axyrs.get_xaxis().get_major_formatter().set_useOffset(False)
        axyrs.set_xlim(*decimalyear)
        axyrs.set_xlabel('{}\nYear'.format(self.starname), fontweight='bold')
        pl.locator_params(axis='x', nbins=5)

        if not self.yscale_auto:
            scale = np.std(self.rawresid+self.rvmod)
            ax.set_ylim(-self.yscale_sigma * scale, self.yscale_sigma * scale)

        ax.set_ylabel('RV [{ms:}]'.format(**plot.latex), weight='bold')
        ticks = ax.yaxis.get_majorticklocs()
        ax.yaxis.set_ticks(ticks[1:])

    def plot_residuals(self):
        """Make a plot of residuals and RV trend in the current Axes.

        """
        ax = pl.gca()

        ax.plot(self.mplttimes, self.slope, 'b-', lw=self.fit_linewidth)

        plot.mtelplot(self.plttimes, self.resid, self.rverr, self.post.likelihood.telvec, ax, telfmts=self.telfmts)
        if not self.yscale_auto:
            scale = np.std(self.resid)
            ax.set_ylim(-self.yscale_sigma * scale, self.yscale_sigma * scale)

        if self.set_xlim is not None:
            ax.set_xlim(self.set_xlim)
        else:
            ax.set_xlim(min(self.plttimes)-0.01*self.dt, max(self.plttimes)+0.01*self.dt)
        ticks = ax.yaxis.get_majorticklocs()
        ax.yaxis.set_ticks([ticks[0], 0.0, ticks[-1]])
        pl.xlabel('JD - {:d}'.format(int(np.round(self.epoch))), weight='bold')
        ax.set_ylabel('Residuals', weight='bold')
        ax.yaxis.set_major_locator(MaxNLocator(5, prune='both'))

    def plot_phasefold(self, pltletter, pnum):
        """Plot phased orbit plots for each planet in the fit.

        Args:
            pltletter (int): integer representation of
                letter to be printed in the corner of the first
                phase plot.
                Ex: ord("a") gives 97, so the input should be 97.
            pnum (int): the number of the planet to be plotted. Must be
                the same as the number used to define a planet's
                Parameter objects (e.g. 'per1' is for planet #1)

        """
        ax = pl.gca()

        if len(self.post.likelihood.x) < 20:
            self.nobin = True

        bin_fac = 1.75
        bin_markersize = bin_fac * rcParams['lines.markersize']
        bin_markeredgewidth = bin_fac * rcParams['lines.markeredgewidth']

        rvmod2 = self.model(self.rvmodt, planet_num=pnum) - self.slope
        modph = t_to_phase(self.post.params, self.rvmodt, pnum, cat=True) - 1
        rvdat = self.rawresid + self.model(self.rvtimes, planet_num=pnum) - self.slope_low
        phase = t_to_phase(self.post.params, self.rvtimes, pnum, cat=True) - 1
        rvdatcat = np.concatenate((rvdat, rvdat))
        rverrcat = np.concatenate((self.rverr, self.rverr))
        rvmod2cat = np.concatenate((rvmod2, rvmod2))
        bint, bindat, binerr = fastbin(phase+1, rvdatcat, nbins=25)
        bint -= 1.0

        ax.axhline(0, color='0.5', linestyle='--', )
        ax.plot(sorted(modph), rvmod2cat[np.argsort(modph)], 'b-', linewidth=self.fit_linewidth)
        plot.labelfig(pltletter)

        telcat = np.concatenate((self.post.likelihood.telvec, self.post.likelihood.telvec))

        plot.mtelplot(phase, rvdatcat, rverrcat, telcat, ax, telfmts=self.telfmts)
        if not self.nobin and len(rvdat) > 10:
            ax.errorbar(
                bint, bindat, yerr=binerr, fmt='ro', mec='w', ms=bin_markersize,
                mew=bin_markeredgewidth
            )

        if self.phase_limits:
            ax.set_xlim(self.phase_limits[0], self.phase_limits[1])
        else:
            ax.set_xlim(-0.5, 0.5)

        if not self.yscale_auto:
            scale = np.std(rvdatcat)
            ax.set_ylim(-self.yscale_sigma*scale, self.yscale_sigma*scale)

        keys = [p+str(pnum) for p in ['per', 'k', 'e']]

        labels = [self.post.params.tex_labels().get(k, k) for k in keys]
        if pnum < self.num_planets:
            ticks = ax.yaxis.get_majorticklocs()
            ax.yaxis.set_ticks(ticks[1:-1])

        ax.set_ylabel('RV [{ms:}]'.format(**plot.latex), weight='bold')
        ax.set_xlabel('Phase', weight='bold')

        print_params = ['per', 'k', 'e']
        units = {'per': 'days', 'k': plot.latex['ms'], 'e': ''}

        anotext = []
        for l, p in enumerate(print_params):
            val = self.post.params["%s%d" % (print_params[l], pnum)].value

            if self.uparams is None:
                _anotext = '$\\mathregular{%s}$ = %4.2f %s' % (labels[l].replace("$", ""), val, units[p])
            else:
                if hasattr(self.post, 'medparams'):
                    val = self.post.medparams["%s%d" % (print_params[l], pnum)]
                else:
                    print("WARNING: medparams attribute not found in " +
                          "posterior object will annotate with " +
                          "max-likelihood values and reported uncertainties " +
                          "may not be appropriate.")
                err = self.uparams["%s%d" % (print_params[l], pnum)]
                if err > 1e-15:
                    val, err, errlow = sigfig(val, err)
                    _anotext = '$\\mathregular{%s}$ = %s $\\mathregular{\\pm}$ %s %s' \
                               % (labels[l].replace("$", ""), val, err, units[p])
                else:
                    _anotext = '$\\mathregular{%s}$ = %4.2f %s' % (labels[l].replace("$", ""), val, units[p])

            anotext += [_anotext]

        anotext = '\n'.join(anotext)
        plot.add_anchored(
            anotext, loc=1, frameon=True, prop=dict(size=self.phasetext_size,
                                                    weight='bold'),
            bbox=dict(ec='none', fc='w', alpha=0.8)
        )

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
            # periodogram for this planet doesn't exist, assume it was previously-known
            ax = pl.gca()
            ax.annotate('Pre-defined Planet', xy=(0.5, 0.5), xycoords='axes fraction',
                        horizontalalignment='center', verticalalignment='center',
                        fontsize=self.text_size+8)
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
            lower = np.amin(self.power['bic'])

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
            # Print unitsl and axis label at the bottom.
            ax.set_xlabel('Period [day]', fontweight='bold')
            ax.tick_params(axis='x', which='both', direction='out',
                           bottom='on', top='off', labelbottom='on')

    def plot_window(self, pltletter):
        """Plot the window function of the data.
        """
        ax = pl.gca()

        #Put axis and label on the right side of the plot.
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position('right')

        plot.labelfig(pltletter)

        ax.set_xlabel('Period [day]', fontweight='bold')
        ax.set_ylabel('Window function power', fontweight='bold')
        ax.set_xscale('log')
        ax.set_xlim([np.amin(self.pers), np.amax(self.pers)])
        ax.set_ylim(bottom=0)
        '''
        baseline    = np.amax(self.rvtimes) - np.amin(self.rvtimes)
        window      = utils.window(self.rvtimes, np.flip(1/self.pers))

        min = 3
        max = baseline/2

        window_safe = window[np.where(np.logical_and(
                                      self.pers < max, self.pers > min))]
        pers_safe   = self.pers[np.where(np.logical_and(
                                         self.pers < max, self.pers > min))]
        '''
        # Plot separate window function for each instrument.
        for like in self.like_list:
            inst = like.telvec[0]
            window, f_safe, fap_min, thresh = utils.window(like.x,
                                                   np.flip(1/self.pers))
            ax.plot(1/f_safe, window, alpha=0.5, label=inst)
        ax.legend(loc=0, prop=dict(size=self.phasetext_size, weight='bold'))

        #ax.axvspan(np.amin(self.pers), 1/np.amax(f_safe), alpha=0.25, color='purple')
        #ax.axvspan(1/np.amin(f_safe), np.amax(self.pers), alpha=0.25, color='purple')

    def plot_multipanel(self, nophase=False, letter_labels=True):
        """Provision and plot an RV multipanel plot

        Args:
            nophase (bool, optional): if True, don't
                include phase plots. Default: False.
            letter_labels (bool, optional): if True, include
                letter labels on orbit and residual plots.
                Default: True.

        Returns:
            tuple containing:
                - current matplotlib Figure object
                - list of Axes objects

        """
        if nophase:
            scalefactor = 1
        else:
            scalefactor = self.phase_nrows

        figheight = self.ax_rv_height + self.ax_phase_height * scalefactor

        # provision figure
        fig = pl.figure(figsize=(self.figwidth, figheight))

        fig.subplots_adjust(left=0.12, right=0.95)
        gs_rv = gridspec.GridSpec(2, 1, height_ratios=[1., 0.5])

        divide = 1 - self.ax_rv_height / figheight
        gs_rv.update(left=0.12, right=0.93, top=0.93,
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

        # phase-folded plots
        if not nophase:
            gs_phase = gridspec.GridSpec(self.phase_nrows, self.phase_ncols)

            if self.phase_ncols == 1:
                gs_phase.update(left=0.12, right=0.93,
                                top=divide - self.rv_phase_space * 0.5,
                                bottom=0.07, hspace=0.003)
            else:
                gs_phase.update(left=0.12, right=0.93,
                                top=divide - self.rv_phase_space * 0.5,
                                bottom=0.07, hspace=0.25, wspace=0.25)

            for i in range(self.num_planets):
                i_row = int(i / self.phase_ncols)
                i_col = int(i - i_row * self.phase_ncols)
                ax_phase = pl.subplot(gs_phase[i_row, i_col])
                self.ax_list += [ax_phase]

                pl.sca(ax_phase)
                self.plot_phasefold(pltletter, i+1)
                pltletter += 1

        if self.saveplot is not None:
            pl.savefig(self.saveplot, dpi=150)
            print("RV multi-panel plot saved to %s" % self.saveplot)

        return fig, self.ax_list

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

        fig.subplots_adjust(left=0.12, right=0.95)
        gs_rv = gridspec.GridSpec(2, 1, height_ratios=[1., 0.5])

        divide = 1 - self.ax_rv_height / figheight
        gs_rv.update(left=0.12, right=0.93, top=0.93,
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
            gs_phase.update(left=0.12, right=0.93,
                            top=divide - self.rv_phase_space * 0.2,
                            bottom=0.07, hspace=0.003)
        else:
            gs_phase.update(left=0.12, right=0.93,
                            top=divide - self.rv_phase_space * 0.2,
                            bottom=0.07, hspace=0.003, wspace=0.05)

        for i in range(self.num_planets):
            #Plot phase.
            #i_row = int(i / self.summary_ncols)
            i_row = i
            #i_col = int(i - i_row * self.summary_ncols)
            i_col = 0
            ax_phase = pl.subplot(gs_phase[i_row, i_col])
            self.ax_list += [ax_phase]

            pl.sca(ax_phase)
            self.plot_phasefold(pltletter, i+1)
            pltletter += 1

            #Plot periodogram.
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
