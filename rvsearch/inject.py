"""Injection and recovery class"""

import os
import numpy as np
import pandas as pd
import pylab as pl
import pickle
import pathos.multiprocessing as mp
import radvel

class Injections(object):
    """
    Class to perform and record injection and recovery tests for a planetary system.

    Args:
        searchpath (string): Path to a saved rvsearch.Search object
        plim (tuple): lower and upper period bounds for injections
        klim (tuple): lower and upper k bounds for injections
        elim (tuple): lower and upper e bounds for injections
        num_sim (int): number of planets to simulate
    """

    def __init__(self, searchpath, plim, klim, elim, num_sim=1):
        self.searchpath = searchpath
        self.plim = plim
        self.klim = klim
        self.elim = elim
        self.num_sim = num_sim

        self.search = pickle.load(open(searchpath, 'rb'))
        seed = np.round(self.search.data['time'].values[0] * 1000).astype(int)

        self.injected_planets = self.random_planets(seed)
        self.recoveries = self.injected_planets

        self.outdir = os.path.dirname(searchpath)

    def random_planets(self, seed):
        """Generate random planets

        Produce a DataFrame with random planet parameters

        Args:
            seed (int): seed for random number generator

        Returns:
            DataFrame: with columns inj_period, inj_tp, inj_e, inj_w, inj_k

        """

        p1, p2 = self.plim
        k1, k2 = self.klim
        e1, e2 = self.elim
        num_sim = self.num_sim

        np.random.seed(seed)

        if p1 == p2:
            sim_p = np.zeros(num_sim) + p1
        else:
            sim_p = 10 ** np.random.uniform(np.log10(p1), np.log10(p2), size=num_sim)

        if k1 == k2:
            sim_k = np.zeros(num_sim) + k1
        else:
            sim_k = 10 ** np.random.uniform(np.log10(k1), np.log10(k2), size=num_sim)

        if e1 == e2:
            sim_e = np.zeros(num_sim) + e1
        else:
            sim_e = np.random.uniform(e1, e2, size=num_sim)

        sim_tp = np.random.uniform(0, sim_p, size=num_sim)
        sim_om = np.random.uniform(0, 2 * np.pi, size=num_sim)

        df = pd.DataFrame(dict(inj_period=sim_p, inj_tp=sim_tp, inj_e=sim_e,
                               inj_w=sim_om, inj_k=sim_k))

        return df

    def run_injections(self, num_cpus=1):
        """Launch injection/recovery tests

        Try to recover all planets defined in self.simulated_planets

        Args:
            num_cpus (int): number of CPUs to utilize. Each injection will run
                on a separate CPU. Individual injections are forced to be single-threaded
        Returns:
            DataFrame: summary of injection/recovery tests

        """

        def _run_one(orbel):
            sfile = open(self.searchpath, 'rb')
            search = pickle.load(sfile)
            sfile.close()

            recovered, recovered_orbel = search.inject_recover(orbel, num_cpus=1)

            bic = search.best_bics[-1]

            return recovered, recovered_orbel, bic

        outcols = ['inj_period', 'inj_tp', 'inj_e', 'inj_w', 'inj_k',
                   'rec_period', 'rec_tp', 'rec_e', 'rec_w', 'rec_k',
                   'recovered', 'bic']
        outdf = pd.DataFrame([], index=range(self.num_sim),
                             columns=outcols)
        outdf[self.injected_planets.columns] = self.injected_planets

        pool = mp.Pool(processes=num_cpus)

        in_orbels = []
        out_orbels = []
        recs = []
        bics = []
        for i, row in self.injected_planets.iterrows():
            in_orbels.append(list(row.values))

        outputs = pool.map(_run_one, in_orbels)

        for out in outputs:
            recovered, recovered_orbel, bic = out
            out_orbels.append(recovered_orbel)
            recs.append(recovered)
            bics.append(bic)

        out_orbels = np.array(out_orbels)
        outdf['rec_period'] = out_orbels[:, 0]
        outdf['rec_tp'] = out_orbels[:, 1]
        outdf['rec_e'] = out_orbels[:, 2]
        outdf['rec_w'] = out_orbels[:, 3]
        outdf['rec_k'] = out_orbels[:, 4]

        outdf['recovered'] = recs
        outdf['bic'] = bics

        self.recoveries = outdf

        return outdf

    def interpolate(self, period, k):
        pass

    def save(self):
        self.recoveries.to_csv(os.path.join('recoveries.csv'), index=False)


class Completeness(object):
    """Calculate completeness surface from a suite of injections

    Args:
        recoveries (DataFrame): DataFrame with injection/recovery tests from Injections.save
    """

    def __init__(self, recoveries, mstar=1.0):
        """Object to handle a suite of injection/recovery tests

        Args:
            recoveries (DataFrame): DataFrame of injection/recovery tests from Injections class
            mstar (float): (optional) stellar mass to use in conversion from p, k to au, msini

        """
        self.recoveries = recoveries

        self.recoveries['inj_msini'] = radvel.utils.Msini(self.recoveries['inj_k'],
                                                          self.recoveries['inj_period'],
                                                          mstar, self.recoveries['inj_e'])
        self.recoveries['rec_msini'] = radvel.utils.Msini(self.recoveries['rec_k'],
                                                          self.recoveries['rec_period'],
                                                          mstar, self.recoveries['rec_e'])

        self.recoveries['inj_au'] = radvel.utils.semi_major_axis(self.recoveries['inj_period'], mstar)
        self.recoveries['rec_au'] = radvel.utils.semi_major_axis(self.recoveries['rec_period'], mstar)

    @classmethod
    def from_csv(cls, recovery_file):
        """Read recoveries and create Completeness object"""
        recoveries = pd.read_csv(recovery_file)
        return cls(recoveries)

    def completeness_grid(self, xcol, ycol, xlim, ylim, resolution=50, xlogwin=0.5, ylogwin=0.5):
        """Calculate completeness on a fine grid

        Compute a 2D moving average in loglog space

        Args:
            xcol (string): x column label from self.recoveries
            ycol (string): y column label from self.recoveries
            xlim (tuple): min and max x limits
            ylim (tuple): min and max y limits
            resolution (int): (optional) grid is sampled at this resolution
            xlogwin (float): (optional) x width of moving average
            ylogwin (float): (optional) y width of moving average

        """
        xgrid = np.logspace(np.log10(xlim[0]),
                            np.log10(xlim[1]),
                            resolution)
        ygrid = np.logspace(np.log10(ylim[0]),
                            np.log10(ylim[1]),
                            resolution)

        xinj = self.recoveries[xcol]
        yinj = self.recoveries[ycol]

        good = self.recoveries['recovered']

        z = np.zeros((len(ygrid), len(xgrid)))
        last = 0
        for i,x in enumerate(xgrid):
            for j,y in enumerate(ygrid):
                xlow = 10**(np.log10(x) - xlogwin/2)
                xhigh = 10**(np.log10(x) + xlogwin/2)
                ylow = 10**(np.log10(y) - ylogwin/2)
                yhigh = 10**(np.log10(y) + ylogwin/2)

                xbox = yinj[np.where((xinj <= xhigh) & (xinj >= xlow))[0]]
                if y > max(xbox) or y < min(xbox):
                    z[j, i] = np.nan
                    continue

                boxall = np.where((xinj <= xhigh) & (xinj >= xlow) &
                                  (yinj <= yhigh) & (yinj >= ylow))[0]
                boxgood = np.where((xinj[good] <= xhigh) &
                                   (xinj[good] >= xlow) & (yinj[good] <= yhigh) &
                                   (yinj[good] >= ylow))[0]
                # print(x, y, xlow, xhigh, ylow, yhigh, len(boxgood), len(boxall))
                if len(boxall) > 10:
                    z[j, i] = float(len(boxgood))/len(boxall)
                    last = float(len(boxgood))/len(boxall)
                else:
                    z[j, i] = np.nan

        return (xgrid, ygrid, z)
