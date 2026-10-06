"""Regression tests for rvsearch against radvel>=1.6.

radvel>=1.4 evaluates the model from ``post.vector``; rvsearch edits
``post.params`` directly, so a missed ``utils.sync`` makes the search run
without error but return the wrong planets. These tests pin the answer.
"""
import os

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pytest
import radvel

from rvsearch import search, utils

DATA = os.path.join(os.path.dirname(__file__), '..', 'example_data', 'HD128311.csv')


def _one_planet_post():
    data = utils.read_from_csv(DATA, binsize=0.5, verbose=False)
    tels = [str(t) for t in np.unique(data['tel'].values)]
    params = utils.initialize_default_pars(instnames=tels, times=data['jd'].values)
    return utils.initialize_post(data, params=params)


def test_sync_pushes_param_edits_into_vector():
    post = _one_planet_post()
    post.params['per1'].value = 123.4
    post.params['k1'].vary = False
    utils.sync(post)
    assert post.vector.vector[post.vector.indices['per1']][0] == pytest.approx(123.4)
    assert 'k1' not in post.name_vary_params()


def test_maxlike_honours_fixed_period():
    post = _one_planet_post()
    post.params['per1'].value = 300.0
    post.params['per1'].vary = False
    post.params['k1'].value = 10.0
    post = utils.maxlike(post, verbose=False)
    assert post.params['per1'].value == pytest.approx(300.0)


def test_search_param_names_survive_radvel_vector():
    # np.unique(tel) yields np.str_; radvel 1.6.5's Vector drops non-str keys,
    # which later breaks MCMC with an IndexError.
    data = utils.read_from_csv(DATA, binsize=0.5, verbose=False)
    s = search.Search(data, starname='HD128311', mcmc=False, verbose=False,
                      save_outputs=False)
    s.add_planet()  # rebuilds params from likelihood.extra_params (np.str_)
    for tel in s.tels:
        assert 'jit_' + tel in s.post.vector.names


@pytest.mark.slow
def test_hd128311_finds_two_planets(tmp_path):
    data = utils.read_from_csv(DATA, binsize=0.5, verbose=False)
    s = search.Search(data, starname='HD128311', min_per=60, workers=2,
                      mcmc=False, verbose=False, mstar=[0.83, 0.05])
    s.run_search(outdir=str(tmp_path / 'out'))

    assert s.num_planets == 2
    synth = s.post.params.basis.to_synth(s.post.params)
    pers = sorted(synth['per{}'.format(n)].value for n in (1, 2))
    assert pers[0] == pytest.approx(453.7, abs=5)
    assert pers[1] == pytest.approx(918.4, abs=10)
    assert (tmp_path / 'out' / 'post_final.pkl').exists()
