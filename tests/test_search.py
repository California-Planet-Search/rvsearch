"""Regression tests for rvsearch against radvel>=1.6.

radvel>=1.4 evaluates the model from ``post.vector``; rvsearch edits
``post.params`` directly, so a missed ``utils.sync`` makes the search run
without error but return the wrong planets. These tests pin the answer.
"""
import copy
import os

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
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


def _seeded_two_planets(per1=4.6455, vary_per1=True,
                        basis='per tc secosw sesinw k'):
    """Synthetic HD 156668-like data, seeded as jump2 seeds it.

    Planet b is at 4.6426 d over an 8500 d baseline. The seed puts it at
    4.6455 d, one resolution element (P**2/T) off, where one free fit loses
    it: on this data the plain fit runs b off to about 5.02 d.
    """
    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(2452800, 2461300, 400))
    truth = radvel.Parameters(2, basis='per tc e w k')
    for key, val in dict(per1=4.6426, tc1=2455201.7, e1=0.0, w1=0.0, k1=2.4,
                         per2=794.4, tc2=2455890.0, e2=0.13, w2=1.0,
                         k2=2.7).items():
        truth[key] = radvel.Parameter(value=val)
    rv = radvel.RVModel(truth)(t) + rng.normal(0, 3.0, t.size)
    data = pd.DataFrame(dict(time=t, mnvel=rv, errvel=np.full(t.size, 1.5),
                             tel='j'))

    params = radvel.Parameters(2, basis='per tc secosw sesinw k')
    defaults = utils.initialize_default_pars(instnames=['j'], times=t)
    for key in ['gamma_j', 'jit_j', 'dvdt', 'curv']:
        params[key] = defaults[key]
    for key, val in dict(per1=per1, tc1=2455201.6, secosw1=0.0, sesinw1=0.0,
                         k1=1.9, per2=811.3, tc2=2455890.0, secosw2=0.29,
                         sesinw2=-0.06, k2=2.56).items():
        params[key] = radvel.Parameter(value=val)
    params['dvdt'].vary = False
    params['curv'].vary = False
    params['per1'].vary = vary_per1
    if basis != params.basis.name:
        params = params.basis.to_any_basis(params, basis)
    priors = [radvel.prior.EccentricityPrior(2), radvel.prior.PositiveKPrior(2)]
    return utils.initialize_post(data, params=params, priors=priors)


@pytest.fixture(scope='module')
def seeded_fit():
    post = _seeded_two_planets()
    before = {key: post.params[key].value for key in post.params.keys()}
    return post, before, utils.fit_known(post)


def test_fit_known_recovers_an_offset_seed(seeded_fit):
    _, _, fit = seeded_fit
    # A quarter of a resolution element: 4.6426**2 / 8500 / 4 = 0.0006 d.
    assert fit.params['per1'].value == pytest.approx(4.6426, abs=6e-4)
    assert fit.params['k1'].value == pytest.approx(2.4, abs=0.5)
    assert fit.params['per2'].value == pytest.approx(794.4, abs=15)


def test_fit_known_is_never_worse_than_one_fit(seeded_fit):
    post, _, fit = seeded_fit
    plain = utils.maxlike(copy.deepcopy(post), verbose=False)
    assert fit.logprob() >= plain.logprob()


def test_fit_known_leaves_its_input_alone(seeded_fit):
    post, before, fit = seeded_fit
    assert fit is not post
    assert {key: post.params[key].value for key in post.params.keys()} == before
    assert post.params['per1'].vary


def test_fit_known_keeps_a_fixed_period_and_the_vary_flags():
    post = _seeded_two_planets(vary_per1=False)
    fit = utils.fit_known(post)
    assert fit.params['per1'].value == 4.6455
    assert not fit.params['per1'].vary
    assert fit.params['per2'].vary and fit.params['secosw1'].vary


def test_fit_known_handles_a_log_period_basis():
    post = _seeded_two_planets(basis='logper tc secosw sesinw k')
    fit = utils.fit_known(post)
    assert np.exp(fit.params['logper1'].value) == pytest.approx(4.6426, abs=6e-4)
