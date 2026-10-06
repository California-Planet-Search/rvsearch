"""HTTP service tests: RadVel's API plus the search/injections job kinds."""
import os
import time

import pandas as pd
import pytest

pytest.importorskip('fastapi')
pytest.importorskip('radvel.api.extensions')

DATA = os.path.join(os.path.dirname(__file__), '..', 'example_data', 'HD128311.csv')


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv('RADVEL_API_RUNS_DIR', str(tmp_path / 'runs'))
    monkeypatch.setenv('RADVEL_API_DB_PATH', str(tmp_path / 'jobs.db'))
    from radvel.api.config import get_settings
    get_settings.cache_clear()

    from fastapi.testclient import TestClient
    from rvsearch.api import build_app
    with TestClient(build_app()) as test_client:
        yield test_client
    get_settings.cache_clear()


def _payload(stellar=True):
    raw = pd.read_csv(DATA)
    # The example CSV stores telescope names as repr'd bytes ("b'j'").
    tels = raw['tel'].str.replace(r"^b'(.*)'$", r'\1', regex=True)
    rows = [{'time': float(t), 'mnvel': float(v), 'errvel': float(e), 'tel': tel}
            for t, v, e, tel in zip(raw.jd, raw.mnvel, raw.errvel, tels)]
    params = {'per1': {'value': 100.0}, 'tc1': {'value': 2455000.0},
              'secosw1': {'value': 0.0}, 'sesinw1': {'value': 0.0},
              'k1': {'value': 10.0}, 'dvdt': {'value': 0.0}, 'curv': {'value': 0.0}}
    for tel in sorted(set(tels)):
        params['gamma_' + tel] = {'value': 0.0, 'linear': True, 'vary': False}
        params['jit_' + tel] = {'value': 2.0}
    payload = {'starname': 'HD128311', 'nplanets': 1, 'instnames': sorted(set(tels)),
               'fitting_basis': 'per tc secosw sesinw k', 'params': params,
               'data': {'kind': 'inline', 'rows': rows}}
    if stellar:
        payload['stellar'] = {'mstar': 0.83, 'mstar_err': 0.05}
    return payload


def _wait(client, job_id, timeout=600):
    deadline = time.time() + timeout
    job = {'state': 'unpolled'}
    while time.time() < deadline:
        job = client.get('/jobs/{}'.format(job_id)).json()
        if job['state'] in ('succeeded', 'failed', 'cancelled'):
            return job
        time.sleep(0.5)
    raise AssertionError('job {} still {} after {}s'.format(job_id, job['state'], timeout))


def test_radvel_endpoints_still_served(client):
    assert client.get('/healthz').status_code == 200
    assert client.post('/runs', json=_payload()).status_code == 201


def test_unknown_run_is_404(client):
    assert client.post('/runs/run-AAAAAAAAAA/search', json={}).status_code == 404
    assert client.get('/runs/run-AAAAAAAAAA/search').status_code == 404


def test_injections_need_a_finished_search(client):
    run_id = client.post('/runs', json=_payload()).json()['run_id']
    resp = client.post('/runs/{}/injections'.format(run_id), json={'num_inject': 2})
    assert resp.status_code == 409


def test_bad_period_range_is_rejected(client):
    run_id = client.post('/runs', json=_payload()).json()['run_id']
    resp = client.post('/runs/{}/search'.format(run_id), json={'min_per': 100, 'max_per': 50})
    assert resp.status_code == 422


@pytest.mark.slow
def test_search_then_injections_end_to_end(client):
    run_id = client.post('/runs', json=_payload()).json()['run_id']

    kick = client.post('/runs/{}/search'.format(run_id),
                       json={'min_per': 60, 'workers': 2})
    assert kick.status_code == 202, kick.text
    assert kick.json()['kind'] == 'search'
    job = _wait(client, kick.json()['job_id'])
    assert job['state'] == 'succeeded', job['error']
    assert job['progress']['stage'] == 'done'
    assert job['progress']['planets_found'] == 2

    result = client.get('/runs/{}/search'.format(run_id)).json()
    assert result['num_planets'] == 2
    pers = sorted(p['per'] for p in result['planets'])
    assert pers[0] == pytest.approx(453.7, abs=5)
    assert pers[1] == pytest.approx(918.4, abs=10)
    assert all(p['bic'] > p['bic_threshold'] for p in result['planets'])
    assert result['mstar'] == [0.83, 0.05]
    assert set(result['instruments']) == {'j', 'k'}

    files = {f['name'] for f in client.get('/runs/{}/files'.format(run_id)).json()}
    assert {'HD128311_summary.png', 'search.pkl', 'search_result.json'} <= files
    png = client.get('/runs/{}/files/HD128311_summary.png'.format(run_id))
    assert png.headers['content-type'] == 'image/png'

    kick = client.post('/runs/{}/injections'.format(run_id),
                       json={'num_inject': 4, 'workers': 2})
    assert kick.status_code == 202, kick.text
    job = _wait(client, kick.json()['job_id'])
    assert job['state'] == 'succeeded', job['error']

    result = client.get('/runs/{}/search'.format(run_id)).json()
    assert result['injections']['num_inject'] == 4
    assert result['plots']['recoveries'] == 'HD128311_recoveries.png'
    files = {f['name'] for f in client.get('/runs/{}/files'.format(run_id)).json()}
    assert {'recoveries.csv', 'HD128311_recoveries.png'} <= files
