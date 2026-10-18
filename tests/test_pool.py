"""rvsearch's worker pools stop when told to, and are gone before it returns.

A program that calls rvsearch may handle SIGTERM itself (the RadVel API
service does, to cancel a job). Forked pool workers inherit that handler, so
``Pool.terminate()``, which stops workers with SIGTERM, cannot stop them if
the handler does not exit. And a pool that is closed but never joined is
shut down later by its finalizer, where an exception raised by a signal
handler is printed ("Exception ignored in: <Finalize object, dead>") and
dropped. Both happened to a cancelled search on 2026-10-17.
"""
import contextlib
import os
import signal
import time

import multiprocess
import numpy as np
import pandas as pd
import pytest

import rvsearch.utils as utils
from rvsearch import inject, periodogram

DATA = os.path.join(os.path.dirname(__file__), '..', 'example_data', 'HD128311.csv')


def _sigterm_is_default(_):
    return signal.getsignal(signal.SIGTERM) == signal.SIG_DFL


def _pid(_):
    return os.getpid()


def _write_pid_then_sleep(path):
    with open(path, 'w') as f:
        f.write(str(os.getpid()))
    time.sleep(20)


def _wait_for_pids(paths, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            return [int(open(p).read()) for p in paths]
        except (FileNotFoundError, ValueError):
            time.sleep(0.05)
    raise AssertionError('the pool workers never started')


def _assert_gone(pids):
    # ProcessLookupError means exited and reaped; a zombie would still answer.
    for pid in pids:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


@pytest.fixture
def deadline():
    """Fail, rather than hang, if the pool cannot stop its workers.

    Workers that ignore SIGTERM can block ``Pool.terminate()`` for good,
    and then pytest itself at exit. After 30 s this kills them and fails
    the test.
    """
    def give_up(signum, frame):
        for child in multiprocess.active_children():
            os.kill(child.pid, signal.SIGKILL)
        raise AssertionError('the pool did not stop its workers within 30 s')

    old = signal.signal(signal.SIGALRM, give_up)
    signal.alarm(30)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


@pytest.fixture
def host_handles_sigterm():
    """A SIGTERM handler that only notes the signal, like a host's
    graceful-shutdown flag. Forked workers inherit it."""
    seen = []
    old = signal.signal(signal.SIGTERM, lambda signum, frame: seen.append(signum))
    try:
        yield seen
    finally:
        signal.signal(signal.SIGTERM, old)


def test_workers_take_the_default_sigterm_action(deadline, host_handles_sigterm):
    with utils.worker_pool(2) as pool:
        assert pool.map(_sigterm_is_default, range(4)) == [True] * 4


def test_workers_are_reaped_when_the_block_ends(deadline):
    with utils.worker_pool(2) as pool:
        pids = set(pool.map(_pid, range(4)))
    _assert_gone(pids)


@pytest.mark.parametrize('error', [RuntimeError('caller gave up'), SystemExit(143)],
                         ids=['exception', 'systemexit'])
def test_workers_are_stopped_when_the_caller_raises(tmp_path, deadline, host_handles_sigterm,
                                                    error):
    # SystemExit is what a SIGTERM handler that raises (the RadVel API's)
    # delivers in the middle of pool.map().
    paths = [str(tmp_path / 'worker{}'.format(i)) for i in range(2)]
    start = time.monotonic()
    with pytest.raises(type(error)):
        with utils.worker_pool(2) as pool:
            pool.map_async(_write_pid_then_sleep, paths)
            pids = _wait_for_pids(paths)
            raise error
    # The tasks sleep 20 s. Stopping them must not wait that out.
    assert time.monotonic() - start < 10
    _assert_gone(pids)


class _Pool:
    """Stands in for a worker pool. Runs tasks in this process unless
    given canned results."""

    def __init__(self, results=None):
        self.results = results

    def map(self, func, iterable):
        items = list(iterable)
        if self.results is None:
            return [func(x) for x in items]
        return self.results(items)


def _record_pools(monkeypatch, pool):
    calls = []

    @contextlib.contextmanager
    def fake_worker_pool(processes):
        calls.append(processes)
        yield pool

    monkeypatch.setattr(utils, 'worker_pool', fake_worker_pool)
    return calls


def _one_planet_post():
    data = utils.read_from_csv(DATA, binsize=0.5, verbose=False)
    tels = [str(t) for t in np.unique(data['tel'].values)]
    params = utils.initialize_default_pars(instnames=tels, times=data['jd'].values)
    return utils.initialize_post(data, params=params)


def test_periodogram_uses_worker_pool(monkeypatch):
    calls = _record_pools(monkeypatch, _Pool())
    per = periodogram.Periodogram(_one_planet_post(), manual_grid=[50., 100., 200., 400.],
                                  workers=2, verbose=False)
    per.per_bic()
    assert calls == [2]
    assert len(per.bic) == 4


def test_injections_use_worker_pool(monkeypatch):
    # Recover every planet exactly, without running a search.
    calls = _record_pools(monkeypatch, _Pool(
        results=lambda orbels: [(True, orbel, 30.0, 10.0) for orbel in orbels]))
    inj = inject.Injections.__new__(inject.Injections)
    inj.searchpath = 'unused'
    inj.num_sim = 2
    inj.full_grid = False
    inj.verbose = False
    inj.injected_planets = pd.DataFrame(dict(
        inj_period=[10., 20.], inj_tp=[1., 2.], inj_e=[0., 0.1],
        inj_w=[0., 1.], inj_k=[5., 6.]))
    out = inj.run_injections(num_cpus=3)
    assert calls == [3]
    assert list(out['rec_period']) == [10., 20.]
    assert list(out['recovered']) == [True, True]
