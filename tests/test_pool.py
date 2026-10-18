"""rvsearch's worker pools stop when told to, and are gone before it returns.

A program that calls rvsearch may handle SIGTERM itself (the RadVel API
service does, to cancel a job). Forked pool workers inherit that handler, so
``Pool.terminate()``, which stops workers with SIGTERM, cannot stop them if
the handler does not exit. That includes a worker that has been forked but
has not run its initializer yet. And a pool that is closed but never joined
is shut down later by its finalizer, where an exception raised by a signal
handler is printed ("Exception ignored in: <Finalize object, dead>") and
dropped. The inherited handler and the unjoined pool both bit a cancelled
search on 2026-10-17.
"""
import contextlib
import os
import signal
import threading
import time
import types

import multiprocess
import numpy as np
import pandas as pd
import pytest

import rvsearch.utils as utils
from rvsearch import inject, periodogram

DATA = os.path.join(os.path.dirname(__file__), '..', 'example_data', 'HD128311.csv')


def _sigterm_state(_):
    """(SIGTERM has its default action, SIGTERM is not blocked)"""
    return (signal.getsignal(signal.SIGTERM) == signal.SIG_DFL,
            signal.SIGTERM not in signal.pthread_sigmask(signal.SIG_BLOCK, set()))


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
def fork_workers():
    """Start pool workers by fork for the test.

    Only forked workers inherit the host's signal handlers, which is what
    these tests are about. Under spawn (the macOS default) and forkserver
    (the Linux default from Python 3.14) workers start with default
    handlers and the tests would pass with or without the fix.
    """
    if 'fork' not in multiprocess.get_all_start_methods():
        pytest.skip('needs the fork start method, the only one whose workers '
                    'inherit the host signal handlers')
    previous = multiprocess.get_start_method(allow_none=True)
    multiprocess.set_start_method('fork', force=True)
    try:
        yield
    finally:
        multiprocess.set_start_method(previous, force=True)


@pytest.fixture
def host_handles_sigterm(fork_workers):
    """A SIGTERM handler that only notes the signal, like a host's
    graceful-shutdown flag. Forked workers inherit it."""
    seen = []
    old = signal.signal(signal.SIGTERM, lambda signum, frame: seen.append(signum))
    try:
        yield seen
    finally:
        signal.signal(signal.SIGTERM, old)


@pytest.fixture
def host_exits_on_sigterm(fork_workers):
    """A SIGTERM handler that raises SystemExit, like the RadVel API's.
    Forked workers inherit it."""
    def cancel(signum, frame):
        raise SystemExit(128 + signum)

    old = signal.signal(signal.SIGTERM, cancel)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, old)


def _send_sigterm_once_tasks_are_running(paths):
    """Send this process a SIGTERM when every task has written its pid."""
    def cancel():
        _wait_for_pids(paths)
        os.kill(os.getpid(), signal.SIGTERM)

    threading.Thread(target=cancel, daemon=True).start()


def _spy_on_pool(monkeypatch, interrupt=()):
    """Make worker_pool build real pools whose close, join and terminate
    calls are recorded. A method named in ``interrupt`` records itself and
    then raises KeyboardInterrupt, as a signal landing in it would.

    Returns the list of calls and the list of pools.
    """
    calls = []
    pools = []
    real_pool = utils.mp.Pool

    def make_pool(*args, **kwargs):
        pool = real_pool(*args, **kwargs)
        pools.append(pool)
        for name in ('close', 'join', 'terminate'):
            def spy(real=getattr(pool, name), name=name):
                calls.append(name)
                if name in interrupt:
                    raise KeyboardInterrupt
                return real()
            setattr(pool, name, spy)
        return pool

    monkeypatch.setattr(utils, 'mp', types.SimpleNamespace(Pool=make_pool))
    return calls, pools


def _active_worker_pids():
    return [child.pid for child in multiprocess.active_children()]


def test_workers_take_the_default_sigterm_action(deadline, host_handles_sigterm):
    with utils.worker_pool(2) as pool:
        assert pool.map(_sigterm_state, range(4)) == [(True, True)] * 4


def test_workers_are_reaped_when_the_block_ends(deadline):
    with utils.worker_pool(2) as pool:
        pids = set(pool.map(_pid, range(4)))
    _assert_gone(pids)


@pytest.mark.parametrize(
    'error', [RuntimeError('caller gave up'), SystemExit(143), KeyboardInterrupt()],
    ids=['exception', 'systemexit', 'keyboardinterrupt'])
def test_workers_are_stopped_when_the_caller_raises(tmp_path, deadline, host_handles_sigterm,
                                                    error):
    # SystemExit is what a SIGTERM handler that raises (the RadVel API's)
    # delivers in the middle of pool.map(). KeyboardInterrupt is Ctrl-C.
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


def test_a_sigterm_in_the_middle_of_pool_map_stops_the_workers(tmp_path, deadline,
                                                              host_exits_on_sigterm):
    paths = [str(tmp_path / 'worker{}'.format(i)) for i in range(2)]
    _send_sigterm_once_tasks_are_running(paths)
    start = time.monotonic()
    with pytest.raises(SystemExit):
        with utils.worker_pool(2) as pool:
            pool.map(_write_pid_then_sleep, paths)
    assert time.monotonic() - start < 10
    _assert_gone(_wait_for_pids(paths))


_REAL_INITIALIZER = utils._default_sigterm


def _late_initializer():
    """The real initializer, run late: a worker that has been forked but
    is not set up yet. Forked workers see this module as the test patched it."""
    time.sleep(0.5)
    _REAL_INITIALIZER()


def test_a_worker_that_is_not_set_up_yet_still_stops(monkeypatch, deadline,
                                                     host_handles_sigterm):
    # Until its initializer has run, a forked worker has the host's handler,
    # which here swallows SIGTERM. A terminate() sent in that window used to
    # be lost, and the pool then waited for the worker for good.
    monkeypatch.setattr(utils, '_default_sigterm', _late_initializer)
    start = time.monotonic()
    with pytest.raises(RuntimeError):
        with utils.worker_pool(2):
            pids = _active_worker_pids()
            assert len(pids) == 2
            raise RuntimeError('cancelled straight away')
    assert time.monotonic() - start < 10
    _assert_gone(pids)


@pytest.mark.parametrize('blocked', [False, True], ids=['unblocked', 'blocked'])
@pytest.mark.parametrize('fails', [False, True], ids=['clean', 'raises'])
def test_the_callers_sigterm_mask_is_left_as_it_was(deadline, blocked, fails):
    how = signal.SIG_BLOCK if blocked else signal.SIG_UNBLOCK
    before = signal.pthread_sigmask(how, {signal.SIGTERM})
    try:
        try:
            with utils.worker_pool(2):
                if fails:
                    raise RuntimeError('caller gave up')
        except RuntimeError:
            pass
        now = signal.pthread_sigmask(signal.SIG_BLOCK, set())
        assert (signal.SIGTERM in now) == blocked
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, before)


def test_the_callers_sigterm_mask_is_restored_if_the_pool_cannot_start(monkeypatch):
    def no_pool(*args, **kwargs):
        raise OSError('cannot fork')

    monkeypatch.setattr(utils, 'mp', types.SimpleNamespace(Pool=no_pool))
    before = signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGTERM})
    try:
        with pytest.raises(OSError):
            with utils.worker_pool(2):
                pytest.fail('no pool, so no block')
        assert signal.SIGTERM not in signal.pthread_sigmask(signal.SIG_BLOCK, set())
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, before)


def test_workers_are_terminated_if_interrupted_while_the_pool_is_joined(monkeypatch, deadline,
                                                                       fork_workers):
    calls, _ = _spy_on_pool(monkeypatch, interrupt={'join'})
    with pytest.raises(KeyboardInterrupt):
        with utils.worker_pool(2):
            pids = _active_worker_pids()
    assert calls == ['close', 'join', 'terminate']
    _assert_gone(pids)


def test_a_terminate_that_is_interrupted_is_not_followed_by_a_join(monkeypatch, deadline,
                                                                  fork_workers):
    # terminate() joins the workers itself. If a second signal cuts it short,
    # a join() afterwards would wait for workers that were never stopped.
    calls, pools = _spy_on_pool(monkeypatch, interrupt={'terminate'})
    try:
        with pytest.raises(KeyboardInterrupt):
            with utils.worker_pool(2):
                raise RuntimeError('caller gave up')
        assert calls == ['terminate']
    finally:
        for pool in pools:
            type(pool).terminate(pool)


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
