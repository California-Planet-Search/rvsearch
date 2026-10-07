"""RVsearch HTTP service: the RadVel HTTP API plus search jobs.

Requires the ``[api]`` extra::

    pip install 'rvsearch[api] @ git+https://github.com/California-Planet-Search/rvsearch'
    rvsearch serve --host 0.0.0.0 --port 8000

The app is RadVel's (``radvel.api.main.create_app``) with two more job
kinds mounted on the same runs, job registry and file endpoints:

* ``POST /runs/{run_id}/search``      — planet search on the run's RVs
* ``POST /runs/{run_id}/injections``  — injection-recovery on a finished search
* ``GET  /runs/{run_id}/search``      — the search result JSON

A client creates a run with RadVel's ``POST /runs`` (same payload as a
RadVel fit), starts a job, polls ``GET /jobs/{job_id}``, then reads
``GET /runs/{run_id}/search`` and downloads plots from
``GET /runs/{run_id}/files``. Everything RadVel serves keeps working, so
this image can replace a plain ``radvel-api`` container.
"""

from __future__ import annotations

import copy
import json
import os
import pickle
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

import radvel
from radvel.api import extensions, schemas
from radvel.api.config import get_settings
from radvel.api.main import create_app

import rvsearch

SEARCH_KIND = 'search'
INJECTIONS_KIND = 'injections'
RESULT_FILE = 'search_result.json'
SEARCH_PICKLE = 'search.pkl'
RECOVERIES_FILE = 'recoveries.csv'

extensions.register_job_kind(SEARCH_KIND)
extensions.register_job_kind(INJECTIONS_KIND)


class SearchRequest(BaseModel):
    """Parameters for one planet search (mirror ``rvsearch find``)."""

    min_per: float = Field(3.0, gt=0, description="Shortest period searched [days]")
    max_per: float = Field(10000.0, gt=0, description="Longest period searched [days]")
    max_planets: int = Field(8, ge=1, le=20)
    known: bool = Field(False, description="Start from the planets in the run's setup "
                                           "instead of a blind search")
    trend: bool = Field(False, description="Allow a linear trend (dvdt) from the start")
    mcmc: bool = Field(False, description="Run MCMC on the final model (slow)")
    mstar: Optional[Tuple[float, float]] = Field(
        None, description="(mass, error) in solar masses; used for Msini/a")
    workers: int = Field(4, ge=1, le=64, description="Processes for the periodogram")


class InjectionsRequest(BaseModel):
    """Parameters for injection-recovery (mirror ``rvsearch inject``)."""

    num_inject: int = Field(3000, ge=1, le=100000)
    min_per: float = Field(3.1, gt=0)
    max_per: float = Field(1e6, gt=0)
    min_k: float = Field(0.1, gt=0)
    max_k: float = Field(1000.0, gt=0)
    min_e: float = Field(0.0, ge=0, lt=1)
    max_e: float = Field(0.9, ge=0, lt=1)
    beta_e: bool = False
    full_grid: bool = False
    workers: int = Field(4, ge=1, le=64)


def _run_registry() -> extensions.RunRegistry:
    return extensions.RunRegistry(settings=get_settings())


def _resolve_run(run_id: str, registry: extensions.RunRegistry):
    if not extensions.is_valid_run_id(run_id):
        raise HTTPException(status_code=404, detail="unknown run_id")
    try:
        return registry.get(run_id)
    except extensions.RunNotFound:
        raise HTTPException(status_code=404, detail="unknown run_id")


def _submit(request: Request, run_id: str, kind: str, params: dict, worker):
    runner = getattr(request.app.state, 'job_runner', None)
    if runner is None:
        raise HTTPException(status_code=503, detail="job runner not initialised")
    try:
        row = runner.submit(run_id, kind, params, worker=worker)
    except extensions.JobActiveError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return schemas.JobKickoffResponse(
        job_id=row.job_id, run_id=row.run_id, kind=row.kind, state=row.state)


router = APIRouter(tags=["rvsearch"])


@router.post("/runs/{run_id}/search", response_model=schemas.JobKickoffResponse,
             status_code=status.HTTP_202_ACCEPTED)
def start_search(run_id: str, body: SearchRequest, request: Request,
                 registry: extensions.RunRegistry = Depends(_run_registry)):
    """Start a planet search on the run's RVs. Poll ``GET /jobs/{job_id}``."""
    _resolve_run(run_id, registry)
    if body.min_per >= body.max_per:
        raise HTTPException(status_code=422, detail="min_per must be < max_per")
    return _submit(request, run_id, SEARCH_KIND, body.model_dump(mode='json'),
                   _search_worker)


@router.post("/runs/{run_id}/injections", response_model=schemas.JobKickoffResponse,
             status_code=status.HTTP_202_ACCEPTED)
def start_injections(run_id: str, body: InjectionsRequest, request: Request,
                     registry: extensions.RunRegistry = Depends(_run_registry)):
    """Start injection-recovery tests. Requires a finished search on the run."""
    record = _resolve_run(run_id, registry)
    if not (record.outputdir / SEARCH_PICKLE).is_file():
        raise HTTPException(status_code=409,
                            detail="run has no finished search; POST /runs/{id}/search first")
    if body.min_per >= body.max_per or body.min_k >= body.max_k or body.min_e > body.max_e:
        raise HTTPException(status_code=422, detail="each min must be below its max")
    return _submit(request, run_id, INJECTIONS_KIND, body.model_dump(mode='json'),
                   _injections_worker)


@router.get("/runs/{run_id}/search")
def get_search_result(run_id: str,
                      registry: extensions.RunRegistry = Depends(_run_registry)) -> Dict[str, Any]:
    """The latest search result for the run (404 until a search succeeds)."""
    record = _resolve_run(run_id, registry)
    path = record.outputdir / RESULT_FILE
    if not path.is_file():
        raise HTTPException(status_code=404, detail="no search result for this run")
    with open(path) as f:
        return json.load(f)


# ---- workers (run in a child process; must stay module-level) -------------


def _search_worker(run_id: str, params_json: str) -> Dict[str, Any]:
    record = extensions.worker_setup(run_id)
    params = json.loads(params_json)
    progress = extensions.ProgressWriter(extensions.progress_path(record, SEARCH_KIND))
    progress.write({'stage': 'starting', 'pcomplete': 0.0})
    outdir = str(record.outputdir)
    started = time.time()

    with extensions.capture_output(record, step='search'):
        P, post = radvel.utils.initialize_posterior(str(record.setup_py))
        data = P.data
        mstar = params.get('mstar')
        if mstar is None:
            stellar = getattr(P, 'stellar', None) or {}
            if 'mstar' in stellar:
                mstar = (stellar['mstar'], stellar.get('mstar_err', 0.0))

        known_post = None
        if params['known'] and P.nplanets > 0:
            known_post = rvsearch.utils.maxlike(post, verbose=False)

        progress.write({'stage': 'searching', 'pcomplete': 5.0})
        searcher = rvsearch.search.Search(
            data, post=known_post, starname=record.starname,
            min_per=params['min_per'], max_per=params['max_per'],
            max_planets=params['max_planets'], trend=params['trend'],
            mcmc=params['mcmc'], mstar=mstar, workers=params['workers'],
            verbose=False)
        searcher.run_search(outdir=outdir, mkoutdir=False)

        progress.write({'stage': 'plotting', 'pcomplete': 90.0,
                        'planets_found': int(searcher.num_planets)})
        summary_png = '{}_summary.png'.format(record.starname)
        rvsearch.plots.PeriodModelPlot(
            searcher, saveplot=os.path.join(outdir, summary_png)).plot_summary()

    result = summarize(searcher)
    result['params'] = params
    result['plots'] = {'summary': summary_png}
    result['elapsed_s'] = round(time.time() - started, 1)
    _write_json(record.outputdir / RESULT_FILE, result)
    progress.write({'stage': 'done', 'pcomplete': 100.0,
                    'planets_found': result['num_planets']})
    return {'num_planets': result['num_planets']}


def _injections_worker(run_id: str, params_json: str) -> Dict[str, Any]:
    record = extensions.worker_setup(run_id)
    params = json.loads(params_json)
    progress = extensions.ProgressWriter(extensions.progress_path(record, INJECTIONS_KIND))
    progress.write({'stage': 'injecting', 'pcomplete': 0.0,
                    'num_inject': params['num_inject']})
    outdir = str(record.outputdir)
    search_path = os.path.join(outdir, SEARCH_PICKLE)

    # Injections.save() and completeness_plot() write into the cwd.
    with extensions.capture_output(record, step='injections'), \
            radvel.utils.working_directory(outdir):
        inj = rvsearch.inject.Injections(
            search_path, (params['min_per'], params['max_per']),
            (params['min_k'], params['max_k']), (params['min_e'], params['max_e']),
            num_sim=params['num_inject'], full_grid=params['full_grid'],
            verbose=False, beta_e=params['beta_e'])
        recoveries = inj.run_injections(num_cpus=params['workers'])
        inj.save()

        progress.write({'stage': 'plotting', 'pcomplete': 95.0})
        with open(search_path, 'rb') as f:
            searcher = pickle.load(f)
        if searcher.mstar is not None:
            axes = dict(xcol='inj_au', ycol='inj_msini', xlabel='$a$ [AU]',
                        ylabel=r'M$\sin{i_p}$ [M$_\oplus$]')
        else:  # no stellar mass: stay in observable units
            axes = dict(xcol='inj_period', ycol='inj_k', xlabel='Period [days]',
                        ylabel='K [m/s]')
        comp = rvsearch.inject.Completeness.from_csv(
            RECOVERIES_FILE, xcol=axes['xcol'], ycol=axes['ycol'], mstar=searcher.mstar)
        fig = rvsearch.plots.CompletenessPlots(comp, searches=[searcher]).completeness_plot(
            title=record.starname, xlabel=axes['xlabel'], ylabel=axes['ylabel'])
        recoveries_png = '{}_recoveries.png'.format(record.starname)
        fig.savefig(recoveries_png, dpi=150)

    num_recovered = int(recoveries['recovered'].astype(bool).sum())
    result_path = record.outputdir / RESULT_FILE
    if result_path.is_file():
        with open(result_path) as f:
            result = json.load(f)
        result['injections'] = {'params': params, 'num_inject': len(recoveries),
                                'num_recovered': num_recovered,
                                'recoveries_csv': RECOVERIES_FILE}
        result.setdefault('plots', {})['recoveries'] = recoveries_png
        _write_json(result_path, result)
    progress.write({'stage': 'done', 'pcomplete': 100.0})
    return {'num_inject': len(recoveries), 'num_recovered': num_recovered}


# ---- helpers ----------------------------------------------------------------


def summarize(searcher) -> Dict[str, Any]:
    """JSON-safe summary of a finished :class:`rvsearch.search.Search`."""
    post = searcher.post
    synth = post.params.basis.to_synth(copy.deepcopy(post.params))
    uparams = getattr(post, 'uparams', None) or {}

    def _val(params, key):
        return float(params[key].value) if key in params else None

    planets: List[Dict[str, Any]] = []
    for n in range(1, int(searcher.num_planets) + 1):
        planet: Dict[str, Any] = {'index': n}
        for name in ('per', 'tc', 'tp', 'e', 'w', 'k'):
            key = '{}{}'.format(name, n)
            planet[name] = _val(synth, key)
            if key in uparams:
                planet[name + '_err'] = float(uparams[key])
        planet['bic'] = _float_or_none(searcher.best_bics.get(n - 1))
        planet['bic_threshold'] = _float_or_none(searcher.bic_threshes.get(n - 1))
        planets.append(planet)

    instruments = {}
    for tel in searcher.tels:
        instruments[tel] = {'gamma': _val(post.params, 'gamma_' + tel),
                            'jit': _val(post.params, 'jit_' + tel)}

    return {
        'starname': searcher.starname,
        'num_planets': int(searcher.num_planets),
        'planets': planets,
        'dvdt': _val(post.params, 'dvdt'),
        'curv': _val(post.params, 'curv'),
        'instruments': instruments,
        'mcmc': bool(searcher.mcmc),
        'mstar': ([float(searcher.mstar), float(searcher.mstar_err)]
                  if searcher.mstar is not None else None),
        'nobs': int(len(post.likelihood.x)),
        'rvsearch_version': rvsearch.__version__,
        'radvel_version': radvel.__version__,
    }


def _float_or_none(value):
    return None if value is None else float(value)


def _write_json(path, payload) -> None:
    tmp = str(path) + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def build_app():
    """RadVel's app plus the search routes."""
    app = create_app()
    app.title = 'RVsearch + RadVel HTTP API'
    app.version = '{} (radvel {})'.format(rvsearch.__version__, radvel.__version__)
    app.include_router(router)
    return app


app = build_app()
