# radvel-search
RV Planet Search Pipeline Based on RadVel

[![Powered by RadVel](https://img.shields.io/badge/powered_by-RadVel-EB5368.svg?style=flat)](https://radvel.readthedocs.io)

Use RadVel setup files to load:
- parameters for "known" planets
- data and instruments
- fix/vary within search (not implemented)
- fitting (search) basis (not implemented)

See the [documentation](https://california-planet-search.github.io/rvsearch/) for installation instructions. Installing into a fresh anaconda environment is highly recommended.

Example calling syntax:

`rvsearch find -s path-to-setup`

`rvsearch plot -t summary -d path-to-outputdir`

`rvsearch inject -d path-to-outputdir`

`rvsearch plot -t recovery -d path-to-outputdir`


See `rvsearch --help` or `rvsearch plot --help` to see all available options.

## HTTP service

`rvsearch serve` runs RadVel's HTTP API (every `radvel serve` endpoint) plus
search jobs, so one container serves both:

```
pip install 'rvsearch[api]'          # needs radvel[api] >= 1.6.6
rvsearch serve --host 0.0.0.0 --port 8000
# or: docker compose up --build
```

1. `POST /runs` with a RadVel setup payload (data, instruments, optional
   `stellar` mass). This is the same call used for a RadVel fit.
2. `POST /runs/{run_id}/search`, for example `{"min_per": 3, "max_planets": 8, "mcmc": false}`.
   Set `"known": true` to start from the setup's planets.
3. Poll `GET /jobs/{job_id}` until `state` is `succeeded`. `progress.stage`
   and `progress.planets_found` update along the way.
4. `GET /runs/{run_id}/search` returns the detected planets (per, tc, tp,
   e, w, k, BIC vs. threshold), trend, and per-instrument gamma/jitter.
   `GET /runs/{run_id}/files/<starname>_summary.png` returns the
   summary plot.
5. Optional: `POST /runs/{run_id}/injections` (`{"num_inject": 3000}`) runs
   injection-recovery on the finished search and adds the recoveries CSV
   and completeness plot.

Interactive docs are served at `/docs`.
