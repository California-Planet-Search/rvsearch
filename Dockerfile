# syntax=docker/dockerfile:1
# RVsearch HTTP service = the RadVel API image + rvsearch.
#
# Every RadVel endpoint (runs, fit, mcmc, plots, files, /docs) is unchanged,
# plus /runs/{id}/search and /runs/{id}/injections, so this image is a
# drop-in replacement for a radvel-api container.
#
# Build:
#   docker build -t rvsearch-api:dev .
#   docker build --build-arg RADVEL_IMAGE=radvel-api:dev -t rvsearch-api:dev .  # local radvel
#
# Run:
#   docker run --rm -p 8000:8000 -v $PWD/.runs:/data rvsearch-api:dev

ARG RADVEL_IMAGE=ghcr.io/california-planet-search/radvel-api:1.6.6
FROM ${RADVEL_IMAGE}

USER root
COPY . /src/rvsearch
# radvel[api] is already in the base image; install only rvsearch's own deps.
RUN pip install /src/rvsearch && rm -rf /src/rvsearch

USER radvel
# Inherits WORKDIR /data, the RADVEL_API_* env, tini and the /healthz check.
CMD ["uvicorn", "rvsearch.api:app", "--host", "0.0.0.0", "--port", "8000"]
