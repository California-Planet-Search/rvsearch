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
