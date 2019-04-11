.. |br| raw:: html

   <br />

RVsearch: |br| For All Your Planet-Hunting Needs
================================================

Blind RV planet search pipeline based on `RadVel <http://radvel.readthedocs.org>`_.

- Use RadVel setup files to load:
 + parameters for "known" planets
 + data and instruments
 + fix/vary within search
 + fitting (search) basis


Installation
++++++++++++

Install ``rvsearch`` directly from the
`GitHub repository <https://github.com/California-Planet-Search/rvsearch>`_
using pip:

.. code-block:: bash

    $ pip install git+https://github.com/California-Planet-Search/rvsearch

Please report any bugs or feature requests to the
`GitHub issue tracker <https://github.com/California-Planet-Search/rvsearch>`_.

Check out the features available in the command-line-interface:

.. code-block:: bash

    $ rvsearch --help
    usage: rvsearch [-h] [--version] {find,inject,plot} ...

    RadVel-Search: Automated planet detection pipeline

    optional arguments:
      -h, --help          show this help message and exit
      --version           Print version number and exit.

    subcommands:
      {find,inject,plot}


Contents:

.. toctree::
   :maxdepth: 3

   tutorial_api
   api


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
