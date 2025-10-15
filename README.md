# Normalizing flows as a generative model for lattice field theory

[![GitHub Super-Linter](https://github.com/interTwin-eu/itwinai-plugin-template/actions/workflows/lint.yml/badge.svg)](https://github.com/marketplace/actions/super-linter)
[![GitHub Super-Linter](https://github.com/interTwin-eu/itwinai-plugin-template/actions/workflows/check-links.yml/badge.svg)](https://github.com/marketplace/actions/markdown-link-check)
 [![SQAaaS source code](https://github.com/EOSC-synergy/itwinai-plugin-template.assess.sqaaas/raw/main/.badge/status_shields.svg)](https://sqaaas.eosc-synergy.eu/#/full-assessment/report/https://raw.githubusercontent.com/eosc-synergy/itwinai-plugin-template.assess.sqaaas/main/.report/assessment_output.json)

The normflow plugin is developed as part of the
[itwinai](https://github.com/interTwin-eu/itwinai) repository,
for integration of the Lattice QCD use-case in the interTwin project.
This plugin allows the `normflow` developers to continue their developments
independently in this repository, which is already integrated with itwinai.
For further details about the normflow package, you are referred to the
repository of the use-case [here](https://github.com/jkomijani/normflow_).

Integration Author: Rakesh Sarma, Juelich

The dependencies in this project are managed with the `pyproject.toml` file,
which requires installation of `itwinai` and the `normflow` source files,
provided in this repository. To get started, clone the repository and follow
the steps below. In this project, it is recommened to use the `uv` package
manager, which allows much faster installation of Python packages compared
to `pip`, and also manages library overlaps and dependencies across different
packages.

- Move to the root of the directory and create a Python virtual environment
with `python -m venv .venv`.
- Activate the virtual environment `source .venv/bin/activate`.
- Install `uv` package manager with `pip install uv`.
- Then install the plugin with `uv pip install .`. This will install both
`itwinai` and `normflow` repisotory. For new developments in the `normflow`
package, the source files located under `src/itwinai/plugins/normflow` have
to be modified.
To install in an HPC environment, please follow the instructions on the required
modules in the `itwinai` documentation
[here](https://itwinai.readthedocs.io/latest/installation/developer_installation.html).
Once the modules are loaded, rest of the installation
procedure is same as above.

These steps will setup the environment for running this plugin. For launching
a pipeline or training, further instructions can be found in the `README.md`
under `src/itwinai/plugins/normflow`.

If you want to use the remote interTwin Mlflow server, remember to run the
`mlflow-setup.sh` script.
