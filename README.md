# acropora

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/stephan-911/acropora/main/docs/assets/acropora-logo-dark.svg">
    <img src="https://raw.githubusercontent.com/stephan-911/acropora/main/docs/assets/acropora-logo-light.svg"
         alt="acropora: computing statistical solutions of nonlinear partial differential equations"
         width="640">
  </picture>
</p>

acropora evolves probability measures by pushing forward individual states.
It approximates statistical solutions of nonlinear PDEs by propagating Monte
Carlo ensembles through deterministic solvers, on CPU (NumPy) or GPU (CuPy)
from a single code path. Its first scheme is a vectorial lattice Boltzmann
method for the compressible Euler equations used in the reference application
to compute statistical solutions of a Mach 2000 astrophysical jet.

acropora is the companion code to:
S. Simonis and G. Wissocq,
Computing statistical solutions of a Mach 2000 astrophysical jet,
arXiv:2605.25282 (2026).
url: [https://arxiv.org/abs/2605.25282](https://arxiv.org/abs/2605.25282)
doi: [10.48550/arXiv.2605.25282](https://doi.org/10.48550/arXiv.2605.25282)

The data of the paper are archived on Zenodo:
doi: [10.5281/zenodo.23415318](https://doi.org/10.5281/zenodo.23415318)

## Install

Requires Python 3.9 or newer.

    git clone https://github.com/stephan-911/acropora.git
    cd acropora
    pip install -e .              # CPU only
    pip install -e ".[gpu]"       # + CuPy (CUDA 12)
    pip install -e ".[post,dev]"  # post-processing / tests
    python -m pytest tests -q     # 33 passed

On a cluster, load a recent Python first (`module load ...`; for the GPU extra
also the CUDA 12 module) and place the virtual environment on a work
filesystem:

    python3 -m venv $WORK/acropora-venv
    $WORK/acropora-venv/bin/pip install --upgrade pip
    $WORK/acropora-venv/bin/pip install -e "/path/to/acropora[post,dev]"

An editable install records the repository path, so the environment keeps
tracking the checkout wherever it sits. The batch jobs take the environment
from `VENV` in the site file (see *Batch jobs*).

## Quick start (CPU)

```python
import numpy as np
from acropora import (Euler2D, VectorialD2Q5, Mach2000JetBC, run_sample,
                     grid_from_nx)

Nx, Ny, dx = grid_from_nx(250)          # 250 x 200 cells, dx = 0.01
bc = Mach2000JetBC.make("ghost", Ny, seed=42)          # portable RNG
solver = VectorialD2Q5(Euler2D(gamma=5/3), Nx, Ny, dx,
                       xp=np, boundary=bc)
bc.initial_condition(solver)
run_sample(solver, np.linspace(0, 0.0035, 8),
           lambda ck, rho: print(ck, rho.max()))
```

Switch to GPU by `xp = acropora.get_xp("gpu")`
(or `--backend gpu` in the example CLI);
nothing else changes.

## Layout

    src/acropora/
      backend.py          array-namespace dispatch (numpy/cupy), device mgmt
      grid.py             grid geometry from the x-resolution Nx
      random_fields.py    stochastic inlet perturbation + RNG modes
      models/             PDE systems: fluxes + wave-speed bound (Euler2D; add
                          your system here)
      schemes/            time-stepping schemes (vectorial D2Q5 with
                          positivity limiter; add schemes here)
      limiters.py         reusable limiter components
      boundaries.py       boundary/IC objects (Mach2000JetBC; add setups here)
      runner.py           single-sample time loop + checkpointing
      ensemble.py         Monte Carlo orchestration (node partitioning,
                          per-sample restart, persistent worker pool)
      io/                 memmapped ensemble store
      post/               post-processing: grid restriction, one-/
                          two-point Wasserstein distances, strong L1 error,
                          mean/std errors, rate fits, part merging,
                          centerline bands and one-point densities,
                          PDF-heatmap rasterization with pgfplots output,
                          quantile-band diagnostics
    examples/mach2000_jet/  CLI + SLURM script for reproducibility
    baseline/             reference implementation behind a numpy shim
                          (regression baseline)
    tests/                bit-exact regression vs baseline + physics tests
    benchmarks/           CPU benchmark + cluster GPU benchmark/repro script

The split: a *model* provides fluxes and a wave-speed bound, a *scheme*
integrates any model, and a *boundary* object owns setup-specific state, so a
new PDE or flow case is one new file, not a fork of the stepper.

## Reproducibility

- Bit-exact **within** a backend: the CPU path reproduces the production
  solver to the last bit (tested, `tests/test_regression_bitexact.py`);
  the GPU path with `--rng legacy-cupy` reproduces the published dataset
  seed-for-seed (verify with `benchmarks/bench_gpu_cluster.py --paper-npy ...`).
- Identical initial data **across** backends (`portable` RNG, default).
- Pathwise CPU/GPU bit-equality at t = 0.0035 is *not* claimed: reductions
  and transcendentals differ by ULPs between backends and the flow is
  chaotic; the benchmark reports the divergence growth instead.

## Reproducing the paper dataset

    cd examples/mach2000_jet
    ./jobs.sh gen-full                # 20-node GPU array, --rng legacy-cupy, writes $DATA_FULL
    ./jobs.sh merge --after <jobid>   # merge the part files into samples_{Nx}.npy

The analysis stages read the dataset from `DATA_NATIVE`. To analyse the
regenerated data, set `DATA_NATIVE` to the same directory as `DATA_FULL` in
`acropora-env.sh`; to analyse the published dataset instead, point it there.
The two are kept separate so a regeneration never overwrites existing data.

The Zenodo record ([10.5281/zenodo.23415318](https://doi.org/10.5281/zenodo.23415318))
holds the ensembles restricted to the coarsest grid, the convergence and
centerline results, and the generation records. To rerun the convergence study
without the GPU generation, rebuild the ensemble files there with
`bash merge_parts.sh`, set `DATA_ON500` to that folder in `acropora-env.sh`,
and run `./jobs.sh main`, `./jobs.sh w2-reference` and `./jobs.sh w2-cauchy`.

Post-processing (`acropora.post`, `pip install acropora[post]`) reproduces the
paper's quantities: both comparison protocols (Cauchy: consecutive pairs;
reference: each vs the finest grid) with selectable restriction (coarsest,
the default and published methodology: every field is block-averaged to the
coarsest grid, so all comparisons are evaluated there; pairwise: only the
finer field is averaged, onto the coarser member's native grid), the
one-point per-cell Wasserstein distance, the two-point Wasserstein distance
(POT exact EMD, spatial subsampling), the per-sample strong L1
error, ensemble mean/std L1 errors, log-log rate fits, the centerline
percentile bands and one-point density grids, the log-scale PDF heatmaps
(rasterized to PNG with the matching `\addplot graphics` code emitted), and
the quantile-band diagnostics behind the statements about those figures.
Golden tests assert value-identical output against the reference analysis
script in `baseline/`.

Resolutions are the x-resolutions `Nx` -- arguments, file
names, and CSV columns -- matching the data files: the jet ladder is
`--resolutions 500 1000 2000 4000`, producing `samples_500.npy` ...
`samples_4000.npy`, `centerline_nx500_...` ... `centerline_nx4000_...`, and
`coarse_nx` / `fine_nx` / `compare_nx` columns. The transverse count and mesh
width follow from the domain via `grid_from_nx` (`Nx=500` -> 500x400,
`dx=5e-3`).

## Batch jobs

Batch jobs are submitted through one driver that reads a site file:

    cp acropora-env.example.sh acropora-env.sh     # at the repository root, git-ignored
    $EDITOR acropora-env.sh                      # venv, partitions, account, data paths
    cd examples/mach2000_jet
    ./jobs.sh restrict                         # once: restrict onto the coarsest grid
    ./jobs.sh main --after <jobid>             # convergence study (chained)
    ./jobs.sh w2-reference; ./jobs.sh w2-cauchy
    ./jobs.sh centerline                       # reads the native data
    ./jobs.sh verify-gpu                       # GPU bit-exactness vs the dataset
    ./jobs.sh gen-test                         # small regeneration on one GPU node

`./jobs.sh <stage> --dry-run` prints the resolved paths and the exact sbatch
command without submitting. Nothing has to be exported in the shell: the jobs
load the site file themselves, so a fresh login cannot break them.


## Funding

*Funded by the Federal Ministry of Research, Technology and Space (BMFTR)
through the PRIME programme of the German Academic Exchange Service (DAAD).
Responsibility for the content of this publication lies with the author.*

<p>
  <img src="https://raw.githubusercontent.com/stephan-911/acropora/main/docs/logos/bmftr_funded_by.svg"
       alt="With funding from the Federal Ministry of Research, Technology and Space"
       height="130">
  &nbsp;&nbsp;
  <img src="https://raw.githubusercontent.com/stephan-911/acropora/main/docs/logos/daad_logo.svg"
       alt="DAAD – German Academic Exchange Service"
       height="130">
</p>

## Citing

If you use acropora, please cite the companion paper (see `CITATION.cff`,
which GitHub and Zenodo parse automatically):

    S. Simonis and G. Wissocq,
    Computing statistical solutions of a Mach 2000 astrophysical jet,
    arXiv:2605.25282 (2026).
    https://doi.org/10.48550/arXiv.2605.25282

## License

BSD-2-Clause-Patent (BSD 2-clause license with an explicit patent grant).
Copyright 2026 Stephan Simonis. Full text in `LICENSE`.

The acropora logo is © 2026 Stephan Simonis and is not covered by the BSD-2-Clause-Patent license.

