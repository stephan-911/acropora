# Code scan report: `lbm_video_h200_corr.py` (production baseline)

Scan of the script that generated the paper dataset (5 H200 logs cross-checked:
zero numerical stalls, per-grid timings consistent to ±0.5%). Findings are
classified by severity. "Fixed" refers to the restructured package; the
baseline script itself is kept untouched as the regression reference (single
changed line: `import cupy as cp` -> shim, in `baseline/lbm_baseline_numpy.py`).

## A. Correctness-relevant findings

**I-1 (dead import).** `import ot` (Python Optimal Transport) is never used.
Cosmetic in effect, but it adds a hard dependency that can break deployment
(the module fails to import on any node without POT installed, even for pure
generation runs). *Fixed: removed; dependencies are now declared per-extra in
`pyproject.toml`.*

**I-2 (restart logic loses samples, MEDIUM).** Mode-1 skips a grid if
`data/samples_{Nx}_part{id}.npy` exists and is >1 KiB. `open_memmap(mode='w+')`
creates the full-size (sparse) file *before* any sample is computed, so a node
that dies mid-grid leaves a full-size file; on resubmission the grid is
skipped wholesale and the missing samples remain zero forever. They are later
caught by the health mask ("unwritten") and silently dropped, which biases the
ensemble size, not the physics — but the script's own "Restartable" claim is
not met. *Fixed: per-sample completion is detected (final-checkpoint block
nonzero) and only missing samples are rerun.*

**I-3 (silent index shift on missing parts, HIGH if triggered).** The Mode-2
fast-merge concatenates part files in node order and `continue`s over missing
parts. A missing part shifts all subsequent samples' positions in the merged
array, decoupling array index from sample id (and hence from the recorded
Fourier coefficients and seeds). With all 20 parts present (as in the paper
runs) this never triggered; the hazard is real for reuse. *Fix specified (hard error on missing parts, explicit `--allow-gaps` writing
zeros to keep index alignment); ships with the post-processing module
(version 2).*

**I-4 (remainder samples never run, LOW).** `samples_per_node = samples //
total_nodes` drops `samples % total_nodes` samples entirely (0 for 1000/20).
*Fixed: remainder distributed over the first nodes.*

**I-5 (checkpoint semantics, DOCUMENTED).** Checkpoints store the field at the
first time `t >= target`, i.e. up to one `dt` late, with no interpolation; if
a worker aborts, all remaining checkpoints receive the last valid field (the
stall detector in the health mask then flags the sample via the equality of
the last two checkpoints). Both behaviors are intentional and now documented;
note the stall check would also flag a genuinely stationary flow — impossible
for this jet, but relevant for other setups.

**I-6 (asymmetry health check vs. asymmetric physics, DOCUMENTED).** The
health mask rejects samples whose top/bottom mean |difference| exceeds 0.5.
The inlet perturbation deliberately breaks symmetry, so this check is
conceptually at odds with the setup; it exists to catch "splattered"
structural anomalies. In the paper runs it rejected nothing (1000/1000
healthy). *To be kept as an optional, documented check with the threshold exposed;
ships with the post-processing module (version 2).*

**I-7 (RNG is backend-bound, MEDIUM for portability).** Perturbation
coefficients are drawn from CuPy's legacy global RNG (`cp.random.seed`;
XORWOW). The same seed on NumPy yields different values, so a CPU port cannot
reproduce the GPU dataset by seed alone, and results are not portable across
backends. *Fixed: the package draws coefficients from a backend-independent
`numpy.random.Generator(PCG64)` by default ("portable" mode), so CPU and GPU
runs share identical initial data; a "legacy-cupy" mode reproduces the paper
dataset bit-for-bit on GPU, and a "legacy-numpy" mode reproduces the shim
baseline for regression tests.*

**I-8 (per-sample process respawn, PERFORMANCE).** Mode 1 submits one process
per sample; each pays CUDA context creation (seconds) ~200 times per node.
Harmless for correctness; measurable overhead at small grids. *Fixed:
persistent worker pool with a per-worker device initializer.*

**I-9 (mode-coupled CLI hazard, LOW).** `--render_video` overrides
`num_checkpoints` (duration*fps); running post-processing with flags different
from generation silently misindexes frames (shape mismatch is only caught if
sizes disagree). *Fixed: generation writes a `manifest.json` (shape, times,
seeds, rng mode, git hash); post-processing reads it instead of trusting CLI.*

**I-10 (removed private NumPy API, HIGH under NumPy >= 2).** Post-processing
uses `np.lib.format._read_array_header`, deleted in NumPy 2.x; Mode 2 crashes
on modern environments (generation is unaffected). *Fixed in the generation-side reader (`io/npy_store.py`); the post-processing
module (version 2) must use the same public API.*

## B. Verified-correct items worth recording (checked, no action)

- Limiter/streaming algebra: the rest population absorbs exactly the injected
  anti-diffusive corrections, so collision conserves `U` before streaming;
  interface thetas are consistently paired (`theta_x` with `A1`,
  `roll(theta_x,1)` with `A2`, etc.).
- Ghost padding (3 cells) strictly exceeds the 1-cell stencil, so `roll`
  wraparound only contaminates pads, which `apply_boundaries` overwrites each
  step. Outlet and transverse boundaries are zero-gradient copies of the last
  interior line; inlet pads are forced to the (perturbed) inlet state and
  re-equilibrated.
- Grid convention: nodes at `x=i*dx`, `y=j*dy-1`; `y=0` is a grid line
  (index `Ny/2`), and `0.05/dy` is integer for all four resolutions, so the
  jet mask is symmetric (19/39/79/159 nodes) at every grid.
- Energy of the perturbed inlet uses fixed `v=800`, `p=0.4127` with perturbed
  density only — consistent with the paper's setup and the Mach-variability
  statement therein.
- Coefficient recording (`fourier_coeffs_part*.dat`) replays the exact draw
  order (per k: Y then Z) of the perturbation function.
- Block-mean restriction fine->coarse and the relative L1 strong error match
  the definitions used in the paper's pairwise protocol; W1 normalization by
  point count differs from the integral by the constant |D|, leaving rates
  unaffected.
- `dt` uses the pre-step `a_n`; `theta` bounds are clipped to [0,1]; the
  `1e-12`/`1e-14` guards floor only denominators.

## C. Cross-backend reproducibility (design position)

Elementwise IEEE-754 double ops (+,-,*,/,sqrt) are correctly rounded and
reproduce bit-exactly across NumPy and CuPy. Reductions (`max` in the wave
speed) and transcendentals (`cos`/`sin` in the perturbation) may differ by
ULPs between backends; in a Mach-2000 chaotic flow, single-ULP differences
amplify exponentially. Therefore the package guarantees:
(1) bit-exact reproducibility *within* a backend (tested),
(2) identical initial data *across* backends (portable RNG, tested),
(3) cross-backend agreement to tolerance over short horizons plus statistical
agreement over long horizons (cluster benchmark provided) — but *not* pathwise
bit-equality of GPU and CPU at t=0.0035, which is unattainable for this
problem and should not be claimed in the paper or README.
