# This file is part of acropora: computing statistical solutions of
# nonlinear partial differential equations.
#
# Copyright 2026 Stephan Simonis
# SPDX-License-Identifier: BSD-2-Clause-Patent

"""Post-processing tests: unit checks on every metric, and a golden test
running the reference analysis script (baseline/wasserstein_master.py) against
acropora.post on identical data -- outputs must agree value-for-value.
"""
import os
import shutil
import struct
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from acropora import (Euler2D, VectorialD2Q5, Mach2000JetBC,  # noqa: E402
                     run_sample, grid_from_nx)
from acropora.io import create_store  # noqa: E402
from acropora.post import (coarsen_to, w1_percell, strong_l1, moment_l1,  # noqa: E402
                          steps_for, fit_slope, run_convergence,
                          centerline_stats, centerline_pdf_grid, merge_parts,
                          write_centerline_dat, write_pdf_grid_dat,
                          coarsen_pdf_grid, band_diagnostics,
                          load_pdf_grid, render_pdf_png, global_scale,
                          panel_tex, colorbar_tex)
from acropora.io.npy_store import npy_header_offset  # noqa: E402
from acropora.post.heatmap import rasterize  # noqa: E402

# ---------------- unit tests ----------------

def test_coarsen_block_mean():
    rng = np.random.default_rng(0)
    f = rng.normal(size=(3, 50, 40))          # Nx 50 -> Nx 25 (25x20)
    c = coarsen_to(f, 50, 25)
    assert c.shape == (3, 25, 20)
    assert np.allclose(c[1, 4, 7],
                       f[1, 8:10, 14:16].mean())


def test_w1_percell_matches_scipy():
    import scipy.stats
    rng = np.random.default_rng(1)
    a = rng.normal(size=(30, 4, 3))
    b = rng.normal(1.0, size=(30, 4, 3))
    direct = np.mean([scipy.stats.wasserstein_distance(
        a.reshape(30, -1)[:, i], b.reshape(30, -1)[:, i]) for i in range(12)])
    assert abs(w1_percell(a, b) - direct) < 1e-15


def test_strong_l1_hand_value():
    a = np.ones((2, 2, 2)); b = 2.0 * np.ones((2, 2, 2))
    # per sample: sum|a-b| / sum|b| = 4/8 = 0.5
    assert strong_l1(a, b) == 0.5


def test_moment_l1_hand_value():
    a = np.stack([np.zeros((2, 2)), 2*np.ones((2, 2))])   # mean 1, std 1
    b = np.stack([np.ones((2, 2)), np.ones((2, 2))])      # mean 1, std 0
    me, se = moment_l1(a, b)
    assert me == 0.0 and se == 1.0


def test_steps_and_slope():
    assert steps_for("cauchy", [500, 1000, 2000]) == [
        (500, 1000, 2.5/500), (1000, 2000, 2.5/1000)]
    assert steps_for("reference", [500, 1000, 2000]) == [
        (500, 2000, 2.5/500), (1000, 2000, 2.5/1000)]
    dxs = [2.5/500, 2.5/1000, 2.5/2000]
    vals = [7.0 * dx**0.75 for dx in dxs]
    assert abs(fit_slope(dxs, vals) - 0.75) < 1e-12


def test_fit_abscissa_is_the_mesh_width():
    """dx_fit is the paper's Delta x = 2.5/Nx of the coarser member, and a
    constant rescaling of it leaves the fitted rates unchanged."""
    cau = steps_for("cauchy", [500, 1000, 2000, 4000])
    ref = steps_for("reference", [500, 1000, 2000, 4000])
    assert [s[2] for s in cau] == [5.0e-3, 2.5e-3, 1.25e-3]
    assert [(s[0], s[1], s[2]) for s in ref] == [
        (500, 4000, 5.0e-3), (1000, 4000, 2.5e-3), (2000, 4000, 1.25e-3)]
    vals = [1.4e-2, 9.0e-3, 6.0e-3]
    dx = [s[2] for s in cau]
    assert abs(fit_slope(dx, vals) - fit_slope([0.5 * d for d in dx], vals)) < 1e-12


def test_published_cauchy_w1_rates_reproduced():
    """Fitting the paper's per-pair W1 values must return its printed
    Cauchy W1 rates for every checkpoint."""
    dx = [s[2] for s in steps_for("cauchy", [500, 1000, 2000, 4000])]
    w1 = {1: [5.812775e-3, 4.157481e-3, 3.461188e-3],
          2: [1.646590e-2, 1.480491e-2, 1.029525e-2],
          3: [3.695049e-2, 3.022408e-2, 2.141176e-2],
          4: [6.059197e-2, 4.975557e-2, 3.926781e-2],
          5: [9.703380e-2, 7.553064e-2, 6.082539e-2],
          6: [1.232927e-1, 1.097492e-1, 7.735545e-2],
          7: [1.496704e-1, 1.409947e-1, 9.319198e-2]}
    published = {1: 0.374, 2: 0.339, 3: 0.394, 4: 0.313,
                 5: 0.337, 6: 0.336, 7: 0.342}
    for ck, vals in w1.items():
        assert round(fit_slope(dx, vals), 3) == published[ck], ck


def test_centerline_stats_and_pdf():
    rng = np.random.default_rng(2)
    M, Nx, Ny, T = 40, 10, 8, 2
    data = rng.normal(5.0, 1.0, size=(M, T, Nx, Ny))
    fp = "/tmp/_cl_test.npy"
    st = create_store(fp, M, T, Nx, Ny); st[:] = data; st.flush(); del st
    hoff = npy_header_offset(fp)
    rows = slice(Ny//2, Ny//2 + 1)
    mean, std, n, p16, p84, p05, p95, stack = centerline_stats(
        fp, hoff, np.arange(M), 1, T, Nx, Ny, rows, return_stack=True)
    ref = data[:, 1, :, Ny//2]
    assert n == M
    assert np.allclose(mean, ref.mean(axis=0))
    assert np.allclose(std, ref.std(axis=0))
    assert np.allclose(p16, np.percentile(ref, 16.0, axis=0))
    rc, counts, pdf = centerline_pdf_grid(stack, nbins=16)
    assert counts.shape == (Nx, 16)
    d_rho = rc[1] - rc[0]
    assert np.allclose(pdf.sum(axis=1) * d_rho, 1.0)
    os.remove(fp)


# ---------------- golden test vs reference script ----------------

def _rows(path):
    """Non-empty, non-comment CSV rows below the header."""
    out = []
    for line in open(path).read().splitlines()[1:]:
        if line.strip() and not line.startswith("#"):
            out.append(line.split(","))
    return out


def _generate_tiny(data_dir, samples=6, resolutions=(25, 50), T=3, max_t=2e-4):
    tt = np.linspace(0, max_t, T)
    for nx in resolutions:
        Nx, Ny, dx = grid_from_nx(nx)
        st = create_store(os.path.join(data_dir, f"samples_{Nx}.npy"),
                          samples, T, Nx, Ny)
        for s_id in range(samples):
            bc = Mach2000JetBC.make("ghost", Ny, seed=42 + s_id,
                                    rng_mode="portable")
            sol = VectorialD2Q5(Euler2D(5/3), Nx, Ny, dx, xp=np, boundary=bc)
            bc.initial_condition(sol)
            run_sample(sol, tt, lambda ck, fr, s_id=s_id: st.__setitem__((s_id, ck), fr))
        st.flush(); del st
    return tt


def test_golden_vs_reference_script():
    """The library must reproduce the reference analysis script value for
    value, in both restriction modes."""
    try:
        import matplotlib  # noqa: F401  (the reference script imports it)
    except ImportError:
        return
    wd = "/tmp/_post_golden"
    shutil.rmtree(wd, ignore_errors=True)
    os.makedirs(os.path.join(wd, "data"))
    _generate_tiny(os.path.join(wd, "data"), resolutions=(25, 50, 100), T=3)

    for restriction in ("coarsest", "pairwise"):
        ref_out = os.path.join(wd, f"ref_{restriction}")
        lib_out = os.path.join(wd, f"lib_{restriction}")
        subprocess.run(
            [sys.executable,
             os.path.join(ROOT, "baseline", "wasserstein_master.py"),
             "--samples", "6", "--resolutions", "25", "50", "100",
             "--num_checkpoints", "3", "--max_t", "0.0002",
             "--fixedM", "6", "--protocols", "both", "--marginal", "1",
             "--restriction", restriction,
             "--data_dir", os.path.join(wd, "data"), "--out_dir", ref_out],
            check=True, capture_output=True)
        run_convergence(data_dir=os.path.join(wd, "data"), out_dir=lib_out,
                        samples=6, resolutions=(25, 50, 100),
                        num_checkpoints=3, max_t=2e-4, fixedM=6, draws=1,
                        protocols="both", marginal="1",
                        restriction=restriction, log=lambda *a: None)
        # Both write the same quantities; the library additionally carries a
        # draw column and a summary block, so compare the shared numeric
        # fields keyed by (protocol, checkpoint, coarse_nx, fine_nx).
        ref_keys, lib_keys = (0, 1, 3, 4), (0, 1, 4, 5)
        for name, valcols in [("w1_values.csv", {"ref": 9, "lib": 10}),
                              ("strong_values.csv", {"ref": 8, "lib": 9}),
                              ("moments_values.csv", {"ref": 9, "lib": 10})]:
            ref = _rows(os.path.join(ref_out, name))
            lib = _rows(os.path.join(lib_out, name))
            assert len(ref) == len(lib), f"{name}: row count differs"
            for rr, ll in zip(ref, lib):
                assert ([rr[i] for i in ref_keys]
                        == [ll[i] for i in lib_keys]), name
                assert rr[valcols["ref"]] == ll[valcols["lib"]], \
                    f"{name}: {rr[valcols['ref']]} != {ll[valcols['lib']]}"
                if name == "moments_values.csv":
                    assert rr[10] == ll[11]
        # slopes
        rs = _rows(os.path.join(ref_out, "slopes.csv"))
        ls = [r for r in _rows(os.path.join(lib_out, "slopes.csv"))
              if r and not r[0].startswith("#")][:len(rs)]
        for rr, ll in zip(rs, ls):
            assert rr[0] == ll[0] and rr[1] == ll[1]
            assert rr[3:8] == ll[4:9], (rr, ll)
    shutil.rmtree(wd)


def test_coarsest_is_default_and_records_common_grid():
    """The published methodology: every field averaged to the coarsest grid,
    which the compare_grid column must report for every pair."""
    wd = "/tmp/_coarsest_test"
    shutil.rmtree(wd, ignore_errors=True)
    os.makedirs(os.path.join(wd, "data"))
    _generate_tiny(os.path.join(wd, "data"), samples=4,
                   resolutions=(25, 50, 100), T=2, max_t=1e-4)
    out = run_convergence(data_dir=os.path.join(wd, "data"),
                          out_dir=os.path.join(wd, "o"), samples=4,
                          resolutions=(25, 50, 100), num_checkpoints=2,
                          max_t=1e-4, fixedM=4, draws=1, protocols="both",
                          marginal="1", log=lambda *a: None)   # no restriction arg
    assert {r["cg"] for r in out["w1"]} == {25}, "default must compare on coarsest grid"
    assert {r["cg"] for r in out["strong"]} == {25}
    assert {r["cg"] for r in out["moments"]} == {25}
    csv_text = open(os.path.join(wd, "o", "w1_values.csv")).read().splitlines()
    for line in csv_text[1:]:
        if line:
            assert line.split(",")[9] == "25"   # the coarsest Nx
    shutil.rmtree(wd)


def test_csv_columns_report_actual_x_resolution():
    """Outputs must carry the x-resolutions Nx that name the data files."""
    wd = "/tmp/_nx_test"
    shutil.rmtree(wd, ignore_errors=True)
    os.makedirs(os.path.join(wd, "data"))
    _generate_tiny(os.path.join(wd, "data"), samples=4,
                   resolutions=(25, 50, 100), T=2, max_t=1e-4)
    run_convergence(data_dir=os.path.join(wd, "data"),
                    out_dir=os.path.join(wd, "o"), samples=4,
                    resolutions=(25, 50, 100), num_checkpoints=2, max_t=1e-4,
                    fixedM=4, draws=1, protocols="cauchy", marginal="1",
                    log=lambda *a: None)
    lines = open(os.path.join(wd, "o", "w1_values.csv")).read().splitlines()
    assert lines[0].split(",")[4:6] == ["coarse_nx", "fine_nx"]
    assert lines[0].split(",")[9] == "compare_nx"
    body = [l.split(",") for l in lines[1:] if l]
    assert {r[4] for r in body} <= {"25", "50"}
    assert {r[5] for r in body} <= {"50", "100"}
    assert {r[9] for r in body} == {"25"}
    shutil.rmtree(wd)


def test_pdf_grid_dat_roundtrip_and_coarsening():
    """The long-format grid must reload to the same array, and coarsening
    must preserve the total count."""
    rng = np.random.default_rng(5)
    stack = rng.normal(5.0, 1.0, size=(50, 12))
    x = np.linspace(0, 2.5, 12)
    rc, counts, pdf = centerline_pdf_grid(stack, nbins=8)
    p = "/tmp/_pdfgrid.dat"
    write_pdf_grid_dat(p, x, rc, pdf, counts, header="test grid")
    xu, yu, F = load_pdf_grid(p)
    assert np.allclose(xu, x) and np.allclose(yu, rc)
    assert np.allclose(F, pdf.T)          # F is [rho, x]
    xc, pdfc, cntc = coarsen_pdf_grid(x, pdf, counts, 4)
    assert xc.shape == (4,) and pdfc.shape == (4, 8)
    assert cntc.sum() == counts.sum()
    os.remove(p)


def test_heatmap_png_and_tex():
    """Rasterization writes a valid PNG of the requested size and the
    emitted pgfplots code carries the shared scale."""
    rng = np.random.default_rng(6)
    stack = rng.normal(5.0, 1.0, size=(60, 10))
    x = np.linspace(0, 2.5, 10)
    rc, counts, pdf = centerline_pdf_grid(stack, nbins=8)
    src, dst = "/tmp/_hm.dat", "/tmp/_hm.png"
    write_pdf_grid_dat(src, x, rc, pdf, counts)
    vmin, vmax = global_scale([src])
    assert vmin < vmax
    extents, pix, cells, _ = render_pdf_png(src, dst, vmin, vmax,
                                            x_window=(0.0, 2.5),
                                            y_window=(0.0, 10.0),
                                            width_pix=300)
    assert cells == (10, 8) and pix[0] == 300
    with open(dst, "rb") as fh:
        head = fh.read(8)
        assert head == b"\x89PNG\r\n\x1a\n"
        fh.seek(16)
        w, h = struct.unpack(">II", fh.read(8))
        assert (w, h) == pix
    tex = panel_tex(dst, extents, vmin, vmax, "caption")
    assert "addplot graphics" in tex and f"point meta min={vmin:.3f}" in tex
    assert "colorbar horizontal" in colorbar_tex(vmin, vmax)
    os.remove(src); os.remove(dst)


def test_heatmap_leaves_window_outside_data_empty():
    """Pixels of the plot window beyond the density axis must be drawn as
    empty, not as copies of the edge bins (which shows as vertical stripes
    wherever an edge bin is occupied)."""
    xu = np.array([0.5, 1.5, 2.5])
    yu = np.array([10.0, 20.0, 30.0])           # bins cover rho in [5, 35]
    F = np.zeros((3, 3))
    F[0, 1] = 1.0                               # occupied bottom bin
    F[2, 1] = 1.0                               # occupied top bin
    img, _, _, _ = rasterize(F, xu, yu, (0.0, 3.0), (0.0, 60.0),
                             width_pix=30, aspect=(1, 2), fill=-1.0)
    py = (np.arange(img.shape[0]) + 0.5) * 60.0 / img.shape[0]
    outside = (py < 5.0) | (py > 35.0)
    assert np.all(img[outside] == -1.0)
    assert np.all(img[~outside][:, 10:20].max(axis=0) == 1.0)


def test_band_diagnostics_detects_heavy_tails_and_exceedance():
    """Synthetic bands: a Gaussian-like window gives ratio ~1.645; a
    heavy-tailed window gives a much larger ratio; an over-p95 sample is
    reported."""
    x = np.linspace(1.0, 2.5, 60)
    mean = np.full_like(x, 10.0)
    std = np.ones_like(x)
    # gaussian quantiles
    p16, p84 = mean - std, mean + std
    p05, p95 = mean - 1.6449 * std, mean + 1.6449 * std
    sample = mean.copy()
    d = "/tmp/_band.dat"
    write_centerline_dat(d, x, sample, mean, std, p16, p84, p05, p95)
    r = band_diagnostics(d, head_region=(1.4, 2.3))
    assert abs(r["ratio_median"] - 1.6449) < 1e-6
    assert r["heavy_tailed"] is False
    assert r["sample_exceeds_p95"] is None
    # heavy tail + exceedance
    p95_h = mean + 8.0 * std
    sample_h = mean + 9.0 * std
    write_centerline_dat(d, x, sample_h, mean, std, p16, p84, p05, p95_h)
    r2 = band_diagnostics(d, head_region=(1.4, 2.3))
    assert r2["ratio_median"] > 4.0 and r2["heavy_tailed"]
    assert r2["skew_proxy_median"] > 1.0
    assert r2["sample_exceeds_p95"]["count"] > 0
    os.remove(d)


def test_prerestricted_dataset_matches_on_the_fly_restriction():
    """Pre-restricting the ensembles and analyzing them must give exactly the
    same numbers as restricting on the fly during the study."""
    import subprocess
    wd = "/tmp/_restrict_ds"
    shutil.rmtree(wd, ignore_errors=True)
    os.makedirs(os.path.join(wd, "data"))
    _generate_tiny(os.path.join(wd, "data"), samples=4,
                   resolutions=(25, 50, 100), T=2, max_t=1e-4)

    script = os.path.join(ROOT, "examples", "mach2000_jet",
                          "restrict_dataset.py")
    subprocess.run([sys.executable, script,
                    "--data-dir", os.path.join(wd, "data"),
                    "--out-dir", os.path.join(wd, "on25"),
                    "--resolutions", "25", "50", "100", "--target", "25",
                    "--samples", "4", "--num-checkpoints", "2",
                    "--max-t", "0.0001", "--verify", "3"],
                   check=True, capture_output=True)

    kw = dict(samples=4, resolutions=(25, 50, 100), num_checkpoints=2,
              max_t=1e-4, fixedM=4, draws=1, protocols="both",
              marginal="1", log=lambda *a: None)
    native = run_convergence(data_dir=os.path.join(wd, "data"),
                             out_dir=os.path.join(wd, "n"),
                             restriction="coarsest", **kw)
    pre = run_convergence(data_dir=os.path.join(wd, "on25"),
                          out_dir=os.path.join(wd, "p"),
                          restricted_on=25, **kw)
    for key in ("w1", "strong", "moments", "slopes"):
        assert len(native[key]) == len(pre[key])
    for a, b in zip(native["w1"], pre["w1"]):
        assert a["w1"] == b["w1"], (a, b)
    for a, b in zip(native["strong"], pre["strong"]):
        assert a["err"] == b["err"]
    for a, b in zip(native["moments"], pre["moments"]):
        assert a["me"] == b["me"] and a["se"] == b["se"]
    def same(u, v):
        return (np.isnan(u) and np.isnan(v)) or u == v
    for a, b in zip(native["slopes"], pre["slopes"]):
        for k in ("w1s", "sts", "mes", "ses"):
            assert same(a[k], b[k]), (k, a[k], b[k])
    shutil.rmtree(wd)


def test_centerline_cli_stages_run():
    """Execute the centerline, heatmap and band-diagnostic stages through the
    command line exactly as the batch jobs do, so import-scope or wiring
    errors in the CLI are caught here rather than in a queued job."""
    wd = "/tmp/_cli_stages"
    shutil.rmtree(wd, ignore_errors=True)
    os.makedirs(os.path.join(wd, "data"))
    _generate_tiny(os.path.join(wd, "data"), samples=4,
                   resolutions=(25, 50), T=3, max_t=2e-4)
    cli = os.path.join(ROOT, "examples", "mach2000_jet", "run_postprocess.py")
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "src"))
    base = [sys.executable, cli,
            "--data-dir", os.path.join(wd, "data"),
            "--centerline-out", os.path.join(wd, "centerline"),
            "--samples", "4", "--resolutions", "25", "50",
            "--num-checkpoints", "3", "--max-t", "0.0002"]
    for extra in (["--centerline", "--pdf-grid", "--pdf-nbins", "12",
                   "--pdf-xslices", "5"],
                  ["--heatmap", "--heatmap-nx", "50"],
                  ["--check-bands", "--head-region", "0.1", "2.0"]):
        r = subprocess.run(base + extra, capture_output=True, text=True, env=env)
        assert r.returncode == 0, (extra, r.stderr[-600:])
    cdir = os.path.join(wd, "centerline")
    out = sorted(os.listdir(cdir))
    assert any(f.startswith("centerline_nx50_") for f in out), out
    assert any(f.startswith("centerline_pdf_nx50_") for f in out), out
    assert any(f.endswith(".png") for f in out), out
    # PDF grids exist for t > 0 only and share one density axis from 0
    pdfs = sorted(f for f in out if f.startswith("centerline_pdf_nx50_"))
    assert [f.split("_")[3] for f in pdfs] == ["ck1", "ck2"], pdfs
    axes = [load_pdf_grid(os.path.join(cdir, f))[1] for f in pdfs]
    assert np.array_equal(axes[0], axes[1])
    assert np.isclose(axes[0][0] - 0.5 * (axes[0][1] - axes[0][0]), 0.0,
                      atol=1e-6)                 # lowest bin edge is rho = 0
    # centerline x are cell centers
    xs = np.loadtxt(os.path.join(cdir, out[0]))[:, 0]
    assert np.isclose(xs[0], 0.5 * (xs[1] - xs[0]))
    # the LaTeX and the band report go into published files: no directories
    for extra in (["--heatmap", "--heatmap-nx", "50"],
                  ["--check-bands", "--head-region", "0.1", "2.0"]):
        r = subprocess.run(base + extra, capture_output=True, text=True, env=env)
        assert r.returncode == 0, (extra, r.stderr[-600:])
        assert wd not in r.stdout, (extra, r.stdout[:600])
    shutil.rmtree(wd)


def test_generation_cli_rng_modes():
    """Run the generation CLI as the batch jobs do.

    The portable path must produce data and a coefficient file; the
    legacy-cupy path must be rejected up front on a CPU backend rather than
    failing later inside the coefficient loop.
    """
    wd = "/tmp/_gen_cli"
    shutil.rmtree(wd, ignore_errors=True)
    os.makedirs(wd)
    cli = os.path.join(ROOT, "examples", "mach2000_jet", "run_generation.py")
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "src"))
    base = [sys.executable, cli, "--samples", "2", "--resolutions", "25",
            "--max-t", "0.0002", "--num-checkpoints", "3",
            "--backend", "cpu", "--workers", "1", "--num-devices", "0",
            "--node-id", "0", "--total-nodes", "1"]

    r = subprocess.run(base + ["--rng", "portable",
                               "--data-dir", os.path.join(wd, "ok")],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr[-600:]
    produced = os.listdir(os.path.join(wd, "ok"))
    assert "samples_25_part0.npy" in produced, produced
    assert "fourier_coeffs_part0.csv" in produced, produced
    assert "manifest_part0.json" in produced, produced

    r = subprocess.run(base + ["--rng", "legacy-cupy",
                               "--data-dir", os.path.join(wd, "bad")],
                       capture_output=True, text=True, env=env)
    assert r.returncode != 0
    assert "requires --backend gpu" in (r.stderr + r.stdout), r.stderr[-400:]
    shutil.rmtree(wd)


def test_legacy_cupy_coefficients_use_the_device_rng():
    """draw_coefficients must consult the array namespace it is given.

    A stand-in for cupy records the calls, so the legacy-cupy path is covered
    without a GPU; passing numpy instead must still be refused.
    """
    import types as _types
    from acropora import draw_coefficients

    calls = {"seed": [], "uniform": 0}

    class _FakeRandom:
        @staticmethod
        def seed(s):
            calls["seed"].append(s)

        @staticmethod
        def uniform(lo, hi):
            calls["uniform"] += 1
            return 0.25

    fake = _types.SimpleNamespace(random=_FakeRandom)
    Y, Z = draw_coefficients(7, 10, "legacy-cupy", fake)
    assert calls["seed"] == [7]
    assert calls["uniform"] == 20          # Y and Z for ten modes
    assert np.allclose(Y, 0.25) and np.allclose(Z, 0.25)

    try:
        draw_coefficients(7, 10, "legacy-cupy", np)
        raise AssertionError("numpy namespace must be refused")
    except ValueError:
        pass


def test_merge_missing_part_raises():
    wd = "/tmp/_merge_test"
    shutil.rmtree(wd, ignore_errors=True); os.makedirs(wd)
    p0 = os.path.join(wd, "p0.npy")
    st = create_store(p0, 2, 2, 4, 4); st[:] = 1.0; st.flush(); del st
    try:
        merge_parts(os.path.join(wd, "m.npy"), [p0, os.path.join(wd, "gone.npy")],
                    4, 2, 4, 4, parts_sizes=[2, 2])
        raise AssertionError("missing part not caught")
    except FileNotFoundError:
        pass
    merge_parts(os.path.join(wd, "m2.npy"), [p0, os.path.join(wd, "gone.npy")],
                4, 2, 4, 4, parts_sizes=[2, 2], allow_gaps=True)
    m = np.load(os.path.join(wd, "m2.npy"), mmap_mode="r")
    assert np.all(np.asarray(m[0]) == 1.0) and np.all(np.asarray(m[2]) == 0.0)
    shutil.rmtree(wd)
