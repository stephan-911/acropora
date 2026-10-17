# This file is part of acropora: computing statistical solutions of
# nonlinear partial differential equations.
#
# Copyright 2026 Stephan Simonis
# SPDX-License-Identifier: BSD-2-Clause-Patent

"""Rasterize centerline PDF grids to PNG and emit matching pgfplots code.

The PDF grids are drawn in LaTeX with \\addplot graphics rather than as
pgfplots surfaces, because a full-resolution (Nx x nbins) surface exceeds
TeX memory. Cells may be non-uniform in rho, so values are rasterized onto
uniform pixels through the true cell edges, which is what `matrix plot*`
would draw.

Color scale: meta = log10(max(f, floor)) clipped to [vmin, vmax], with
floor = 10**vmin; a shared scale across panels makes them comparable.
Only numpy and the standard library are used (the PNG is written directly),
so no plotting backend is required on a compute node.
"""
from __future__ import annotations
import struct
import zlib

import numpy as np

# pgfplots colormap/cool: white -> soft blue -> magenta
_COOL_STOPS_T = np.array([0.0, 0.5, 1.0])
_COOL_STOPS_RGB = np.array([[255, 255, 255],
                            [97, 142, 255],
                            [255, 0, 255]], dtype=float)


def load_pdf_grid(path):
    """Read a long-format (x, rho, pdf[, count]) table and pivot to a 2-D
    array F[rho, x]."""
    data = np.loadtxt(path)
    x, y, f = data[:, 0], data[:, 1], data[:, 2]
    xu = np.unique(np.round(x, 12))
    yu = np.unique(np.round(y, 12))
    F = np.full((yu.size, xu.size), np.nan)
    F[np.searchsorted(yu, np.round(y, 12)),
      np.searchsorted(xu, np.round(x, 12))] = f
    if np.isnan(F).any():
        raise ValueError(f"{path}: grid is not complete/regular "
                         f"({int(np.isnan(F).sum())} missing cells)")
    return xu, yu, F


def cell_edges(centers):
    """Cell edges (len n+1): midpoints between centers, extrapolated at the
    ends -- the sizing `matrix plot*` uses."""
    mid = 0.5 * (centers[1:] + centers[:-1])
    first = centers[0] - (centers[1] - centers[0]) / 2
    last = centers[-1] + (centers[-1] - centers[-2]) / 2
    return np.concatenate([[first], mid, [last]])


def rasterize(F, xu, yu, x_window=None, y_window=None,
              width_pix=8000, aspect=(75, 9), fill=None):
    """Nearest-cell rasterization onto uniform pixels. Pixels outside the
    data cells take `fill` (if given) instead of the nearest edge cell.
    Returns (image, (xlo, xhi, ylo, yhi), (nx_pix, ny_pix), warnings)."""
    xe, ye = cell_edges(xu), cell_edges(yu)
    xlo, xhi = (xe[0], xe[-1]) if x_window is None else x_window
    ylo, yhi = (ye[0], ye[-1]) if y_window is None else y_window

    nx_pix = width_pix
    ny_pix = int(round(nx_pix * aspect[1] / aspect[0]))

    warn = []
    for edges, lo, hi, npix, name in ((xe, xlo, xhi, nx_pix, "x"),
                                      (ye, ylo, yhi, ny_pix, "y")):
        widths = np.diff(edges)
        mask = (edges[1:] > lo) & (edges[:-1] < hi)
        if mask.any():
            need = int(np.ceil((hi - lo) / widths[mask].min()))
            if npix < need:
                warn.append(f"finest {name}-cells span < 1 px "
                            f"({need} px would resolve them; have {npix})")

    px = xlo + (np.arange(nx_pix) + 0.5) * (xhi - xlo) / nx_pix
    py = ylo + (np.arange(ny_pix) + 0.5) * (yhi - ylo) / ny_pix
    ix = np.clip(np.searchsorted(xe, px) - 1, 0, xu.size - 1)
    iy = np.clip(np.searchsorted(ye, py) - 1, 0, yu.size - 1)
    img = F[np.ix_(iy, ix)]
    if fill is not None:
        out_y = (py < ye[0]) | (py > ye[-1])
        out_x = (px < xe[0]) | (px > xe[-1])
        img = np.where(out_y[:, None] | out_x[None, :], fill, img)
    return img, (xlo, xhi, ylo, yhi), (nx_pix, ny_pix), warn


def cool_rgb(meta, vmin, vmax):
    """Map metadata values to pgfplots colormap/cool RGB."""
    t = np.clip((meta - vmin) / (vmax - vmin), 0.0, 1.0)
    rgb = np.empty(t.shape + (3,), dtype=np.uint8)
    for c in range(3):
        rgb[..., c] = np.round(np.interp(t, _COOL_STOPS_T,
                                         _COOL_STOPS_RGB[:, c]))
    return rgb


def write_png(path, rgb_bottom_up):
    """Minimal 8-bit RGB PNG writer (standard library only). Row 0 = bottom."""
    rgb = np.ascontiguousarray(rgb_bottom_up[::-1])
    ny, nx, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[i].tobytes() for i in range(ny))

    def chunk(tag, data):
        c = tag + data
        return (struct.pack(">I", len(data)) + c
                + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF))

    with open(path, "wb") as fh:
        fh.write(b"\x89PNG\r\n\x1a\n")
        fh.write(chunk(b"IHDR", struct.pack(">IIBBBBB", nx, ny, 8, 2, 0, 0, 0)))
        fh.write(chunk(b"IDAT", zlib.compress(raw, 9)))
        fh.write(chunk(b"IEND", b""))


def render_pdf_png(src_dat, dst_png, vmin, vmax, *, x_window=(0.0, 2.5),
                   y_window=(0.0, 150.0), width_pix=8000, aspect=(75, 9)):
    """Rasterize one PDF grid to PNG on the shared log color scale.
    Returns (extents, (nx_pix, ny_pix), n_cells, warnings)."""
    xu, yu, F = load_pdf_grid(src_dat)
    meta = np.log10(np.maximum(F, 10.0 ** vmin))
    img, extents, pix, warn = rasterize(meta, xu, yu, x_window, y_window,
                                        width_pix, aspect, fill=vmin)
    write_png(dst_png, cool_rgb(img, vmin, vmax))
    return extents, pix, (xu.size, yu.size), warn


def global_scale(paths):
    """Shared (vmin, vmax) = log10 of the global min f>0 and max f over all
    grids, so every panel carries the same color scale."""
    fmax = -np.inf
    fmin = np.inf
    for p in paths:
        _, _, F = load_pdf_grid(p)
        fmax = max(fmax, float(F.max()))
        pos = F[F > 0]
        if pos.size:
            fmin = min(fmin, float(pos.min()))
    if not np.isfinite(fmin) or not np.isfinite(fmax):
        raise ValueError("no positive density values found")
    return round(float(np.log10(fmin)), 3), round(float(np.log10(fmax)), 3)


def panel_tex(png, extents, vmin, vmax, caption, *, panel_w=0.75,
              panel_h=0.09, first_col=True, last_row=True, overlay_dat=None):
    """One subfloat panel. The axis box is panel_w x panel_h \\textwidth via
    `scale only axis` and the limits equal the raster window, so pixels stay
    square (panel_w:panel_h must equal the raster aspect)."""
    xlo, xhi, ylo, yhi = extents
    xlabel = r"xlabel={$x$}," if last_row else "xticklabels={},"
    ylabel = (r"ylabel={$\rho(x,0)$}," if first_col else "yticklabels={},")
    overlay = ""
    if overlay_dat:
        overlay = (f"\n                \\addplot[orange, line width=1.2pt,"
                   f" each nth point=10]\n                    "
                   f"table[x index=0, y index=2] {{{overlay_dat}}};")
    return rf"""    \subfloat[{caption}]{{%
        \begin{{tikzpicture}}
            \begin{{axis}}[
                scale only axis,
                width={panel_w}\textwidth, height={panel_h}\textwidth,
                {xlabel} {ylabel}
                xmin={xlo:.6g}, xmax={xhi:.6g},
                ymin={ylo:.6g}, ymax={yhi:.6g},
                enlargelimits=false, axis on top,
                colormap/cool,
                point meta min={vmin:.3f},
                point meta max={vmax:.3f}
                ]
                \addplot graphics[xmin={xlo:.6g}, xmax={xhi:.6g},
                    ymin={ylo:.6g}, ymax={yhi:.6g}] {{{png}}};{overlay}
            \end{{axis}}
        \end{{tikzpicture}}}}"""


def colorbar_tex(vmin, vmax):
    """Standalone horizontal colorbar carrying the shared scale."""
    ticks = ([vmin] + [t for t in range(int(np.ceil(vmin)),
                                        int(np.floor(vmax)) + 1)
                       if vmin + 0.15 < t < vmax - 0.15] + [vmax])
    tickstr = ",".join(f"{t:g}" for t in ticks)
    return rf"""    \centerline{{%
        \begin{{tikzpicture}}
            \pgfplotscolorbardrawstandalone[
                colormap/cool,
                point meta min={vmin:.3f},
                point meta max={vmax:.3f},
                colorbar horizontal,
                colorbar style={{
                    ylabel={{$\log_{{10}} f(x,\rho(x,0))$}},
                    height={{0.2cm}},
                    width={{6cm}},
                    ylabel style={{rotate=-90, font=\footnotesize}},
                    xtick={{{tickstr}}},
                    xticklabel style={{font=\footnotesize}}
                }}
            ]
        \end{{tikzpicture}}}}"""
