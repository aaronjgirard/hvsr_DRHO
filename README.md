# hvsr_DRHO

A portable C / Madagascar reimplementation of the multitaper H/V spectral
ratio (HVSR) workflow with per-window STA/LTA rejection and log-mean
stacking across the kept windows.

## Attribution

This code is based on the IDL workflow released with:

> O'Connell, D. (2026). *HVSR code for 2023 Kilauea Horizontal-Component
> Signal/Noise Assessment of a Large Three-Component Nodal Array* [software].
> Zenodo. <https://doi.org/10.5281/zenodo.19929263>

The accompanying paper:

> O'Connell, D.R.H. and Girard, A.J. (2026).
> 2023 Kilauea horizontal-component signal/noise assessment of a large
> three-component nodal array. *Submitted to* **SEISMICA**.

The original Zenodo release contains the full IDL pipeline used to produce
the figures in O'Connell & Girard (2026).  This repository ports the core
HVSR computation (Stage 1 of the original workflow) to C with Madagascar
bindings, providing OpenMP (CPU) and CUDA (GPU) executables that operate on
RSF-format three-component data.

---

## What this repository provides

| component             | purpose                                                |
|-----------------------|--------------------------------------------------------|
| `Mhvsr.c`             | OpenMP CPU implementation -> `sfhvsr`                  |
| `Mhvsr_gpu.c`         | CUDA / cuFFT GPU implementation -> `sfhvsr_gpu`        |
| `Mhvsr_env.c`         | per-frequency percentile envelope helper -> `sfhvsr_env` |
| `SConstruct`          | scons build script for `sfhvsr` / `sfhvsr_gpu` / `sfhvsr_env` |
| `SConstruct_example`  | example Madagascar workflow (compute + plot per receiver) |
| `doc/sfhvsr.md`       | user manual for `sfhvsr` / `sfhvsr_gpu`                |
| `doc/oconnell_2026_workflow_report.md` | description of the original Zenodo IDL pipeline |

Both executables produce identical output.  Use `sfhvsr` on any system with
a C compiler + OpenMP; use `sfhvsr_gpu` on NVIDIA GPU nodes for large
datasets.

---

## Method (summary)

Three-component seismic data (Z, H1, H2), pre-windowed into segments of
equal length, are processed as follows:

1. Discrete prolate spheroidal sequence (DPSS / Slepian) tapers are
   generated for the chosen time-bandwidth product (Thomson, 1982).
2. For each receiver and each window, every component is multiplied by each
   taper and FFT'd to obtain amplitude spectra.
3. The per-window H/V ratio is the geometric mean of per-taper ratios:

   $$\frac{H}{V}(f) = \exp\!\left(\frac{1}{K}\sum_{k=0}^{K-1}
   \ln\frac{|H_k(f)|}{|V_k(f)|}\right)$$

4. Each window is screened by a sliding-maximum STA/LTA detector on all
   three components (default $\mathrm{STA}=1\,\mathrm{s}$,
   $\mathrm{LTA}=10\,\mathrm{s}$, threshold $2.5$).  Windows whose maximum
   STA/LTA ratio on any component exceeds the threshold -- transients,
   spikes, data gaps -- are dropped.  Receivers with fewer than `nmin`
   surviving windows (default 5) are zeroed; this catches stations with
   dead vertical channels whose few accepted windows would otherwise
   produce a spurious monotonic ramp.  This replaces the earlier CC-AR
   weighting, which was unstable for windows whose high-frequency band
   floored out (a single empty window could dominate the weighted mean and
   collapse the receiver-level estimate to zero).
5. The receiver-level $H_N/V$ and $H_E/V$ are plain log-means over the kept
   windows, log-frequency smoothed with the Konno-Ohmachi (1998) operator
   $W(f, f_c) = \left(\frac{\sin(b\,\log_{10}(f/f_c))}{b\,\log_{10}(f/f_c)}\right)^4$
   (default bandwidth $b = 40$; `kob=0` disables).  HVSR is then evaluated
   from the smoothed components.  The horizontal-combine formula is
   selectable via the `combine=` parameter (default `nakamura`); see the
   next section for the four supported modes.  Ln-standard deviation across
   the kept windows is reported as uncertainty.

See `doc/sfhvsr.md` for the full algorithm, parameter list, and examples.

---

## Horizontal-combine formula (`combine=`)

The two horizontal Fourier amplitude spectra $H_N(f)$ and $H_E(f)$ can be
combined into a single HVSR curve in several ways.  `sfhvsr` exposes four
modes; the choice changes the peak amplitude but **not** the peak frequency
$f_0$ — that is identical for all four.

| `combine=` | Formula | Notes |
|---|---|---|
| `nakamura` (default) | $\dfrac{\sqrt{H_N^2 + H_E^2}}{V}$ | Nakamura (1989).  Dominated by the larger horizontal component when the two are unequal. |
| `geomean` | $\dfrac{\sqrt{H_N\,H_E}}{V}$ | SESAME (2004) standard; used by geopsy and somar `hvsr_lite`.  Robust to channel asymmetry — if one horizontal has a different gain or different ambient-noise floor, the geometric mean masks the asymmetry. |
| `rms` | $\dfrac{\sqrt{(H_N^2 + H_E^2)/2}}{V}$ | True horizontal RMS; physically the magnitude of the horizontal motion vector divided by $\sqrt 2$.  Equals `nakamura`$/\sqrt 2$. |
| `max` | $\dfrac{\max(H_N, H_E)}{V}$ | Picks the larger channel.  Useful for highly polarized sites (basin edges, instrument tilt). |

### How the choices differ in practice

- When $H_N \approx H_E$ (most isotropic ambient-noise sites): `nakamura`
  $=\sqrt 2 \cdot$`geomean`. The two curves are a constant $\sqrt 2 \approx 1.41$
  apart at all frequencies.
- When $H_N \gg H_E$ (polarized motion): `nakamura` and `max` track the
  dominant component; `geomean` is pulled down by the quieter channel.
- For *resonance frequency* extraction, any mode works — the location of
  $f_0$ is invariant.  Only the *amplification* estimate at $f_0$ changes.

### Choosing a mode

| If you want… | use… |
|---|---|
| To reproduce Nakamura (1989) | `nakamura` |
| To compare against geopsy / SESAME-compliant published HVSR catalogs | `geomean` |
| A physically interpretable horizontal-motion RMS | `rms` |
| Worst-case polarized amplification (e.g. basin-edge sites) | `max` |

If you are comparing this code against a hvsr_lite (somar / geopsy)
reference and see a constant $\sim 1.5\times$ offset, that is the expected
signature of `nakamura` vs `geomean`; switch to `combine=geomean` for a
like-for-like comparison.

---

## Input / output

**Input.**  Three RSF files with matching dimensions:

| axis | meaning                  |
|------|--------------------------|
| `n1` | time samples per window  |
| `n2` | number of receivers      |
| `n3` | number of time windows   |

**Output.**  One RSF file with axes `(nfreq, n_receivers, 4)`, where the
fourth axis holds: combined H/V, H1/V, H2/V, and ln-standard deviation.

---

## Building

### CPU (portable)

```bash
cc -O2 -fopenmp \
   -I$RSFROOT/include -L$RSFROOT/lib \
   -o $RSFROOT/bin/sfhvsr Mhvsr.c -lrsf -lm
```

### GPU (NVIDIA CUDA toolkit required)

```bash
nvcc -O2 \
   -I$RSFROOT/include -L$RSFROOT/lib \
   -o $RSFROOT/bin/sfhvsr_gpu Mhvsr_gpu.c -lrsf -lm -lcufft
```

---

## Quick start

```bash
sfhvsr < data_z.rsf h1=data_n.rsf h2=data_e.rsf > hvsr.rsf \
   nwin=5 npi=3 fmin=0.1 fmax=45.0 \
   sta=1.0 lta=10.0 sthr=2.5 nmin=5 kob=40
```

The `ccweight=` flag is still accepted for backward compatibility but is
ignored; stacking is always a plain log-mean over windows that pass STA/LTA.

A full SConstruct workflow that computes HVSR and produces per-receiver
plots is provided in `SConstruct`.  Switch between CPU and GPU by setting
`par['exe'] = './sfhvsr_gpu'`.

---

## References

- Nakamura, Y. (1989). A method for dynamic characteristics estimation
  of subsurface using microtremor on the ground surface.  *Quarterly
  Report of the Railway Technical Research Institute*, 30(1):25-33.
  *(HVSR formula used here: $\mathrm{HVSR}(f) = \sqrt{H_N^2 + H_E^2} / V$.)*
- Konno, K. and Ohmachi, T. (1998). Ground-motion characteristics
  estimated from spectral ratio between horizontal and vertical
  components of microtremor. *Bulletin of the Seismological Society of
  America*, 88(1):228-241.
- O'Connell, D. (2026). HVSR code for 2023 Kilauea Horizontal-Component
  Signal/Noise Assessment of a Large Three-Component Nodal Array
  [software]. Zenodo. <https://doi.org/10.5281/zenodo.19929263>
- O'Connell, D.R.H. and Girard, A.J. (2026). 2023 Kilauea
  horizontal-component signal/noise assessment of a large three-component
  nodal array. Submitted to *SEISMICA*.
- Thomson, D.J. (1982). Spectrum estimation and harmonic analysis.
  *Proceedings of the IEEE*, 70:1055-1096.
- Lees, J.M. and Park, J. (1995). Multiple-taper spectral analysis: a
  stand-alone C-subroutine. *Computers & Geosciences*, 21(2):199-236.
