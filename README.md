# hvsr_DRHO

A portable C / Madagascar reimplementation of the multitaper H/V spectral
ratio (HVSR) workflow with cross-correlation and amplitude-ratio (CC-AR)
window weighting.

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

4. CC-AR weighting (O'Connell & Girard, 2026) combines windows: each
   window is weighted by its cross-correlation with the global ln-mean
   reference and inversely by an amplitude-ratio diagnostic, so noisy or
   unrepresentative windows contribute less to the receiver-level estimate.
5. The combined H/V is $\sqrt{(H_1/V)^2 + (H_2/V)^2}$, with ln-standard
   deviation reported as uncertainty.

See `doc/sfhvsr.md` for the full algorithm, parameter list, and examples.

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
   nwin=5 npi=3 fmin=0.1 fmax=45.0 ccweight=y
```

A full SConstruct workflow that computes HVSR and produces per-receiver
plots is provided in `SConstruct`.  Switch between CPU and GPU by setting
`par['exe'] = './sfhvsr_gpu'`.

---

## References

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
