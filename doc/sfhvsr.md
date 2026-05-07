# sfhvsr / sfhvsr\_gpu -- multitaper H/V spectral ratio (HVSR)

## Description

`sfhvsr` computes horizontal-to-vertical spectral ratios (H/V or HVSR) from
three-component seismic data using multitaper spectral analysis with discrete
prolate spheroidal sequence (DPSS/Slepian) tapers.  The method follows
O'Connell & Girard (2026), which uses cross-correlation (CC) and
amplitude-ratio (AR) weighting to produce robust, frequency-dependent H/V
estimates.

H/V spectral ratios quantify the ratio of horizontal to vertical ground motion
as a function of frequency.  An H/V value near 1 indicates similar horizontal
and vertical amplitudes; values significantly above 1 indicate elevated
horizontal energy (from site resonance, source effects, or noise).
H/V is used for:

- site-response estimation (peak frequency ~ fundamental resonance)
- horizontal-component signal-to-noise assessment
- station quality control for S-wave picking, moment tensor analysis,
  interferometry, and tomography

Two executables are provided:

| source          | executable      | parallelism                        |
|-----------------|-----------------|------------------------------------|
| `Mhvsr.c`       | `sfhvsr`        | OpenMP over receivers (CPU)        |
| `Mhvsr_gpu.c`   | `sfhvsr_gpu`    | batched cuFFT on GPU (CUDA)        |

Both produce identical output.  Use `sfhvsr` on any system; use `sfhvsr_gpu`
on NVIDIA GPU nodes for large datasets.

---

## Algorithm

1. **DPSS taper generation** -- compute `nwin` Slepian tapers of length `n1`
   with time-bandwidth product `npi` (Thomson, 1982).

2. **Per-window multitaper spectra** -- for each time window (axis 3) and each
   receiver (axis 2), multiply each component (Z, H1, H2) by each taper and
   compute the FFT.  The amplitude spectrum
   $|X_k(f)| = \sqrt{\mathrm{Re}^2 + \mathrm{Im}^2}$ is retained.

3. **H/V ratio** -- for each frequency bin:

$$\frac{H}{V}(f) = \exp\!\left(\frac{1}{K}\sum_{k=0}^{K-1}
\ln\frac{|H_k(f)|}{|V_k(f)|}\right)$$

   where $K$ = `nwin`.  This is the geometric mean of the per-taper ratios.

4. **CC-AR weighting across windows** (enabled by `ccweight=y`):
   - compute the raw ln-mean H/V across all windows
   - normalize each window's H/V at a reference frequency
   - cross-correlate each window with the reference (Pearson $r$)
   - compute an amplitude ratio AR = median(5--10 Hz) / median(1--2 Hz)
   - final weight = CC / AR$^2$, normalized to sum to 1
   - output = weighted ln-mean across windows

5. **Combined H/V** -- $\sqrt{(H_1/V)^2 + (H_2/V)^2}$

6. **Uncertainty** -- ln-standard deviation of per-window combined H/V

---

## Input

Three RSF files with matching dimensions:

| axis | meaning                  |
|------|--------------------------|
| `n1` | time samples per window  |
| `n2` | number of receivers      |
| `n3` | number of time windows   |

`d1` = sample interval in seconds (e.g., 0.01 for 100 Hz).

Data must be pre-windowed (non-overlapping segments of equal length).

---

## Output

One RSF file:

| axis | meaning                  | size        |
|------|--------------------------|-------------|
| `n1` | frequency bins           | `nfreq`     |
| `n2` | receivers (from input)   | same as input `n2` |
| `n3` | output component         | 4           |

The four output components (`n3` axis):

| `f3` | content                                |
|------|----------------------------------------|
| 0    | combined H/V = $\sqrt{H_1/V^2 + H_2/V^2}$ |
| 1    | H1/V (first horizontal / vertical)    |
| 2    | H2/V (second horizontal / vertical)   |
| 3    | ln-standard deviation of combined H/V  |

The frequency axis is labeled in Hz with `o1 = fmin` and `d1 = df`.

---

## Parameters

| parameter    | type  | default | description                              |
|--------------|-------|---------|------------------------------------------|
| `nwin`       | int   | 5       | number of Slepian tapers                 |
| `npi`        | int   | 3       | time-bandwidth product (NW)              |
| `fmin`       | float | 0.1     | minimum output frequency [Hz]            |
| `fmax`       | float | 45.0    | maximum output frequency [Hz]            |
| `ccweight`   | bool  | y       | enable CC-AR weighting across windows    |
| `verb`       | int   | 1       | verbosity (0=silent, 1=progress bar)     |

The input files are specified as:

- `in` (stdin) = vertical component (Z)
- `h1` = first horizontal component (N or 1)
- `h2` = second horizontal component (E or 2)

---

## Compilation

### CPU version (portable)

```bash
cc -O2 -fopenmp \
   -I$RSFROOT/include -L$RSFROOT/lib \
   -o $RSFROOT/bin/sfhvsr Mhvsr.c -lrsf -lm
```

### GPU version (requires NVIDIA CUDA toolkit)

```bash
nvcc -O2 \
   -I$RSFROOT/include -L$RSFROOT/lib \
   -o $RSFROOT/bin/sfhvsr_gpu Mhvsr_gpu.c -lrsf -lm -lcufft
```

---

## Examples

### 1. basic HVSR computation

```bash
sfhvsr < data_z.rsf h1=data_n.rsf h2=data_e.rsf > hvsr.rsf \
   nwin=5 npi=3 fmin=0.1 fmax=45.0
```

### 2. without CC-AR weighting (uniform ln-mean)

```bash
sfhvsr < data_z.rsf h1=data_n.rsf h2=data_e.rsf > hvsr.rsf \
   ccweight=n
```

### 3. GPU version (same interface)

```bash
sfhvsr_gpu < data_z.rsf h1=data_n.rsf h2=data_e.rsf > hvsr.rsf \
   nwin=5 npi=3
```

### 4. extract and plot combined H/V for all receivers

```bash
sfwindow < hvsr.rsf n3=1 f3=0 > hv_combined.rsf
sfgraph < hv_combined.rsf \
   title="Combined H/V" label1="Frequency" unit1="Hz" label2="H/V"
```

### 5. extract single receiver and plot with +/- 1 sigma

```bash
sfwindow < hvsr.rsf n3=1 f3=0 n2=1 f2=42 > hv_r42.rsf
sfwindow < hvsr.rsf n3=1 f3=3 n2=1 f2=42 > std_r42.rsf
sfmath < hv_r42.rsf c=std_r42.rsf output="input*exp(c)"  > upper.rsf
sfmath < hv_r42.rsf c=std_r42.rsf output="input*exp(-c)" > lower.rsf
```

### 6. SConstruct workflow

```python
from rsf.proj import *

par = {
    'nwin'   : 5,
    'npi'    : 3,
    'fmin'   : 0.1,
    'fmax'   : 45.0,
    'ccwt'   : 'y',
    'zfile'  : 'data_z',
    'h1file' : 'data_n',
    'h2file' : 'data_e',
    'exe'    : './sfhvsr',     # or './sfhvsr_gpu'
}

Flow('hvsr', [par['zfile'], par['h1file'], par['h2file']], '''
     %(exe)s h1=${SOURCES[1]} h2=${SOURCES[2]}
     nwin=%(nwin)d npi=%(npi)d
     fmin=%(fmin)g fmax=%(fmax)g
     ccweight=%(ccwt)s verb=1
     ''' % par)

for ic, name in enumerate(['combined', 'h1v', 'h2v', 'stddev']):
    Flow(name, 'hvsr', 'window n3=1 f3=%d' % ic)

Result('hvsr_all', 'combined', '''
       graph title="Combined H/V"
       label1="Frequency" unit1="Hz" label2="H/V"
       ''')

End()
```

Run with `scons` or `scons Fig/hvsr_all.vpl`.

---

## Verification

For white-noise input (all three components independent), the expected output
is:

- H1/V $\approx$ 1.0 and H2/V $\approx$ 1.0
- combined H/V $\approx \sqrt{2} \approx$ 1.414
- ln-stddev $\approx$ 0 (all windows similar)

For real data, look for site-response peaks (H/V maxima at the fundamental
resonance frequency).  Site-specific noise sources, sensor coupling artifacts,
and instrument-specific resonances may also appear and should be evaluated
relative to the goals of the analysis.

---

## References

O'Connell, D.R.H. and Girard, A.J. (2026).
2023 Kilauea horizontal-component signal/noise assessment of a large
three-component nodal array.
*Non-peer reviewed research article submitted to* **SEISMICA**.

O'Connell, D. (2026).
HVSR code for 2023 Kilauea horizontal-component signal/noise assessment of a
large three-component nodal array [software].
Zenodo. [doi:10.5281/zenodo.19929263](https://doi.org/10.5281/zenodo.19929263)

Thomson, D.J. (1982).
Spectrum estimation and harmonic analysis.
*Proceedings of the IEEE*, 70:1055--1096.

Lees, J.M. and Park, J. (1995).
Multiple-taper spectral analysis: a stand-alone C-subroutine.
*Computers & Geosciences*, 21(2):199--236.
