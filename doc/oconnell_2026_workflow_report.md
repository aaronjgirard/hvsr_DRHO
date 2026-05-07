# O'Connell & Girard (2026) HVSR Workflow -- Code Inventory and Function Report

## Reference

- **Paper:** O'Connell, D.R.H. and Girard, A.J. (2026). *2023 Kilauea
  horizontal-component signal/noise assessment of a large three-component
  nodal array.* Submitted to **SEISMICA**.
- **Code archive:** O'Connell, D. (2026). HVSR code... [software]. Zenodo.
  doi:[10.5281/zenodo.19929263](https://doi.org/10.5281/zenodo.19929263)

This report describes the contents and function of the IDL workflow scripts
distributed in `hvsr_code.tar.gz` on Zenodo.  The companion paper analyzes
1802 three-component nodes deployed during the 2023 Kilauea unrest, divided
into three sub-arrays:

- **PASSCAL SS** -- 1409 SmartSolo IGU-16HR-3C 5 Hz nodes (spiked + bucket)
- **GTI** -- 150 Geospace GS-One 2 Hz / 3C nodes (3-channel U3)
- **HVO** -- 71 SmartSolo SS 0.2 Hz 3C nodes (USGS Hawaiian Volcano Observatory)

The codes use IDL syntax; the supporting libraries `coyoteprograms.zip` and
`pp_lib-master.zip` (Coyote Graphics + Paul Pellissier's IDL utilities) are
bundled in the archive.

---

## File Inventory

### Numerical libraries (`*.pro`)

| file               | lines | purpose                                       |
|--------------------|-------|-----------------------------------------------|
| `opt_multitap.pro` | 663   | DPSS/Slepian taper generation (Thomson 1982). Contains `tridib` (eigenvalue bisection for tridiagonal matrix), `pvect` (eigenvector by inverse iteration), and `opt_multitap` (assembles tapers + bandwidth retention factors) |
| `nrattle.pro`      | 294   | 1-D SH/SV plane-wave site response code (Thomson-Haskell propagator). Used to compare H/V to predicted site amplification from a given Vs profile |
| `return_sac7.pro`  | 47    | reader for SAC v7 binary format (ground-motion time series) |
| `legend2.pro`      | 467   | plot legend helper (Coyote-style) |

### Stage 1 -- raw HVSR computation (one curve per hour, per node)

These take SAC files for each node and produce per-hour H/V CSV files.  All
follow the same algorithm but are split by sub-array and weather subset.

| file | sub-array | filter |
|------|-----------|--------|
| `calculate_hvsr_passcal_rev0.idl`        | PASSCAL SS | all hours |
| `calculate_hvsr_gti_by_low_wind_hours.idl` | GTI      | wind <= 2 knots |
| `calculate_hvsr_gti_medlow.idl`            | GTI      | wind <= 3 knots |
| `calculate_hvsr_hvonodes_rev0.idl`         | HVO nodes | all hours |
| `calculate_hvsr_hvonet_rev0.idl`           | HVO network stations | all hours |

**Inner algorithm** (identical across files, rev0 = baseline version):

1. Load weather CSV (`weatherdata_hilina_pali_road_*.csv`) and select hours
   matching the wind-speed cut.
2. Window length 6144 samples (61.44 s at 100 Hz, = 3 x 2048).
3. Generate 5 Slepian tapers via `opt_multitap` with NW = 3.
4. For each station, each selected hour, each non-overlapping 61 s window:
   - read 3-component SAC files (Z, N, E),
   - apply each taper, FFT, take amplitude spectrum,
   - geometric mean across tapers gives H1/V and H2/V,
5. Aggregate windows within an hour using a weighted ln-mean (weight from
   median 5-10 Hz / 1-2 Hz amplitude ratio raised to a power).
6. Cross-correlate each window's normalized H/V with the hourly ln-mean
   reference; CC weights re-normalize the final hourly mean.
7. Write `<station>_<yyyydoy>_<hour>_hov.csv` containing columns
   `f_hz, hvlnmean, north_hvlnmean, east_hvlnmean`.

### Stage 2 -- multi-hour stacking and weighting variants

Each variant aggregates the per-hour CSVs from Stage 1 into a single
station-level CSV with the columns `f_Hz, hv_lmean, hv_p1s, hv_m1s,
north_hv_lmean, ...`, plus the +/- 1 sigma envelopes.  Variants differ in
which weighting recipe is used.

| file | weighting scheme |
|------|------------------|
| `calculate_hvsr_passcal_by_low_wind_hours_lnmean_cc_scaling.idl`   | low-wind hours, ln-mean CC weight (paper's preferred PASSCAL method) |
| `calculate_hvsr_passcal_by_multi_wind_speed_hours_lnmean_cc_scaling.idl` | repeats above for 3 wind-speed bins (low / median / high) |
| `calculate_hvsr_gti_by_low_wind_hours_lnmean_cc_scaling.idl`       | low-wind hours, ln-mean CC weight (GTI) |
| `calculate_hvsr_gti_by_low_wind_hours_var_scaling.idl`             | low-wind hours, weighted by 1/variance |
| `calculate_hvsr_gti_by_low_wind_hours_5-10Hz_amp_scaling.idl`      | weighted by inverse 5-10 Hz amplitude ratio (down-weights wind-dominated hours) |
| `calculate_hvsr_gti_by_low_wind_hours_5-18Hz_amp_scaling.idl`      | same, but 5-18 Hz band (covers GTI flat-response upper end) |
| `calculate_hvsr_gti_by_multi_wind_speed_hours_lnmean_cc_scaling.idl` | 3-wind-speed-bin version of the GTI ln-mean CC scheme |
| `calculate_hvsr_hvonodes_by_low_wind_hours_lnmean_cc_scaling.idl`  | HVO nodes, low-wind ln-mean CC |
| `calculate_hvsr_hvonode_by_multi_wind_speed_hours_lnmean_cc_scaling.idl` | HVO nodes, 3-wind-speed bins |

The "lnmean_cc" variants produced the H/V curves shown in the paper.  The
amp-scaling and var-scaling files are alternative weighting strategies the
authors evaluated; output filenames (e.g.,
`*_lower_horizontal_noise_*.csv`) propagate into the next stage.

These scripts also run an interactive picking loop where a user clicks on
the H/V plot to mark `MIN_USABLE_FREQ` and `MAX_USABLE_FREQ` per station --
the bandwidth over which that node's H/V is judged reliable for downstream
use.  Picks are written to CSVs such as
`3_wind_speed_refined_passcal_nodes_h_over_v_w_station_info.csv`.

### Stage 3 -- station classification and selection

| file | function |
|------|----------|
| `find_decoupled_passcal_nodes.idl`  | identifies SmartSolo spiked nodes whose horizontals are decoupled from the ground (no useful H/V); writes `decoupled_passcal_node_list.csv` (15 nodes) |
| `expand_passcal_nodes_low_enough_noise_for_interferometry.idl` | sweeps the picking interactively across remaining nodes to extend the "interferometry-grade" subset; produces the lists named below |
| `passcal_compare_bucket_not_node_hvsr_rev2.idl` | compares spiked vs. bucket-deployed PASSCAL nodes; quantifies wind-speed sensitivity differences |
| `gti_node_hvo_networkd_passcal_compare_bucket_not_node_hvsr.idl` | cross-comparison among GTI nodes, HVO network broadbands, and PASSCAL bucket nodes that are co-located |

These scripts produce the four station-subset CSVs that are the practical
deliverable of the paper:

- `*_most_possibly_useable_horizontal_station_list_to_1p2Hz.csv` (1099 nodes)
- `*_most_possibly_useable_horizontal_station_list_to_10Hz.csv` (722 nodes)
- `*_lower_horizontal_noise_station_list.csv` (610 nodes)
- `*_lowest_horizontal_noise_station_list.csv` (486 nodes)

### Stage 4 -- figure generation for the manuscript

| file | figure(s) produced |
|------|--------------------|
| `plot_seismica_figure_7_all_normalized_ln_mean.idl`                  | Figure 7 -- ln-mean normalized H/V over all qualifying nodes |
| `plot_seismica_figure_7_all_normalized_ln_mean_rev2_most.idl`        | revised Figure 7 with broader station inclusion |
| `plot_seismica_figure_7_all_normalized_ln_mean_rev2_most_interferometry.idl` | as above, restricted to interferometry-grade subset |
| `plot_summary_stacks_results_for_all_nodes_rev2.idl`                 | summary panel comparing PASSCAL, GTI, and HVO ln-mean H/V stacks |
| `plot_passcal_cases_for_supplement.idl`                              | supplementary Figures S1-S5 (per-deployment H/V case studies) |
| `plot_passcal_spiked_nodes_h_over_d_density.idl`                     | density plot of spiked-node H/V vs. spike-resonance frequency (uses `all_passcal_spiked_node_all_hour_hv_lmean.idlsave`) |
| `plot_lower_lowest_node_locations_on_map.idl`                        | map of "lower" + "lowest" horizontal-noise PASSCAL nodes |
| `plot_most_lowest_node_locations_on_map.idl`                        | map of "most-useable" + "lowest" subsets |

All map plots use `ge_hvsr_basemap_grayscale.png` as a Google Earth backdrop
and the bounding box from `plot_latlon_for_GE.csv`.

### Supporting CSVs included in the archive

- **Weather:** `weatherdata_hilina_pali_road_*.csv` -- Hilina Pali Road weather
  station record used to bin hours by wind speed.
- **Station metadata:** `kilauea_3d_all_15{0,2}_3C_gti_stations_*.csv`,
  `hvo_nodes_*.csv`, `smartsolo_robust_digisolo_log_locations_*.csv`,
  `decoupled_passcal_node_list.csv`, `gti_dead_channel_list_by_station.csv`.
- **Modal-frequency tables:** `*_low_freq_modal_frequency.csv`.
- **Frequency-band station lists:**
  `0p1-30Hz_band_GTI_nodes_for_HVSR_etc.csv`,
  `decent_0p2-{1,10}Hz_band_PASSCAL_nodes_for_HVSR_etc.csv`,
  `includes_most_possibly_useable_horizontal_passcal_nodes.csv`.
- **Pre-computed stacks:**
  `PASSCAL_HVSR_ln_stack_from_*.csv`,
  `GTI_HVSR_ln_stack_by_station_bandwidth.csv`,
  `hvo_net_ln_mean_h_over_v.csv`.
- **Tomography / picking inputs:** `elevation_corrected_stations_for_tomo.csv`,
  `updated_receiver_locations_for_t-rex_eq_picks.csv`.

---

## End-to-end Pipeline (graphical summary)

```
  raw 3C SAC files (per node, per hour, 100 Hz)
                │
                ▼
  ┌─────────────────────────────────────────────┐
  │ Stage 1: calculate_hvsr_<array>_*.idl       │
  │  - bin hours by wind speed                  │
  │  - 6144-sample windows, 5 Slepian tapers    │
  │  - geometric mean H/V per window            │
  │  - CC + amp-ratio weighted mean per hour    │
  └─────────────────────────────────────────────┘
                │
                ▼
   per-hour CSVs:  <station>_<doy>_<hh>_hov.csv
                │
                ▼
  ┌─────────────────────────────────────────────┐
  │ Stage 2: *_lnmean_cc_scaling.idl variants   │
  │  - aggregate hours into station mean        │
  │  - compute +/- 1 sigma                      │
  │  - INTERACTIVE: pick fmin/fmax per station  │
  └─────────────────────────────────────────────┘
                │
                ▼
   per-station CSVs (logmean/<station>_logmean.csv)
   + refined node tables with usable bandwidths
                │
                ▼
  ┌─────────────────────────────────────────────┐
  │ Stage 3: classification scripts             │
  │  - find decoupled nodes                     │
  │  - sort into 4 quality tiers                │
  └─────────────────────────────────────────────┘
                │
                ▼
   four station-subset CSVs (the paper's product)
                │
                ▼
  ┌─────────────────────────────────────────────┐
  │ Stage 4: plot_*.idl                         │
  │  - ln-mean stack figures (7, S1-S19)        │
  │  - station maps                             │
  │  - density + comparison plots               │
  └─────────────────────────────────────────────┘
```

---

## Notes for a Reader Working from This Code

- All scripts assume IDL 8.x with the bundled Coyote / pp_lib utilities on
  the IDL path.  Scripts use `read_csv_pp` (from pp_lib) for CSV I/O and
  `fsc_color`, `loadct`, `cgImage` style calls from Coyote.
- Filesystem paths are hard-coded for the original author's environment
  (`/mdata/geomagic/...`, `/home/geomagic/...`).  Reproducing the workflow
  on another system requires editing those paths.
- Scripts are organized as IDL "main" files (`.idl`), expected to be
  executed top-to-bottom in an interactive IDL session.  Many contain
  `cursor, /down` calls that block waiting for mouse input -- these are the
  manual fmin/fmax pick steps.
- Several files differ only in their wind-speed cut (`scut=1`, `scut=2`,
  `scut=3` knots) or weighting exponent.  Treat them as parameter sweeps
  rather than independent algorithms.
- The `_rev0` files are the original raw HVSR computations.  The
  `_lnmean_cc_scaling` files are the final paper versions.  Other scaling
  variants (`var_scaling`, `5-10Hz_amp_scaling`, `5-18Hz_amp_scaling`)
  represent alternatives that were evaluated but not used in the published
  figures.

---

## How This Maps to the Madagascar Reimplementation

The project directory (`/Users/agirard/HVSR/`) contains a from-scratch
Madagascar/C implementation of **Stage 1** only: `Mhvsr.c` / `sfhvsr` (CPU)
and `Mhvsr_gpu.c` / `sfhvsr_gpu` (CUDA).  These programs reproduce the core
algorithm -- DPSS taper generation, multitaper FFT, geometric-mean H/V,
CC-AR window weighting -- but operate on RSF inputs (`time x receivers x
windows`) and output a single RSF cube of H/V curves.  See
`doc/sfhvsr.md` for the user manual.

Stages 2, 3, and 4 of the original IDL workflow are not yet ported.  They
are largely interactive (manual frequency picking, manual station
classification) and tied to specific output CSVs and figure formats.
