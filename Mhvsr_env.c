/* Mhvsr_env.c -- build the percentile envelope of an HVSR cube
 *
 * input: HVSR cube with axes (n1, n2, n3)
 *   n1 = nfreq_in    frequency bins (o1, d1 used)
 *   n2 = nrcv        number of receivers
 *   n3 = ncomp       output components from sfhvsr (typically 4)
 *
 * output: percentile envelope with axes (nfrq_out, npct)
 *   n1 = nfrq_out    frequency bins on the [fmin..fmax] grid
 *   n2 = npct        number of requested percentiles
 *
 * algorithm:
 *   1. select one component slab (default 0 = combined H/V)
 *   2. for each frequency bin, sort across receivers and
 *      compute the requested percentiles via linear interpolation
 *   3. linearly remap from the input freq axis [o1..o1+(nf_in-1)*d1]
 *      to the requested [fmin..fmax] grid with nfrq_out points
 *   4. clip the result from below at floor (avoid log(0) downstream)
 *
 * compile (cpu):
 *   cc -O2 -fopenmp -I$RSFROOT/include -L$RSFROOT/lib
 *      -o $RSFROOT/bin/sfhvsr_env Mhvsr_env.c -lrsf -lm
 */
#include <rsf.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* qsort comparator: ascending floats */
static int cmp_float_asc(const void *a, const void *b)
{
    float fa = *(const float *)a;
    float fb = *(const float *)b;
    return (fa > fb) - (fa < fb);
}

/* linear-interpolation percentile of a sorted ascending vector x[0..n-1]
 * p is in [0,1].  matches numpy default (linear).
 */
static float pct_sorted(const float *xs, int n, float p)
{
    if (n <= 0)        return 0.0f;
    if (n == 1)        return xs[0];
    if (p <= 0.0f)     return xs[0];
    if (p >= 1.0f)     return xs[n - 1];

    float pos  = p * (n - 1);
    int   i    = (int)pos;
    float frac = pos - (float)i;
    if (i + 1 >= n)    return xs[n - 1];
    return xs[i] + frac * (xs[i + 1] - xs[i]);
}

/* parse a comma-separated list of float percentiles (in 0..100).
 * returns count parsed, -1 on error.  fills *out (caller frees).
 */
static int parse_pctlist(const char *s, float **out)
{
    if (s == NULL) { *out = NULL; return 0; }

    /* count commas + 1 */
    int n = 1;
    for (const char *p = s; *p; p++) if (*p == ',') n++;

    float *arr = (float *)malloc((size_t)n * sizeof(float));
    if (!arr) return -1;

    int    count = 0;
    char  *dup   = strdup(s);
    char  *save  = NULL;
    char  *tok   = strtok_r(dup, ",", &save);
    while (tok && count < n) {
        char *end = NULL;
        float v   = (float)strtod(tok, &end);
        if (end == tok) { free(arr); free(dup); return -1; }
        if (v < 0.0f || v > 100.0f) { free(arr); free(dup); return -1; }
        arr[count++] = v;
        tok = strtok_r(NULL, ",", &save);
    }
    free(dup);
    *out = arr;
    return count;
}

int main(int argc, char *argv[])
{
    sf_file fin, fout;
    int     nf_in, nrcv, ncomp;
    int     comp, nfrq_out, verb;
    float   o1_in, d1_in;
    float   fmin, fmax, fl_user;
    float   floor_val;
    char   *pctstr;
    float  *pct;
    int     npct;

    sf_init(argc, argv);

    fin  = sf_input ("in");
    fout = sf_output("out");

    if (!sf_histint  (fin, "n1", &nf_in)) sf_error("need n1");
    if (!sf_histint  (fin, "n2", &nrcv))  sf_error("need n2");
    if (!sf_histint  (fin, "n3", &ncomp)) ncomp = 1;
    if (!sf_histfloat(fin, "o1", &o1_in)) o1_in = 0.0f;
    if (!sf_histfloat(fin, "d1", &d1_in)) sf_error("need d1");

    if (!sf_getint  ("comp", &comp))     comp     = 0;
    if (!sf_getint  ("verb", &verb))     verb     = 1;
    if (!sf_getfloat("floor", &floor_val)) floor_val = 1.0f;

    if (comp < 0 || comp >= ncomp)
        sf_error("comp=%d out of range [0,%d)", comp, ncomp);

    /* output frequency band -- default = input band */
    if (!sf_getfloat("fmin", &fmin))     fmin = o1_in;
    if (!sf_getfloat("fmax", &fmax))
        fmax = o1_in + (float)(nf_in - 1) * d1_in;
    if (!sf_getint  ("nfrq", &nfrq_out)) nfrq_out = nf_in;
    if (nfrq_out < 2) sf_error("nfrq must be >= 2");
    if (fmax <= fmin) sf_error("fmax (%g) must be > fmin (%g)", fmax, fmin);

    /* percentile list -- default p01 p05 p50 p95 p99 */
    pctstr = sf_getstring("pctlist");
    if (pctstr == NULL) pctstr = strdup("1,5,50,95,99");
    npct = parse_pctlist(pctstr, &pct);
    if (npct <= 0) sf_error("could not parse pctlist=%s", pctstr);

    if (verb) {
        sf_warning("sfhvsr_env: nf_in=%d nrcv=%d ncomp=%d", nf_in, nrcv, ncomp);
        sf_warning("  comp=%d  npct=%d  nfrq_out=%d", comp, npct, nfrq_out);
        sf_warning("  fmin=%g fmax=%g  floor=%g", fmin, fmax, floor_val);
    }

    /* read full cube and pick the requested component slab */
    size_t slab_n  = (size_t)nf_in * (size_t)nrcv;
    size_t total_n = slab_n * (size_t)ncomp;
    float *cube    = sf_floatalloc(total_n);
    sf_floatread(cube, total_n, fin);
    float *slab    = cube + (size_t)comp * slab_n;  /* (nrcv, nf_in) F-order */

    /* sorted copy: per-frequency receivers ascending. */
    /* layout: sorted[f * nrcv + i] for freq f, sorted-receiver-rank i */
    float *sorted = sf_floatalloc(slab_n);

    /* scratch buffer (one column of nrcv values) per thread */
#ifdef _OPENMP
    int nth = omp_get_max_threads();
#else
    int nth = 1;
#endif

    float *scratch = sf_floatalloc((size_t)nth * (size_t)nrcv);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int f = 0; f < nf_in; f++) {
#ifdef _OPENMP
        int tid = omp_get_thread_num();
#else
        int tid = 0;
#endif
        float *col = scratch + (size_t)tid * (size_t)nrcv;

        /* gather: slab is F-order (nf_in fastest), so element (rcv, freq) is
         * slab[rcv*nf_in + f] */
        for (int r = 0; r < nrcv; r++)
            col[r] = slab[(size_t)r * (size_t)nf_in + (size_t)f];

        qsort(col, nrcv, sizeof(float), cmp_float_asc);

        /* write into sorted in (nrcv, nf_in) F-order: sorted[r*nf_in + f] */
        for (int r = 0; r < nrcv; r++)
            sorted[(size_t)r * (size_t)nf_in + (size_t)f] = col[r];
    }

    /* compute percentiles per input frequency: pct_in[ip * nf_in + f] */
    float *pct_in = sf_floatalloc((size_t)npct * (size_t)nf_in);

    /* gather sorted column for each freq, then call pct_sorted for each ip */
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int f = 0; f < nf_in; f++) {
#ifdef _OPENMP
        int tid = omp_get_thread_num();
#else
        int tid = 0;
#endif
        float *col = scratch + (size_t)tid * (size_t)nrcv;
        for (int r = 0; r < nrcv; r++)
            col[r] = sorted[(size_t)r * (size_t)nf_in + (size_t)f];

        for (int ip = 0; ip < npct; ip++) {
            float p = pct[ip] * 0.01f;
            pct_in[(size_t)ip * (size_t)nf_in + (size_t)f] =
                pct_sorted(col, nrcv, p);
        }
    }

    free(sorted);
    free(scratch);
    free(cube);

    /* remap to [fmin..fmax] with nfrq_out points (linear interp) */
    float  d1_out = (fmax - fmin) / (float)(nfrq_out - 1);
    float *out    = sf_floatalloc((size_t)npct * (size_t)nfrq_out);

    for (int ip = 0; ip < npct; ip++) {
        for (int j = 0; j < nfrq_out; j++) {
            float fq  = fmin + (float)j * d1_out;
            float pos = (fq - o1_in) / d1_in;
            float v;

            if (pos <= 0.0f) {
                v = pct_in[(size_t)ip * (size_t)nf_in + 0];
            } else if (pos >= (float)(nf_in - 1)) {
                v = pct_in[(size_t)ip * (size_t)nf_in + (size_t)(nf_in - 1)];
            } else {
                int   i    = (int)pos;
                float frac = pos - (float)i;
                float a    = pct_in[(size_t)ip * (size_t)nf_in + (size_t)i];
                float b    = pct_in[(size_t)ip * (size_t)nf_in + (size_t)(i + 1)];
                v = a + frac * (b - a);
            }

            if (v < floor_val) v = floor_val;
            out[(size_t)ip * (size_t)nfrq_out + (size_t)j] = v;
        }
    }
    free(pct_in);

    /* setup output header: (nfrq_out, npct) */
    sf_putint   (fout, "n1",     nfrq_out);
    sf_putfloat (fout, "o1",     fmin);
    sf_putfloat (fout, "d1",     d1_out);
    sf_putstring(fout, "label1", "Frequency");
    sf_putstring(fout, "unit1",  "Hz");
    sf_putint   (fout, "n2",     npct);
    sf_putfloat (fout, "o2",     0.0f);
    sf_putfloat (fout, "d2",     1.0f);
    sf_putstring(fout, "label2", "Percentile");
    sf_putstring(fout, "unit2",  "");
    sf_putint   (fout, "n3",     1);

    sf_floatwrite(out, (size_t)npct * (size_t)nfrq_out, fout);

    free(out);
    free(pct);
    free(pctstr);

    sf_close();
    return 0;
}
