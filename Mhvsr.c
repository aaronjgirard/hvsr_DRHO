/* Mhvsr.c -- multitaper H/V spectral ratio (HVSR) computation
 *
 * reads 3 RSF files (Z, H1, H2) with axes n1 x n2 x n3
 *   n1 = time samples per window
 *   n2 = receivers
 *   n3 = time windows
 *
 * outputs RSF file with axes nfreq x n2 x 4
 *   n3=0: combined H/V = sqrt(H1/V^2 + H2/V^2)
 *   n3=1: H1/V
 *   n3=2: H2/V
 *   n3=3: ln-stddev of combined H/V
 *
 * algorithm: O'Connell & Girard (2026) -- multitaper HVSR with
 *   cross-correlation and amplitude-ratio weighting
 *
 * parallelism: OpenMP over receivers (n2 axis)
 *
 * compile: cc -O2 -fopenmp -I$RSFROOT/include -L$RSFROOT/lib
 *          -o $RSFROOT/bin/sfhvsr Mhvsr.c -lrsf -lm
 *
 * see also: Mhvsr_gpu.c for CUDA-accelerated version (sfhvsr_gpu)
 */
#include <rsf.h>
#include <math.h>
#include <string.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* ================================================================== */
/*                    DPSS taper generation                           */
/* ================================================================== */

/* sturm_count: count eigenvalues of tridiagonal matrix <= x
 * d[n] = diagonal, e[n] = subdiagonal (e[0] unused)
 */
static int sturm_count(int n, double *d, double *e, double x)
{
    int    cnt = 0, i;
    double q   = d[0] - x;
    if (q < 0.0) cnt++;
    for (i = 1; i < n; i++) {
        if (q == 0.0) q = 1e-30;
        q = d[i] - x - e[i] * e[i] / q;
        if (q < 0.0) cnt++;
    }
    return cnt;
}

/* bisect_eig: find k-th smallest eigenvalue by bisection (0-based k) */
static double bisect_eig(int n, double *d, double *e, int k,
                         double lo, double hi)
{
    int    iter;
    double mid;
    for (iter = 0; iter < 200; iter++) {
        mid = 0.5 * (lo + hi);
        if (hi - lo < 1e-14 * (fabs(lo) + fabs(hi) + 1e-30)) break;
        if (sturm_count(n, d, e, mid) > k) hi = mid;
        else                                lo = mid;
    }
    return 0.5 * (lo + hi);
}

/* inv_iter: compute eigenvector of tridiagonal matrix for given eigenvalue
 * d[n] = diagonal, e[n] = subdiagonal, ev = eigenvalue, x[n] = output
 */
static void inv_iter(int n, double *d, double *e, double ev, double *x)
{
    int    i, iter;
    double *dl, *dd, *du, *b, nrm;

    dl = (double *)malloc(n * sizeof(double));
    dd = (double *)malloc(n * sizeof(double));
    du = (double *)malloc(n * sizeof(double));
    b  = (double *)malloc(n * sizeof(double));

    /* (T - ev*I) factored via LU with partial pivoting */
    for (i = 0; i < n; i++) dd[i] = d[i] - ev;
    for (i = 1; i < n; i++) dl[i] = e[i];
    for (i = 0; i < n - 1; i++) du[i] = e[i + 1];

    /* simple tridiagonal LU (no pivoting -- sufficient for DPSS) */
    for (i = 0; i < n - 1; i++) {
        if (fabs(dd[i]) < 1e-30) dd[i] = 1e-30;
        double m = dl[i + 1] / dd[i];
        dl[i + 1] = m;
        dd[i + 1] -= m * du[i];
    }
    if (fabs(dd[n - 1]) < 1e-30) dd[n - 1] = 1e-30;

    /* inverse iteration: start with random vector */
    for (i = 0; i < n; i++)
        x[i] = ((i * 7 + 3) % 17) / 17.0 - 0.5;

    for (iter = 0; iter < 5; iter++) {
        /* solve L*U*b = x */
        memcpy(b, x, n * sizeof(double));
        /* forward (L) */
        for (i = 1; i < n; i++)
            b[i] -= dl[i] * b[i - 1];
        /* backward (U) */
        b[n - 1] /= dd[n - 1];
        for (i = n - 2; i >= 0; i--)
            b[i] = (b[i] - du[i] * b[i + 1]) / dd[i];

        /* normalize */
        nrm = 0.0;
        for (i = 0; i < n; i++) nrm += b[i] * b[i];
        nrm = sqrt(nrm);
        if (nrm < 1e-30) nrm = 1e-30;
        for (i = 0; i < n; i++) x[i] = b[i] / nrm;
    }

    free(dl); free(dd); free(du); free(b);
}

/* dpss_tapers: compute discrete prolate spheroidal sequences
 *
 * n1     : number of samples
 * nwin   : number of tapers
 * npi    : time-bandwidth product (NW)
 * tapers : output [nwin * n1] (row-major: taper k starts at k*n1)
 * lam    : output eigenvalues [nwin]
 */
static void dpss_tapers(int n1, int nwin, int npi, double *tapers, double *lam)
{
    double  ww, cs, an, anpi;
    double *diag, *sub;
    double  dfac, drat, gamma;
    double *ell;
    int     i, k;

    an   = (double)n1;
    anpi = (double)npi;
    ww   = anpi / an;
    cs   = cos(2.0 * M_PI * ww);

    diag = (double *)malloc(n1 * sizeof(double));
    sub  = (double *)calloc(n1, sizeof(double));
    ell  = (double *)malloc(nwin * sizeof(double));

    /* tridiagonal matrix (Thomson 1982 eq 2.3) */
    for (i = 0; i < n1; i++)
        diag[i] = cs * ((an - 1.0) * 0.5 - (double)i) *
                       ((an - 1.0) * 0.5 - (double)i);
    for (i = 1; i < n1; i++)
        sub[i] = (double)(i) * (an - (double)i) * 0.5;

    /* gerschgorin bounds for eigenvalue range */
    double glo = diag[0], ghi = diag[0];
    for (i = 0; i < n1; i++) {
        double el = (i > 0) ? fabs(sub[i]) : 0.0;
        double er = (i < n1 - 1) ? fabs(sub[i + 1]) : 0.0;
        double lo = diag[i] - el - er;
        double hi = diag[i] + el + er;
        if (lo < glo) glo = lo;
        if (hi > ghi) ghi = hi;
    }

    /* find largest nwin eigenvalues (they are the ones we want) */
    /* eigenvalues are ordered ascending; we want the top nwin */
    double *eigvals = (double *)malloc(nwin * sizeof(double));
    for (k = 0; k < nwin; k++)
        eigvals[k] = bisect_eig(n1, diag, sub, n1 - 1 - k, glo, ghi);

    /* sort ascending (reverse of what we found) */
    for (i = 0; i < nwin / 2; i++) {
        double tmp = eigvals[i];
        eigvals[i] = eigvals[nwin - 1 - i];
        eigvals[nwin - 1 - i] = tmp;
    }

    /* compute eigenvectors via inverse iteration */
    for (k = 0; k < nwin; k++) {
        inv_iter(n1, diag, sub, eigvals[k], tapers + k * n1);

        /* enforce sign convention: positive integral for even tapers,
         * positive start for odd tapers */
        double sum = 0.0;
        for (i = 0; i < n1; i++) sum += tapers[k * n1 + i];
        if ((k % 2 == 0 && sum < 0.0) || (k % 2 == 1 && tapers[k * n1] < 0.0))
            for (i = 0; i < n1; i++) tapers[k * n1 + i] = -tapers[k * n1 + i];
    }

    /* bandwidth retention factors (Thomson 1982 / Slepian 1978) */
    dfac = an * M_PI * ww;
    drat = 8.0 * dfac;
    dfac = 4.0 * sqrt(M_PI * dfac) * exp(-2.0 * dfac);
    for (k = 0; k < nwin; k++) {
        lam[k] = 1.0 - dfac;
        dfac   = dfac * drat / (double)(k + 1);
    }
    gamma = log(8.0 * an * sin(2.0 * M_PI * ww)) + 0.5772156649;
    for (k = 0; k < nwin; k++) {
        ell[k] = 1.0 / (1.0 + exp(M_PI *
                 (-2.0 * M_PI * (an * ww - (double)k * 0.5 - 0.25) / gamma)));
        if (ell[k] > lam[k]) lam[k] = ell[k];
    }

    /* normalize: unit energy (sum of squares = 1), matching scipy convention */
    for (k = 0; k < nwin; k++) {
        int    kk = k * n1;
        double sq = 0.0, aa;
        for (i = 0; i < n1; i++)
            sq += tapers[kk + i] * tapers[kk + i];
        aa = 1.0 / sqrt(sq);
        for (i = 0; i < n1; i++) tapers[kk + i] *= aa;
    }

    free(diag); free(sub); free(ell); free(eigvals);
}

/* ================================================================== */
/*                    helper: compare for qsort                       */
/* ================================================================== */
static int cmp_float(const void *a, const void *b)
{
    float fa = *(const float *)a;
    float fb = *(const float *)b;
    return (fa > fb) - (fa < fb);
}

static float median_float(float *arr, int n)
{
    float *tmp = (float *)malloc(n * sizeof(float));
    memcpy(tmp, arr, n * sizeof(float));
    qsort(tmp, n, sizeof(float), cmp_float);
    float med = (n % 2) ? tmp[n / 2] : 0.5f * (tmp[n / 2 - 1] + tmp[n / 2]);
    free(tmp);
    return med;
}

/* ================================================================== */
/*                    progress bar                                    */
/* ================================================================== */
static void show_progress(int cur, int tot, time_t t0)
{
    int    pct = (int)(100.0 * cur / tot);
    int    bar = pct / 2;
    double sec = difftime(time(NULL), t0);
    int    i;
    fprintf(stderr, "\r  [");
    for (i = 0; i < 50; i++) fprintf(stderr, i < bar ? "#" : "-");
    fprintf(stderr, "] %3d%% (%d/%d) %.0fs", pct, cur, tot, sec);
    if (cur == tot) fprintf(stderr, "\n");
    fflush(stderr);
}

/* ================================================================== */
/*                    main                                            */
/* ================================================================== */
int main(int argc, char *argv[])
{
    sf_file  fz, fh1, fh2, fout;
    int      n1, n2, n3, nwin, npi, verb;
    float    d1, fmin, fmax;
    int      nf, nfreq, ilo, ihi;
    float    df;
    double  *tapers, *lam;
    float   *outvol;
    time_t   t0;

    sf_init(argc, argv);

    /* open inputs */
    fz  = sf_input("in");
    fh1 = sf_input("h1");
    fh2 = sf_input("h2");
    fout = sf_output("out");

    /* read axes from vertical component */
    if (!sf_histint(fz, "n1", &n1)) sf_error("need n1");
    if (!sf_histint(fz, "n2", &n2)) sf_error("need n2");
    if (!sf_histint(fz, "n3", &n3)) n3 = 1;
    if (!sf_histfloat(fz, "d1", &d1)) sf_error("need d1");

    /* parameters */
    bool ccweight;
    if (!sf_getint("nwin", &nwin)) nwin = 5;
    if (!sf_getint("npi",  &npi))  npi  = 3;
    if (!sf_getfloat("fmin", &fmin)) fmin = 0.1f;
    if (!sf_getfloat("fmax", &fmax)) fmax = 45.0f;
    if (!sf_getint("verb", &verb)) verb = 1;
    if (!sf_getbool("ccweight", &ccweight)) ccweight = true;

    /* frequency axis */
    nf = n1 / 2 + 1;
    df = 1.0f / (n1 * d1);
    ilo = (int)(fmin / df);
    ihi = (int)(fmax / df);
    if (ilo < 0)    ilo = 0;
    if (ihi >= nf)  ihi = nf - 1;
    nfreq = ihi - ilo + 1;

    if (verb) {
        sf_warning("sfhvsr: n1=%d n2=%d n3=%d d1=%g", n1, n2, n3, d1);
        sf_warning("  nwin=%d npi=%d fmin=%g fmax=%g", nwin, npi, fmin, fmax);
        sf_warning("  nf=%d ilo=%d ihi=%d nfreq=%d df=%g", nf, ilo, ihi, nfreq, df);
    }

    /* setup output: nfreq x n2 x 4 */
    sf_putint(fout, "n1", nfreq);
    sf_putfloat(fout, "d1", df);
    sf_putfloat(fout, "o1", ilo * df);
    sf_putstring(fout, "label1", "Frequency");
    sf_putstring(fout, "unit1", "Hz");
    sf_putint(fout, "n2", n2);
    sf_putint(fout, "n3", 4);
    sf_putstring(fout, "label3", "Component");

    /* compute DPSS tapers */
    tapers = (double *)malloc(nwin * n1 * sizeof(double));
    lam    = (double *)malloc(nwin * sizeof(double));
    dpss_tapers(n1, nwin, npi, tapers, lam);
    if (verb) sf_warning("  dpss tapers computed, lam[0]=%g", lam[0]);

    /* read all data into memory: 3 components x n2 x n3 x n1 */
    size_t trcsz  = (size_t)n1;
    size_t slicesz = (size_t)n1 * n2 * n3;
    float *dz  = sf_floatalloc(slicesz);
    float *dh1 = sf_floatalloc(slicesz);
    float *dh2 = sf_floatalloc(slicesz);

    sf_floatread(dz,  slicesz, fz);
    sf_floatread(dh1, slicesz, fh1);
    sf_floatread(dh2, slicesz, fh2);

    if (verb) sf_warning("  data read: %.1f MB per component",
                         slicesz * 4.0 / 1048576.0);

    /* allocate output: 4 x n2 x nfreq */
    outvol = sf_floatalloc((size_t)4 * n2 * nfreq);
    memset(outvol, 0, (size_t)4 * n2 * nfreq * sizeof(float));

    /* cast tapers to float for FFT */
    float *ftap = sf_floatalloc(nwin * n1);
    for (int i = 0; i < nwin * n1; i++) ftap[i] = (float)tapers[i];

    t0 = time(NULL);

    /* ============================================================== */
    /*  main loop: OpenMP parallel over receivers                    */
    /* ============================================================== */
#ifdef _OPENMP
    #pragma omp parallel
    {
        if (omp_get_thread_num() == 0 && verb)
            sf_warning("  using %d OpenMP threads", omp_get_num_threads());
    }
#endif

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
#endif
    for (int ir = 0; ir < n2; ir++) {

        /* thread-local FFT config */
        kiss_fftr_cfg fft_cfg = kiss_fftr_alloc(n1, 0, NULL, NULL);
        float        *tbuf   = (float *)malloc(n1 * sizeof(float));
        kiss_fft_cpx *fbuf   = (kiss_fft_cpx *)malloc(nf * sizeof(kiss_fft_cpx));

        /* per-taper amplitude spectra: [nwin][nf] x 3 components */
        float *spec_z  = (float *)malloc(nwin * nf * sizeof(float));
        float *spec_h1 = (float *)malloc(nwin * nf * sizeof(float));
        float *spec_h2 = (float *)malloc(nwin * nf * sizeof(float));

        /* per-window H/V: [n3][nfreq][2] (h1/v, h2/v) */
        float *hov = (float *)calloc((size_t)n3 * nfreq * 2, sizeof(float));

        int iw, it, ik, ic;

        for (iw = 0; iw < n3; iw++) {
            /* data index: n1 * (ir + n2 * iw) */
            size_t idx = (size_t)n1 * ((size_t)ir + (size_t)n2 * (size_t)iw);
            float *wz  = dz  + idx;
            float *wh1 = dh1 + idx;
            float *wh2 = dh2 + idx;

            /* compute multitaper spectra for each component */
            for (it = 0; it < nwin; it++) {
                float *tap = ftap + it * n1;

                /* Z */
                for (ik = 0; ik < n1; ik++) tbuf[ik] = wz[ik] * tap[ik];
                kiss_fftr(fft_cfg, tbuf, fbuf);
                for (ik = 0; ik < nf; ik++)
                    spec_z[it * nf + ik] = sqrtf(fbuf[ik].r * fbuf[ik].r +
                                                  fbuf[ik].i * fbuf[ik].i);

                /* H1 */
                for (ik = 0; ik < n1; ik++) tbuf[ik] = wh1[ik] * tap[ik];
                kiss_fftr(fft_cfg, tbuf, fbuf);
                for (ik = 0; ik < nf; ik++)
                    spec_h1[it * nf + ik] = sqrtf(fbuf[ik].r * fbuf[ik].r +
                                                   fbuf[ik].i * fbuf[ik].i);

                /* H2 */
                for (ik = 0; ik < n1; ik++) tbuf[ik] = wh2[ik] * tap[ik];
                kiss_fftr(fft_cfg, tbuf, fbuf);
                for (ik = 0; ik < nf; ik++)
                    spec_h2[it * nf + ik] = sqrtf(fbuf[ik].r * fbuf[ik].r +
                                                   fbuf[ik].i * fbuf[ik].i);
            }

            /* H/V ratios: exp(mean(ln(H/V))) across tapers */
            for (ic = 0; ic < nfreq; ic++) {
                int    fi = ic + ilo;
                double lnsum1 = 0.0, lnsum2 = 0.0;
                for (it = 0; it < nwin; it++) {
                    float vz = spec_z[it * nf + fi];
                    if (vz < 1e-30f) vz = 1e-30f;
                    lnsum1 += log((double)(spec_h1[it * nf + fi] / vz));
                    lnsum2 += log((double)(spec_h2[it * nf + fi] / vz));
                }
                hov[(iw * nfreq + ic) * 2 + 0] = (float)exp(lnsum1 / nwin);
                hov[(iw * nfreq + ic) * 2 + 1] = (float)exp(lnsum2 / nwin);
            }
        } /* end windows */

        /* ---------------------------------------------------------- */
        /*  window weighting and stacking                             */
        /* ---------------------------------------------------------- */
        float *hovcc1  = (float *)calloc(nfreq, sizeof(float));
        float *hovcc2  = (float *)calloc(nfreq, sizeof(float));
        float *combined = (float *)malloc(nfreq * sizeof(float));
        float *lnstd    = (float *)calloc(nfreq, sizeof(float));

        if (n3 <= 1) {
            /* single window: just copy */
            for (ic = 0; ic < nfreq; ic++) {
                hovcc1[ic]   = hov[ic * 2 + 0];
                hovcc2[ic]   = hov[ic * 2 + 1];
            }

        } else if (!ccweight) {
            /* uniform ln-mean across windows (no CC/AR weighting) */
            for (ic = 0; ic < nfreq; ic++) {
                double s1 = 0.0, s2 = 0.0;
                for (iw = 0; iw < n3; iw++) {
                    float v1 = hov[(iw * nfreq + ic) * 2 + 0];
                    float v2 = hov[(iw * nfreq + ic) * 2 + 1];
                    s1 += log((double)v1 + 1e-30);
                    s2 += log((double)v2 + 1e-30);
                }
                hovcc1[ic] = (float)exp(s1 / n3);
                hovcc2[ic] = (float)exp(s2 / n3);
            }

        } else {
            /* CC-AR weighted ln-mean (O'Connell & Girard 2026) */

            /* step 1: raw ln-mean across windows for h1/v */
            float *rawlm = (float *)calloc(nfreq, sizeof(float));
            for (ic = 0; ic < nfreq; ic++) {
                double s = 0.0;
                for (iw = 0; iw < n3; iw++)
                    s += log((double)hov[(iw * nfreq + ic) * 2 + 0] + 1e-30);
                rawlm[ic] = (float)exp(s / n3);
            }

            /* step 2: reference freq for normalization */
            int iref = (nfreq > 8) ? 8 : 0;
            float rval = rawlm[iref];
            if (rval < 1e-20f) rval = 1e-20f;

            float *href = (float *)malloc(nfreq * sizeof(float));
            for (ic = 0; ic < nfreq; ic++)
                href[ic] = rawlm[ic] / rval;

            /* step 3: CC weights per window */
            float *wts = (float *)malloc(n3 * sizeof(float));
            float  wmin = 1e30f, wmax = -1e30f;

            for (iw = 0; iw < n3; iw++) {
                float wr = hov[(iw * nfreq + iref) * 2 + 0];
                if (wr < 1e-20f) wr = 1e-20f;

                double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
                for (ic = 0; ic < nfreq; ic++) {
                    double a = (double)href[ic];
                    double b = (double)hov[(iw * nfreq + ic) * 2 + 0] / wr;
                    sx  += a;   sy  += b;
                    sxx += a*a; syy += b*b;
                    sxy += a*b;
                }
                double num = (double)nfreq * sxy - sx * sy;
                double dn2 = ((double)nfreq * sxx - sx*sx) *
                             ((double)nfreq * syy - sy*sy);
                double den = (dn2 > 0.0) ? sqrt(dn2) : 0.0;
                wts[iw] = (den > 1e-30) ? (float)(num / den) : 1.0f;
                if (wts[iw] < wmin) wmin = wts[iw];
                if (wts[iw] > wmax) wmax = wts[iw];
            }

            /* step 4: AR weights (high-freq / low-freq ratio) */
            /* find freq indices for 1-2 Hz and 5-10 Hz bands */
            float lobot = 1.0f, lotop = 2.0f;
            float hibot = 5.0f, hitop = 10.0f;
            for (iw = 0; iw < n3; iw++) {
                float mlo = 0.0f, mhi = 0.0f;
                int   nlo = 0, nhi = 0;
                for (ic = 0; ic < nfreq; ic++) {
                    float f = (ic + ilo) * df;
                    float v = hov[(iw * nfreq + ic) * 2 + 0];
                    if (f >= lobot && f <= lotop) { mlo += v; nlo++; }
                    if (f >= hibot && f <= hitop) { mhi += v; nhi++; }
                }
                if (nlo > 0) mlo /= nlo; else mlo = 1.0f;
                if (nhi > 0) mhi /= nhi; else mhi = 1.0f;
                if (mlo < 1e-20f) mlo = 1e-20f;
                float ar = mhi / mlo;
                /* combine CC and AR: w = cc_w / ar^2 */
                float ccw = wts[iw] - wmin + 0.01f;
                wts[iw] = ccw / (ar * ar + 1e-20f);
            }

            /* normalize weights */
            float wsum = 0.0f;
            for (iw = 0; iw < n3; iw++) wsum += wts[iw];
            if (wsum < 1e-30f) wsum = 1.0f;
            for (iw = 0; iw < n3; iw++) wts[iw] /= wsum;

            /* step 5: weighted ln-mean for both components */
            for (ic = 0; ic < nfreq; ic++) {
                double s1 = 0.0, s2 = 0.0;
                for (iw = 0; iw < n3; iw++) {
                    s1 += wts[iw] * log((double)hov[(iw*nfreq+ic)*2+0] + 1e-30);
                    s2 += wts[iw] * log((double)hov[(iw*nfreq+ic)*2+1] + 1e-30);
                }
                hovcc1[ic] = (float)exp(s1);
                hovcc2[ic] = (float)exp(s2);
            }

            free(rawlm); free(href); free(wts);
        }

        /* combined H/V and ln-stddev */
        for (ic = 0; ic < nfreq; ic++)
            combined[ic] = sqrtf(hovcc1[ic] * hovcc1[ic] +
                                 hovcc2[ic] * hovcc2[ic]);

        if (n3 > 1) {
            for (ic = 0; ic < nfreq; ic++) {
                double mn = log((double)combined[ic] + 1e-30);
                double s2 = 0.0;
                for (iw = 0; iw < n3; iw++) {
                    float c = sqrtf(hov[(iw*nfreq+ic)*2+0] * hov[(iw*nfreq+ic)*2+0] +
                                    hov[(iw*nfreq+ic)*2+1] * hov[(iw*nfreq+ic)*2+1]);
                    double lnc = log((double)c + 1e-30);
                    s2 += (lnc - mn) * (lnc - mn);
                }
                lnstd[ic] = (float)sqrt(s2 / n3);
            }
        }

        /* ---------------------------------------------------------- */
        /*  write to output volume                                    */
        /* ---------------------------------------------------------- */
        /* layout: n3=0: combined, n3=1: h1/v, n3=2: h2/v, n3=3: stddev */
        /* linear in memory: [n3_out][n2][nfreq] */
        size_t roff = (size_t)ir * nfreq;
        memcpy(outvol + 0 * (size_t)n2 * nfreq + roff, combined, nfreq * sizeof(float));
        memcpy(outvol + 1 * (size_t)n2 * nfreq + roff, hovcc1,   nfreq * sizeof(float));
        memcpy(outvol + 2 * (size_t)n2 * nfreq + roff, hovcc2,   nfreq * sizeof(float));
        memcpy(outvol + 3 * (size_t)n2 * nfreq + roff, lnstd,    nfreq * sizeof(float));

        /* cleanup thread-local */
        kiss_fftr_free(fft_cfg);
        free(tbuf); free(fbuf);
        free(spec_z); free(spec_h1); free(spec_h2);
        free(hov); free(hovcc1); free(hovcc2);
        free(combined); free(lnstd);

        if (verb) {
#ifdef _OPENMP
            #pragma omp critical
#endif
            show_progress(ir + 1, n2, t0);
        }
    } /* end receivers */

    /* write output */
    sf_floatwrite(outvol, (size_t)4 * n2 * nfreq, fout);

    if (verb) {
        double elapsed = difftime(time(NULL), t0);
        sf_warning("  done in %.1f seconds", elapsed);
    }

    /* cleanup */
    free(tapers); free(lam); free(ftap);
    free(dz); free(dh1); free(dh2);
    free(outvol);

    return 0;
}
