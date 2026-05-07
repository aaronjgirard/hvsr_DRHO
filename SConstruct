##############################################################################
# HVSR workflow -- multitaper H/V spectral ratio analysis
#
# input: 3 RSF files (Z, H1, H2) with axes n1 x n2 x n3
#   n1 = time samples per window
#   n2 = receivers
#   n3 = time windows
#
# output: HVSR curves per receiver (combined, H1/V, H2/V, stddev)
#
# notes:
#   - compile cpu:  cc -O2 -fopenmp -I$RSFROOT/include -L$RSFROOT/lib
#                   -o sfhvsr Mhvsr.c -lrsf -lm
#   - compile gpu:  nvcc -O2 -I$RSFROOT/include -L$RSFROOT/lib
#                   -o sfhvsr_gpu Mhvsr_gpu.c -lrsf -lm -lcufft
#   - set par['exe'] = './sfhvsr_gpu' in par dict to use gpu version
##############################################################################
from rsf.proj import *
import os

par = {
    'nwin'   : 5,          # number of slepian tapers
    'npi'    : 3,          # time-bandwidth product
    'fmin'   : 0.1,        # [Hz] min output frequency
    'fmax'   : 45.0,       # [Hz] max output frequency
    'ccwt'   : 'y',        # cc-ar weighting (y/n)
    'zfile'  : 'data_z',   # vertical component rsf (no .rsf)
    'h1file' : 'data_n',   # horizontal-1 rsf
    'h2file' : 'data_e',   # horizontal-2 rsf
}

## set exe='./sfhvsr_gpu' to use gpu version
par['exe'] = par.get('exe', './sfhvsr')

## compute hvsr for all receivers
Flow('hvsr', [par['zfile'], par['h1file'], par['h2file']], '''
     %(exe)s h1=${SOURCES[1]} h2=${SOURCES[2]}
     nwin=%(nwin)d npi=%(npi)d
     fmin=%(fmin)g fmax=%(fmax)g
     ccweight=%(ccwt)s verb=1
     ''' % par)

## extract individual output components
for ic, name in enumerate(['hv_combined', 'hv_h1', 'hv_h2', 'hv_stddev']):
    Flow(name, 'hvsr', 'window n3=1 f3=%d' % ic)

## combined H/V for all receivers (overlay)
Plot('hv_combined', '''
     graph title="Combined H/V (all receivers)"
     label1="Frequency" unit1="Hz"
     label2="H/V" plotcol=7
     ''')

## mean +/- 1 sigma
Flow('hv_upper', ['hv_combined', 'hv_stddev'], '''
     math c=${SOURCES[1]} output="input*exp(c)"
     ''')
Flow('hv_lower', ['hv_combined', 'hv_stddev'], '''
     math c=${SOURCES[1]} output="input*exp(-c)"
     ''')

## single receiver example (receiver 0)
Flow('hv_r0', 'hv_combined', 'window n2=1 f2=0')
Flow('h1v_r0', 'hv_h1', 'window n2=1 f2=0')
Flow('h2v_r0', 'hv_h2', 'window n2=1 f2=0')
Flow('std_r0', 'hv_stddev', 'window n2=1 f2=0')
Flow('up_r0', 'hv_upper', 'window n2=1 f2=0')
Flow('lo_r0', 'hv_lower', 'window n2=1 f2=0')

Plot('hv_r0', '''
     graph title="Receiver 0: Combined H/V"
     label1="Frequency" unit1="Hz"
     label2="H/V" plotcol=7
     ''')
Plot('up_r0', '''
     graph title="" wantaxis=n
     plotcol=5 dash=1
     ''')
Plot('lo_r0', '''
     graph title="" wantaxis=n
     plotcol=5 dash=1
     ''')
Plot('h1v_r0', '''
     graph title="" wantaxis=n plotcol=4
     ''')
Plot('h2v_r0', '''
     graph title="" wantaxis=n plotcol=6
     ''')

Result('hvsr_r0', ['hv_r0', 'up_r0', 'lo_r0', 'h1v_r0', 'h2v_r0'], 'Overlay')

Result('hvsr_all', 'hv_combined', '''
       graph title="Combined H/V (all receivers)"
       label1="Frequency" unit1="Hz"
       label2="H/V" plotcol=7
       ''')

End()
