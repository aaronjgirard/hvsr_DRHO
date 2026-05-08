##############################################################################
# build sfhvsr (cpu/openmp) and sfhvsr_gpu (cuda/cufft) madagascar mains
#
# sources:
#   Mhvsr.c     -> sfhvsr     (cc  + openmp,  via bldutil targets.c)
#   Mhvsr_gpu.c -> sfhvsr_gpu (nvcc + cufft,  via custom Command)
#
# usage:
#   scons                 # build both targets in current directory
#   scons sfhvsr          # cpu only
#   scons sfhvsr_gpu      # gpu only (requires nvcc on PATH)
#
# env vars:
#   RSFSRC, RSFROOT, NVCC may be set to override defaults
##############################################################################
import os, sys, shutil, platform, subprocess

sys.path.append('../../framework')

try:
    import bldutil
    glob_build = True              # scons command launched in RSFSRC
    srcroot    = '../..'           # cwd is RSFSRC/build/user/jeff
    Import('env bindir libdir pkgdir')
    env = env.Clone()
except:
    glob_build = False             # scons command launched in the local directory
    srcroot    = os.environ.get('RSFSRC', '../..')
    sys.path.append(os.path.join(srcroot, 'framework'))
    import bldutil
    env        = bldutil.Debug()   # debugging flags for compilers
    bindir     = libdir = pkgdir = None

    ## librsf.a on this system was built without cblas; rsf.h tries to
    ## include <cblas.h> unconditionally unless NO_BLAS is defined
    env.Append(CPPDEFINES=['NO_BLAS'])

targets = bldutil.UserSconsTargets()

## c mains (cpu/openmp)
targets.c = '''
hvsr
'''

## openmp flags
if platform.system() == 'Darwin':
    try:
        omp_prefix = subprocess.check_output(
            ['brew', '--prefix', 'libomp']).decode().strip()
        env.Append(CCFLAGS  =['-Xpreprocessor', '-fopenmp',
                              '-I%s/include' % omp_prefix],
                   LINKFLAGS=['-L%s/lib' % omp_prefix, '-lomp'])
    except Exception:
        env.Append(CPPDEFINES=['NO_OMP'])
else:
    env.Append(CCFLAGS=['-fopenmp'], LINKFLAGS=['-fopenmp'])

targets.build_all(env, glob_build, srcroot, bindir, libdir, pkgdir)

## gpu build via nvcc (Mhvsr_gpu.c -> sfhvsr_gpu)
## skipped under glob_build because RSF top-level scons handles it differently
if not glob_build:
    rsfroot = os.environ.get('RSFROOT', '/usr/local')
    nvcc    = os.environ.get('NVCC') or shutil.which('nvcc') or 'nvcc'

    ## propagate caller PATH so nvcc and helpers it spawns are visible
    gpu_path = os.environ.get('PATH', env['ENV'].get('PATH', ''))

    ## nvcc needs a host g++; if missing on PATH, search common locations
    ## (rhel gcc-toolset-* ships g++ at /opt/rh/gcc-toolset-N/root/usr/bin)
    if not shutil.which('g++', path=gpu_path):
        import glob
        cands = sorted(glob.glob('/opt/rh/gcc-toolset-*/root/usr/bin'),
                       reverse=True)
        if cands:
            gpu_path = cands[0] + os.pathsep + gpu_path

    env['ENV']['PATH'] = gpu_path

    ## -DNO_BLAS  : rsf.h skips <cblas.h> when this is defined (matches
    ##              how librsf.a was built on this system)
    ## -lgomp     : librsf.a was compiled with openmp -- pull in gomp
    gpu_cmd = ('%s -O2 -DNO_BLAS -I%s/include -L%s/lib '
               '-o $TARGET $SOURCE -lrsf -lm -lcufft -lgomp'
               % (nvcc, rsfroot, rsfroot))

    sfhvsr_gpu = env.Command('sfhvsr_gpu', 'Mhvsr_gpu.c', gpu_cmd)
    env.Default(sfhvsr_gpu)
