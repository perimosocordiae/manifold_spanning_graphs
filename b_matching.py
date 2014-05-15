import numpy as np
from os.path import expanduser
from subprocess import check_call

BMatchingSolver_PATH = expanduser('~/Downloads/BMatchingSolver/Release/BMatchingSolver')
DEGREE_PATH = 'degree.txt'
MATRIX_PATH = 'D.txt'
RESULT_PATH = 'results.txt'


def hacky_b_matching(D, b, max_iter=5000, cache_size=None, fmt='%.8f'):
  if cache_size is None:
    cache_size = D.shape[0]//2
  np.savetxt(DEGREE_PATH, np.zeros((D.shape[0],1), dtype=int)+b, fmt='%d')
  np.savetxt(MATRIX_PATH, -D, fmt=fmt)
  cmd = '%s -w %s -d %s -o %s -n %d -t 0 -v 0 -c %d -i %d >/dev/null' % (
      BMatchingSolver_PATH, MATRIX_PATH, DEGREE_PATH,
      RESULT_PATH, D.shape[0], cache_size, max_iter)
  check_call(cmd, shell=True)
  pairs = np.loadtxt(RESULT_PATH, dtype=int)
  W = np.zeros_like(D, dtype=int)
  W[pairs[:,0],pairs[:,1]] = 1
  return W
