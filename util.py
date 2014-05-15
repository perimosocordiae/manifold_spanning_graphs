from matplotlib import pyplot
import os.path
import sys
import time


class Timer(object):
  '''Context manager for simple timing of code:
  with Timer('test 1'):
    do_test1()
  '''
  def __init__(self, name, out=sys.stdout):
    self.name = name
    self.out = out
    self.start = 0

  def __enter__(self):
    self.out.write(self.name + ': ')
    self.out.flush()
    self.start = time.time()

  def __exit__(self,*args):
    self.out.write("%0.3f seconds\n" % (time.time()-self.start))
    return False


def savefig(name, dirname='.'):
  fname = os.path.join(dirname, name)
  print 'saving figure to', os.path.normpath(fname)
  pyplot.savefig(fname)
  pyplot.clf()
