import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
import numpy as np
from os.path import dirname, join as joinpath
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.semi_supervised import LabelPropagation

from neighborhood import neighbor_graph
from b_matching import hacky_b_matching
from msg import manifold_spanning_graph
from util import Timer, savefig

FIGURE_DIR = joinpath(dirname(__file__), 'figures')
MNIST_PATH = joinpath(dirname(__file__), 'data/')


def mnist_experiment(digits, opts):
  X, GT, num_ccs = load_data(digits)
  if opts.cache:
    # TODO: include the digits used in the cache filename
    D = np.load(joinpath(opts.cache, 'mnist_D.npy'))
    Wknn = np.load(joinpath(opts.cache, 'mnist_Wknn.npy'))
    Wbma = np.load(joinpath(opts.cache, 'mnist_Wbma.npy'))
    Wmsg = np.load(joinpath(opts.cache, 'mnist_Wmsg.npy'))
  else:
    D, Wknn, Wbma, Wmsg = compute_Ws(X, num_ccs)

  bad_edges = GT != GT[:,None]
  edge_error(Wknn, bad_edges, 'knn')
  edge_error(Wbma, bad_edges, 'b-matching')
  edge_error(Wmsg, bad_edges, 'MSG')

  if opts.sparsity:
    plot_sparsity(D, Wknn, Wbma, Wmsg)

  classify(Wknn, Wbma, Wmsg, GT, opts.classify)


def edge_error(W, bad_edges, name):
  wrong = np.count_nonzero(W[bad_edges])
  total = np.count_nonzero(W)
  print '%s: err = %d/%d = %f' % (name, wrong, total, float(wrong)/total)


def load_data(digits=None):
  X = np.load(joinpath(MNIST_PATH, 'test_data.npy'))
  GT = np.load(joinpath(MNIST_PATH, 'test_labels.npy'))
  if digits is not None:
    assert all(0 <= d < 10 for d in digits)
    mask = np.logical_or.reduce(GT==np.array(digits)[:,None])
    num_ccs = len(digits)
    GT = GT[mask]
    X = X[mask].reshape((len(GT), -1))
  else:
    num_ccs = 10
    X = X.reshape((len(GT), -1))
  order = np.argsort(GT)
  X = X[order]
  GT = GT[order]
  return X, GT, num_ccs


def compute_Ws(X, num_ccs):
  with Timer('Calculating pairwise distances...'):
    D = pairwise_distances(X, metric='sqeuclidean')
  np.save('mnist_D.npy', D)
  # k-nn
  with Timer('Calculating knn graph...'):
    for k in xrange(1,10):
      Wknn = neighbor_graph(D, precomputed=True, k=k, symmetrize=True)
      n = connected_components(Wknn, directed=False, return_labels=False)
      if n <= num_ccs:
        break
    else:
      assert False, 'k too low'
  np.save('mnist_Wknn.npy', Wknn)
  print 'knn (k=%d)' % k

  # b-matching
  with Timer('Calculating b-matching graph...'):
    # using 8 decimal places kills the disk
    Wbma = hacky_b_matching(D, k, fmt='%.1f')
  np.save('mnist_Wbma.npy', Wbma)

  # msg
  with Timer('Calculating MSG graph...'):
    Wmsg = manifold_spanning_graph(X, 2, num_ccs=num_ccs)
  np.save('mnist_Wmsg.npy', Wmsg)

  return D, Wknn, Wbma, Wmsg


def plot_sparsity(D, Wknn, Wbma, Wmsg):
  # plot distances and sparsity patterns
  pyplot.imshow(D, interpolation='nearest')
  savefig('mnist_l2_dist.png', FIGURE_DIR)
  pyplot.spy(Wknn, markersize=1)
  savefig('mnist_knn_edges.png', FIGURE_DIR)
  pyplot.spy(Wbma, markersize=1)
  savefig('mnist_bma_edges.png', FIGURE_DIR)
  pyplot.spy(Wmsg, markersize=1)
  savefig('mnist_msg_edges.png', FIGURE_DIR)


def classify(Wknn, Wbma, Wmsg, GT, num_labeled):
  # Note: in GT and labels, -1 means missing label
  while True:
    label_idx = np.random.choice(len(GT), size=num_labeled, replace=False)
    labels = np.zeros(GT.shape, dtype=int) - 1
    labels[label_idx] = GT[label_idx]
    n = len(np.unique(labels))
    if n == 11:  # all labels represented
      break

  knn_res = edge_propagation(Wknn, labels)
  bma_res = edge_propagation(Wbma, labels)
  msg_res = edge_propagation(Wmsg, labels)

  confusion(knn_res, 'knn', GT)
  confusion(bma_res, 'bma', GT)
  confusion(msg_res, 'msg', GT)


def confusion(result, name, GT):
  header = '%s classifier: %d' % (name, np.count_nonzero(result == GT))
  print header
  cm = confusion_matrix(GT, result)
  fname = joinpath(FIGURE_DIR, name+'_cm.tex')
  with open(fname, 'w') as fh:
    print >>fh, '%', header
    print >>fh, '\\begin{tabular}{c|*{10}{r}}'
    print >>fh, ' & 0 & 1 & 2 & 3 & 4 & 5 & 6 & 7 & 8 & 9 \\\\'
    print >>fh, '\\hline'
    for i, row in enumerate(cm):
      best = np.argmax(row)
      nums = map(str, row)
      nums[best] = '\\textbf{'+nums[best]+'}'
      print >>fh, i, '&', ' & '.join(nums), '\\\\'
    print >>fh, '\\end{tabular}'


## Simple edge propagator, hacked onto the sklearn version.
def edge_propagation(W, labels):
  '''W is binary adj matrix, label -1 means unknown'''
  W = W.astype(float)
  np.fill_diagonal(W, 1)
  P = np.array([row.nonzero()[0] for row in W])
  return _LP(P=P).fit(W, labels).predict(W)


class _LP(LabelPropagation):
  def __init__(self, P):
    LabelPropagation.__init__(self, kernel='knn')
    self.P = P

  def _get_kernel(self, W, W2=None):
    if W2 is None:
      return W
    return self.P


if __name__ == '__main__':
  from optparse import OptionParser
  op = OptionParser()
  op.add_option('--digits', type=str, default='all',
                help='comma-separated list of digits [all digits]')
  op.add_option('--classify', type=int, metavar='N', default=20,
                help='# of examples for MNIST classification test [20]')
  op.add_option('--no-sparsity', action='store_false', dest='sparsity',
                default=True, help="Don't plot sparsity figures.")
  op.add_option('--cache', type=str, help='Path to cached .npy files')
  opts, args = op.parse_args()
  digits = None if opts.digits == 'all' else map(int,opts.digits.split(','))
  mnist_experiment(digits, opts)
