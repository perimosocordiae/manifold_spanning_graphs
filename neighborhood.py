import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

try:
  from bottleneck import argpartsort
except ImportError:
  try:
    # Added in version 1.8, which is pretty new.
    # Sadly, it's still slower than bottleneck's version.
    argpartsort = np.argpartition
  except AttributeError:
    argpartsort = lambda arr,k: np.argsort(arr)


def min_k_indices(arr, k, inv_ind=False):
  '''Returns indices of the k-smallest values in each row, unsorted.
  The `inv_ind` flag returns the tuple (k-smallest,(n-k)-largest). '''
  psorted = argpartsort(arr, k)
  if inv_ind:
    return psorted[...,:k], psorted[...,k:]
  return psorted[...,:k]


def neighbor_graph(X, precomputed=False, k=None, epsilon=None, symmetrize=True, weighting='binary'):
  '''Construct an adj matrix from a matrix of points (one per row).
  When `precomputed` is True, X is a distance matrix.
  `weighting` param can be one of {binary, none}.'''
  assert ((k is not None) or (epsilon is not None)
          ), "Must provide `k` or `epsilon`"
  assert weighting in ('binary','none'), "Invalid weighting param: "+weighting
  num_pts = X.shape[0]
  if precomputed:
    dist = X.copy()
  else:
    dist = pairwise_distances(X, metric='sqeuclidean')
  if k is not None:
    k = min(k+1, num_pts)
    nn,not_nn = min_k_indices(dist, k, inv_ind=True)
  if epsilon is not None:
    if k is not None:
      dist[np.arange(dist.shape[0]), not_nn.T] = np.inf
    in_ball = dist <= epsilon
    dist[~in_ball] = 0  # zero out neighbors too far away
    if symmetrize and k is not None:
      # filtering may have caused asymmetry
      dist = (dist + dist.T) / 2
  else:
    for i in xrange(num_pts):
      dist[i,not_nn[i]] = 0  # zero out neighbors too far away
    if symmetrize:
      dist = (dist + dist.T) / 2
  if weighting is 'binary':
    # cycle through boolean and back to get 1/0 in floating points
    return dist.astype(bool).astype(float)
  return dist
