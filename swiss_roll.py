# import matplotlib
# matplotlib.use('Agg')
from matplotlib import pyplot
import numpy as np
from os.path import dirname, join as joinpath
from scipy.sparse.csgraph import connected_components, dijkstra
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.decomposition import KernelPCA

from viz import show_neighbor_graph, scatterplot
from neighborhood import neighbor_graph
from b_matching import hacky_b_matching
from msg import manifold_spanning_graph, grow_trees, join_CCs_simple

FIGURE_DIR = joinpath(dirname(__file__), 'figures')
DEBUG = False


def savefig(name):
  if DEBUG:
    print name
    pyplot.show()
  else:
    pyplot.savefig(joinpath(FIGURE_DIR, name))
  pyplot.clf()
  pyplot.close('all')


def swiss_roll_experiment():
  embed_dim = 2
  X, GT = make_test_data(verify=True)

  plot_canonical_roll(X, GT)
  evaluate_sensitivity(X, GT)

  # kNN
  D = pairwise_distances(X, metric='sqeuclidean')
  for k in xrange(3,10):
    Wknn = neighbor_graph(D, precomputed=True, k=k, symmetrize=True)
    n = connected_components(Wknn, directed=False, return_labels=False)
    if n == 1:
      break
  else:
    assert False, 'k too low'
  print 'k:', k, 'error:', error_ratio(Wknn, GT)
  plot_roll(Wknn, X, GT[:,0], embed_dim, 'swiss_knn_result.png')

  # eball
  for eps in np.linspace(0.4, 1.2, 50):
    Weps = neighbor_graph(D, precomputed=True, epsilon=eps, symmetrize=False)
    n = connected_components(Weps, directed=False, return_labels=False)
    if n == 1:
      break
  else:
    assert False, 'eps too low'
  print 'eps:', eps, 'error:', error_ratio(Weps, GT)
  plot_roll(Weps, X, GT[:,0], embed_dim, 'swiss_eps_result.png')

  # b-matching
  for b in xrange(3,10):
    Wbma = hacky_b_matching(D, b)
    n = connected_components(Wbma, directed=False, return_labels=False)
    if n == 1:
      break
  else:
    assert False, 'b too low'
  print 'b:', b, 'error:', error_ratio(Wbma, GT)
  plot_roll(Wbma, X, GT[:,0], embed_dim, 'swiss_bma_result.png')

  # MSG
  Wmsg = manifold_spanning_graph(X, embed_dim)
  print 'MSG error:', error_ratio(Wmsg, GT)
  plot_roll(Wmsg, X, GT[:,0], embed_dim, 'swiss_msg_result.png')


def plot_roll(W, X, colors, embed_dim, image_name):
  fig, (ax1,ax2) = pyplot.subplots(ncols=2)

  show_neighbor_graph(X[:,(0,2)], W, ax=ax1, vertex_colors=colors,
                      vertex_edgecolor='k', edge_style='k-')

  # show the embedding
  D = pairwise_distances(X, metric='euclidean')
  D[~W.astype(bool)] = 0
  embed = isomap(D, embed_dim)
  scatterplot(embed, ax=ax2, c=colors, marker='o', edgecolor='k')
  for ax in (ax1,ax2):
    ax.tick_params(which='both', bottom='off', top='off', left='off',
                   right='off', labelbottom='off', labelleft='off')
  fig.tight_layout()
  savefig(image_name)


def isomap(W, num_vecs):
  G = -0.5 * dijkstra(W, directed=False) ** 2
  embedder = KernelPCA(n_components=num_vecs, kernel='precomputed')
  return embedder.fit_transform(G)


def show_skeleton_issue():
  t = np.linspace(0,4,25)[:,None]
  X = np.hstack((np.cos(t), np.random.uniform(-1,1,t.shape), np.sin(t)))
  GT = np.hstack((t, X[:,1:2]))
  W = neighbor_graph(X, k=1, symmetrize=False)
  W = grow_trees(X, W, 2)
  labels = join_CCs_simple(X, W)
  # switch up the CC order for better contrast between groups
  order = np.arange(labels.max()+1)
  np.random.shuffle(order)
  labels = order[labels]
  show_neighbor_graph(GT, W, vertex_style=None, edge_style='k-')
  ax = pyplot.gca()
  for l,marker in zip(np.unique(labels), "osD^v><"):
    scatterplot(GT[labels==l], marker, ax=ax, edgecolor='k', c='white')
  ax.tick_params(which='both', bottom='off', top='off', left='off',
                 right='off', labelbottom='off', labelleft='off')
  savefig('skeleton.png')


def average_error_rate(num_trials=100):
  knn_err = np.empty((num_trials,2))
  eps_err = np.empty((num_trials,2))
  bma_err = np.empty((num_trials,2))
  msg_err = np.empty((num_trials,2))
  for i in xrange(num_trials):
    X, GT = make_test_data(verify=True)
    D = pairwise_distances(X, metric='sqeuclidean')
    W = neighbor_graph(D, precomputed=True, k=5, symmetrize=True)
    knn_err[i] = error_ratio(W, GT, return_tuple=True)
    W = neighbor_graph(D, precomputed=True, epsilon=1.0)
    eps_err[i] = error_ratio(W, GT, return_tuple=True)
    W = hacky_b_matching(D, 5)
    bma_err[i] = error_ratio(W, GT, return_tuple=True)
    W = manifold_spanning_graph(X, 2)
    msg_err[i] = error_ratio(W, GT, return_tuple=True)
  errors = np.hstack((knn_err[:,:1],eps_err[:,:1],bma_err[:,:1],msg_err[:,:1]))
  edges = np.hstack((knn_err[:,1:],eps_err[:,1:],bma_err[:,1:],msg_err[:,1:]))
  labels = ('$k$-nearest','$\\epsilon$-close','$b$-matching','MSG')

  pyplot.figure(figsize=(5,6))
  ax = pyplot.gca()
  ax.boxplot(errors, widths=0.75)
  ax.set_xticklabels(labels, fontsize=12)
  ymin,ymax = pyplot.ylim()
  pyplot.ylim((ymin-1, ymax))
  savefig('average_error.png')
  pyplot.figure(figsize=(5,6))
  ax = pyplot.gca()
  ax.boxplot(edges, widths=0.75)
  ax.set_xticklabels(labels, fontsize=12)
  savefig('average_edges.png')


def make_test_data(verify=True):
  while True:
    X, theta = swiss_roll(18, 500, radius=4.8, return_theta=True,
                          theta_noise=0, radius_noise=0)
    GT = np.hstack((theta[:,None], X[:,1:2]))
    GT -= GT.min(axis=0)
    GT /= GT.max(axis=0)
    if not verify:
      break
    # ensure our test_data fits our 1-NN assumption
    W = neighbor_graph(X, k=1, symmetrize=False)
    if error_ratio(W, GT) < 1e-10:
      break
  return X, GT


def swiss_roll(radians, num_points, radius=1.0,
               theta_noise=0.1, radius_noise=0.01,
               return_theta=False):
  theta = np.linspace(1, radians, num_points)
  if theta_noise > 0:
    theta += np.random.normal(scale=theta_noise, size=theta.shape)
  r = np.sqrt(np.linspace(0, radius*radius, num_points))
  if radius_noise > 0:
    r += np.random.normal(scale=radius_noise, size=r.shape)
  roll = np.empty((num_points, 3))
  roll[:,0] = r * np.sin(theta)
  roll[:,2] = r * np.cos(theta)
  roll[:,1] = np.random.uniform(-1,1,num_points)
  if return_theta:
    return roll, theta
  return roll


def error_ratio(W, GT_points, max_delta_theta=0.1, return_tuple=False):
  theta_edges = GT_points[np.argwhere(W),0]
  delta_theta = np.abs(np.diff(theta_edges))
  err_edges = np.count_nonzero(delta_theta > max_delta_theta)
  tot_edges = delta_theta.shape[0]
  if return_tuple:
    return err_edges, tot_edges
  return err_edges / float(tot_edges)


def plot_canonical_roll(X, GT):
  # manifold space
  fig = pyplot.figure(figsize=(8,5))
  scatterplot(GT, c=GT[:,0], edgecolor='k', s=20, marker='o', fig=fig)
  ax = pyplot.gca()
  pyplot.setp(ax.get_xticklabels(), visible=False)
  pyplot.setp(ax.get_yticklabels(), visible=False)
  ax.set_xlabel('$\\theta$', labelpad=15, size=20)
  ax.set_ylabel('$r$', labelpad=20, rotation=0, size=20)
  ax.set_xlim((-0.02, 1.02))
  ax.set_ylim((-0.02, 1.02))
  savefig('swiss_roll_flat.png')
  # input space
  pyplot.figure()
  scatterplot(X, c=GT[:,0], edgecolor='k', s=20, marker='o')
  ax = pyplot.gca()
  pyplot.setp(ax.get_xticklabels(), visible=False)
  pyplot.setp(ax.get_yticklabels(), visible=False)
  pyplot.setp(ax.get_zticklabels(), visible=False)
  ax.set_xlabel('$x$', labelpad=15, size=20)
  ax.set_ylabel('$z$', labelpad=20, rotation=0, size=20)
  ax.set_zlabel('$y$', labelpad=20, rotation=0, size=20)
  ax.view_init(elev=6, azim=265)
  ax.autoscale_view(True, True, True)
  savefig('swiss_roll.png')


def eval_method(values, method_func, GT):
  errors = np.empty_like(values, dtype=float)
  edges = np.empty_like(values, dtype=int)
  conn = np.empty_like(values, dtype=int)
  for j, v in enumerate(values):
    adj = method_func(v)
    errors[j] = error_ratio(adj, GT)
    edges[j] = np.count_nonzero(adj)
    conn[j] = connected_components(adj, directed=False, return_labels=False)
  return errors, edges, conn


def evaluate_sensitivity(X, GT):
  eps_values = np.linspace(0.2, 1.2, 50)
  knn_values = np.arange(1,7)
  eps_method = lambda eps: neighbor_graph(X, epsilon=eps)
  knn_method = lambda k: neighbor_graph(X, k=k, symmetrize=False)

  errors, edges, conn = eval_method(knn_values, knn_method, GT)
  one_cc = knn_values[len(conn) - np.searchsorted(conn[::-1], 1, side='right')]

  fig, axes = pyplot.subplots(nrows=2, ncols=2)
  knn_err_ax, eps_err_ax = axes[0]
  knn_edge_ax, eps_edge_ax = axes[1]

  knn_err_ax.set_ylabel('Edge error %', fontsize=14)
  knn_err_ax.plot(knn_values, errors*100, 'k+-')
  knn_err_ax.axvline(one_cc, color='k', linestyle='--')
  knn_err_ax.set_ylim((-0.05, knn_err_ax.get_ylim()[1]))
  knn_edge_ax.set_xlabel('$k$', fontsize=16)
  knn_edge_ax.set_ylabel('Total edges', fontsize=14)
  knn_edge_ax.plot(knn_values, edges, 'k+-')
  knn_edge_ax.axvline(one_cc, color='k', linestyle='--')

  errors, edges, conn = eval_method(eps_values, eps_method, GT)
  one_cc = eps_values[len(conn) - np.searchsorted(conn[::-1], 1, side='right')]

  eps_err_ax.plot(eps_values, errors*100, 'k+-')
  eps_err_ax.axvline(one_cc, color='k', linestyle='--')
  eps_err_ax.set_ylim((-0.1, eps_err_ax.get_ylim()[1]))
  eps_edge_ax.set_xlabel('$\\epsilon$', fontsize=16)
  eps_edge_ax.plot(eps_values, edges, 'k+-')
  eps_edge_ax.axvline(one_cc, color='k', linestyle='--')

  for ax in (knn_err_ax, knn_edge_ax):
    start,end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(start, end+1))
  fig.tight_layout()
  savefig('sensitivity.png')


if __name__ == '__main__':
  from optparse import OptionParser
  op = OptionParser()
  op.add_option('--average', type=int, metavar='N', default=200,
                help='# trials for the average error test [200]')
  opts, args = op.parse_args()
  pyplot.set_cmap('Greys')
  swiss_roll_experiment()
  show_skeleton_issue()
  average_error_rate(num_trials=opts.average)
