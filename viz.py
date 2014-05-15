import numpy as np
from matplotlib import pyplot


def scatterplot(X, marker='.', title=None, fig=None, ax=None, **kwargs):
  '''General plotting function for a set of points X. May be [1-3] dimensional.'''
  assert len(X.shape) in (1,2), 'Only valid for 1 or 2-d arrays of points'
  assert len(X.shape) == 1 or X.shape[1] in (1,2,3), 'Only valid for [1-3] dimensional points'
  is_3d = len(X.shape) == 2 and X.shape[1] == 3
  is_1d = len(X.shape) == 1 or X.shape[1] == 1
  if ax is None:
    if fig is None:
      fig = pyplot.gcf()
    if is_3d:
      from mpl_toolkits.mplot3d import Axes3D
      ax = Axes3D(fig)
    else:
      ax = fig.add_subplot(111)
  elif is_3d:
    assert hasattr(ax, 'zaxis'), 'Must provide an Axes3D axis'
  # Do the plotting
  if is_1d:
    ax.scatter(X, marker=marker, **kwargs)
  elif is_3d:
    ax.scatter(X[:,0], X[:,1], X[:,2], marker=marker, **kwargs)
  else:
    ax.scatter(X[:,0], X[:,1], marker=marker, **kwargs)
  if title:
    ax.set_title(title)
  return pyplot.show


def show_neighbor_graph(X, W, title=None, fig=None, ax=None,
                        edge_style='r-', vertex_style='o', vertex_colors='b',
                        vertex_sizes=20, vertex_edgecolor='none'):
  '''Plot the neighbor connections between points in a data set.'''
  assert X.shape[1] in (2,3), 'can only show neighbor graph for 2d or 3d data'
  is_3d = (X.shape[1] == 3)
  if ax is None:
    if is_3d:
      from mpl_toolkits.mplot3d import Axes3D
      if fig is None:
        fig = pyplot.gcf()
      ax = Axes3D(fig)
    else:
      ax = pyplot.gca()
  pairs = np.transpose(np.nonzero(W))
  t = X[pairs]
  # this uses the 'None trick', to insert discontinuties in the line plot
  tX = np.empty((t.shape[0], t.shape[1]+1))
  tX[:,:-1] = t[:,:,0]
  tX[:,-1] = None
  tY = tX.copy()
  tY[:,:-1] = t[:,:,1]
  if is_3d:
    tZ = tX.copy()
    tZ[:,:-1] = t[:,:,2]
    # needs to be a real array, so we use .ravel() instead of .flat
    ax.plot(tX.ravel(), tY.ravel(), tZ.ravel(), edge_style, zorder=1)
    if vertex_style is not None:
      ax.scatter(X[:,0], X[:,1], X[:,2], marker=vertex_style, zorder=2,
                 edgecolor=vertex_edgecolor, c=vertex_colors, s=vertex_sizes)
  else:
    # tX.flat looks like: [x1,x2,NaN, x3,x4,Nan, ...]
    ax.plot(tX.flat, tY.flat, edge_style, zorder=1)
    if vertex_style is not None:
      ax.scatter(X[:,0], X[:,1], marker=vertex_style, zorder=2,
                 edgecolor=vertex_edgecolor, c=vertex_colors, s=vertex_sizes)
  if title:
    ax.set_title(title)
  return pyplot.show
