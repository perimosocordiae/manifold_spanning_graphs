# Manifold Spanning Graphs

Code for the paper:
 > Carey and Mahadevan, [Manifold Spanning Graphs](http://people.cs.umass.edu/~ccarey/pubs/msg.pdf), AAAI-2014: Twenty-Eighth Conference on Artificial Intelligence. July, 2014.


## Instructions

Ensure that stable versions of the following Python 2 libraries are installed:

 * [numpy](http://www.numpy.org/)
 * [scipy](http://www.scipy.org/)
 * [scikit-learn](http://scikit-learn.org/)
 * [matplotlib](http://matplotlib.org/)

Optionally, install [bottleneck](https://pypi.python.org/pypi/Bottleneck)
for faster k-nearest neighbor search.

Run the experiments to generate the figures:

    python swiss_roll.py
    python mnist.py

Both files have some options,
which you can see by passing them the `--help` flag.


## MNIST data

The MNIST dataset is available at: http://yann.lecun.com/exdb/mnist/

Download the images and labels for the test set (`t10k`).
To convert the data into a `.npy` file for use in `mnist.py`,
a little fiddling is necessary.

TODO: instructions/code for converting MNIST data.
