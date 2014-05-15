# Manifold Spanning Graphs

Code for the paper:
 > Carey and Mahadevan, [Manifold Spanning Graphs](http://people.cs.umass.edu/~ccarey/pubs/msg.pdf), AAAI-2014: Twenty-Eighth Conference on Artificial Intelligence. July, 2014.


## Setup Instructions

Ensure that stable versions of the following Python 2 libraries are installed:

 * [numpy](http://www.numpy.org/)
 * [scipy](http://www.scipy.org/)
 * [scikit-learn](http://scikit-learn.org/)
 * [matplotlib](http://matplotlib.org/)

Optionally, install [bottleneck](https://pypi.python.org/pypi/Bottleneck)
for faster k-nearest neighbor search.

Next, [download and build `BMatchingSolver`](https://github.com/berty38/BMatchingSolver).
This is required for the b-matching parts of the experiments.
Once built, edit the `BMatchingSolver_PATH` in `b_matching.py` to reflect
the path to the binary you built.

Finally, download the MNIST data.
The paper used the `t10k` set,
which is [available on Yann LeCun's site](http://yann.lecun.com/exdb/mnist/).
To convert the data into `.npy` files for use in `mnist.py`,
run the following code:

```bash
cd data/
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
gunzip t10k-*-ubyte.gz
python -c 'import numpy as np; x=np.fromfile("t10k-images-idx3-ubyte", dtype=np.uint8); np.save("test_data", x[16:].reshape((10000,28,28),order="C"))'
python -c 'import numpy as np; x=np.fromfile("t10k-labels-idx1-ubyte", dtype=np.uint8); np.save("test_labels", x[8:])'
```

## Running Experiments

Run the experiments to generate the figures:

    python swiss_roll.py
    python mnist.py

Both files have some options,
which you can see by passing them the `--help` flag.
