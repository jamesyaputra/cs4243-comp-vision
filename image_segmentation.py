import random
import warnings
from collections import defaultdict

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.utils._joblib import Parallel, delayed

def randCent(data, k):
    """randomly generate centroids
    parameters
    ------------
    data: <class 'numpy.ndarray'>, shape=[n_samples, n_features], input data to be randomly select centorids.
            
    k:    <class 'int'>   the number of the centroids
    ------------
    return
        centroids: <class 'numpy.ndarray'>, shape=[k, n_features]
    """

    num_samples, num_features = data.shape
    centroids = np.zeros((k, num_features))
    for i in range(k):
        point = random.randrange(num_samples)
        centroid = data[point]
        centroids[i] = centroid

    return centroids

def KMeans(data, k):
    """KMeans algorithm 
    parameters
    ------------
    data: <class 'numpy.ndarray'>, shape=[n_samples, n_features], input data to be randomly select centorids.
            
    k:    <class 'int'>   the number of the centroids
    ------------
    return
        centroids: <class 'numpy.ndarray'>, shape=[k, n_features]
        clusterAssment:  <class 'numpy.matrix'>, shape=[n_samples, 1]
    """

    num_samples = data.shape[0]
    centroids = randCent(data, k)
    distances = np.zeros((num_samples, k))
    clusterAssment = np.zeros((num_samples, 1))

    old_centroids = np.zeros(centroids.shape)
    error = np.linalg.norm(centroids - old_centroids)
    while error != 0:
        for i in range(k):
            distances[:, i] = np.linalg.norm(data - centroids[i], axis = 1)

        clusterAssment = np.argmin(distances, axis = 1)
        old_centroids = centroids.copy()

        for i in range(k):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                centroids[i] = np.mean(data[clusterAssment == i], axis = 0)

        error = np.linalg.norm(centroids - old_centroids)

    return centroids, clusterAssment

def colors(k):
    """ generate the color for the plt.scatter
    parameters
    ------------
    k:    <class 'int'>   the number of the centroids
    ------------
    return
        ret: <class 'list'>, len = k
    """
    ret = list()
    for i in range(k):
        ret.append((random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)))
    
    return ret
 
def _mean_shift_single_seed(centroid, X, nbrs, max_iter):
    """mean shift cluster for single seed.
    Parameters
    ----------
    X : array-like, shape=[n_samples, n_features]
        Samples to cluster.
    nbrs: NearestNeighbors(radius=bandwidth, n_jobs=1).fit(X)
    max_iter: max interations 
    return:
        mean(center) and the total number of pixels which is in the sphere
    """
    centroid = np.array(list(centroid))
    centroid = centroid.reshape(1, -1)
    for i in range(max_iter):
        neighbors = nbrs.radius_neighbors(centroid, return_distance = False)
        in_bandwidth = list()
        for neighbor in neighbors:
            in_bandwidth.append(X[neighbor])
        
        old_centroid = centroid.copy()
        centroid = np.mean(in_bandwidth, axis = 1)
        if np.array_equal(old_centroid, centroid):
            break

    return tuple(np.round(centroid[0]).tolist())

def mean_shift(X, bandwidth=None, seeds=None, bin_seeding=False, min_bin_freq=1, cluster_all=True, max_iter=300,
               n_jobs=None):
    """pipeline of mean shift clustering
    Parameters
    ----------
    X : array-like, shape=[n_samples, n_features]
    bandwidth: the radius of the sphere
    seeds: whether use the bin seed algorithm to generate the initial seeds
    bin_size:    bin_size = bandwidth.
    min_bin_freq: for each bin_seed, the minimize of the points should cover
    return:
        cluster_centers <class 'numpy.ndarray'> shape=[n_cluster, n_features] ,labels <class 'list'>, len = n_samples
    """
    nbrs = NearestNeighbors(radius = bandwidth, n_jobs = 1).fit(X)

    if bin_seeding:
        seeds = get_bin_seeds(X, bandwidth, min_bin_freq)

    all_res = Parallel(n_jobs = n_jobs)(
        delayed(_mean_shift_single_seed)
        (seed, X, nbrs, max_iter) for seed in seeds)
    cluster_centers = np.array(list(set(all_res))).tolist()

    distances = np.zeros((len(X), len(cluster_centers)))
    for i in range(len(cluster_centers)):
        distances[:, i] = np.linalg.norm(X - cluster_centers[i], axis = 1)
    labels = np.argmin(distances, axis = 1)

    return cluster_centers, labels

def get_bin_seeds(X, bin_size, min_bin_freq=1):
    """generate the initial seeds, in order to use the parallel computing 
    Parameters
    ----------
    X : array-like, shape=[n_samples, n_features]
    bin_size:    bin_size = bandwidth.
    min_bin_freq: for each bin_seed, the minimize of the points should cover
    return:
        bin_seeds: dict-like bin_seeds = {key=seed, key_value=he total number of pixels which is in the sphere }
    """
    seeds = dict()
    compressed_X = np.round(np.divide(X, bin_size))
    for point in compressed_X:
        seed = tuple((point * bin_size).tolist())
        if seed not in seeds:
            seeds[seed] = 1
        else:
            seeds[seed] += 1
    
    bin_seeds = list()
    for seed in seeds:
        if seeds[seed] >= min_bin_freq:
            bin_seeds.append(seed)

    return bin_seeds
