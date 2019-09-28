import math

import numpy as np
from scipy.ndimage.filters import convolve
from scipy.spatial.distance import cdist
from skimage import filters
from skimage.feature import corner_peaks

from utils import get_output_space, pad, warp_image


def harris_corners(img, window_size=3, k=0.04):
    """
    Compute Harris corner response map. Following the equation
    R=Det(M)-k(Trace(M)^2).
        
    Args:
        img: Grayscale image of shape (H, W)
        window_size: size of the window function
        k: sensitivity parameter

    Returns:
        response: Harris response image of shape (H, W)
    """

    window = np.ones((window_size, window_size))
    dx = filters.sobel_v(img)
    dy = filters.sobel_h(img)

    # Calculate the elements in the matrix in the formula for M
    ixx = dx ** 2
    ixy = dx * dy
    iyy = dy ** 2

    # Calculate the sum using convolution
    sxx = convolve(ixx, window)
    sxy = convolve(ixy, window)
    syy = convolve(iyy, window)

    # Calculate determinant and trace
    det = (sxx * syy) - (sxy ** 2)
    trace = sxx + syy

    response = det - k * (trace ** 2)

    return response


def simple_descriptor(patch):
    """
    Describe the patch by normalizing the image values into a standard 
    normal distribution (having mean of 0 and standard deviation of 1) 
    and then flattening into a 1D array. 
    
    The normalization will make the descriptor more robust to change 
    in lighting condition.
    
    Args:
        patch: grayscale image patch of shape (h, w)
    
    Returns:
        feature: 1D array of shape (h * w)
    """

    xmin = np.amin(patch)
    xmax = np.amax(patch)
    denominator = xmax - xmin
    if denominator == 0:
        denominator = 1

    feature = np.divide(np.subtract(patch, xmin), denominator).flatten()
    ### END YOUR CODE
    
    return feature


def describe_keypoints(image, keypoints, desc_func, patch_size=16):
    """
    Args:
        image: grayscale image of shape (H, W)
        keypoints: 2D array containing a keypoint (y, x) in each row
        desc_func: function that takes in an image patch and outputs
            a 1D feature vector describing the patch
        patch_size: size of a square patch at each keypoint
                
    Returns:
        desc: array of features describing the keypoints
    """

    image.astype(np.float32)
    desc = []
    for i, kp in enumerate(keypoints):
        y, x = kp
        patch = image[y-(patch_size//2):y+((patch_size+1)//2),
                      x-(patch_size//2):x+((patch_size+1)//2)]
        desc.append(desc_func(patch))
    return np.array(desc)


def match_descriptors(desc1, desc2, threshold=0.5):
    """
    Match the feature descriptors by finding distances between them. A match is formed 
    when the distance to the closest vector is much smaller than the distance to the 
    second-closest, that is, the ratio of the distances should be smaller
    than the threshold. Return the matches as pairs of vector indices.

    Args:
        desc1: an array of shape (M, P) holding descriptors of size P about M keypoints
        desc2: an array of shape (N, P) holding descriptors of size P about N keypoints
        
    Returns:
        matches: an array of shape (Q, 2) where each row holds the indices of one pair 
        of matching descriptors
    """
    matches = []
    M = desc1.shape[0]
    N = desc2.shape[0]

    distances = cdist(desc1, desc2)
    for m in range(M):
        distance_m = dict()
        for n in range(N):
            distance_m[n] = distances[m, n]

        sorted_distance_m = sorted(distance_m.items(), key = lambda x : x[1])
        ratio = sorted_distance_m[0][1] / sorted_distance_m[1][1]
        if ratio < threshold:
            matches.append(np.array([m, sorted_distance_m[0][0]]))

    matches = np.array(matches)
    return matches


def fit_affine_matrix(p1, p2):
    """ Fit affine matrix such that p2 * H = p1 
    
    Args:
        p1: an array of shape (M, P)
        p2: an array of shape (M, P)
        
    Return:
        H: a matrix of shape (P * P) that transform p2 to p1.
    """

    assert (p1.shape[0] == p2.shape[0]),\
        'Different number of points in p1 and p2'
    p1 = pad(p1)
    p2 = pad(p2)

    H = np.linalg.lstsq(p2, p1)[0]

    # Sometimes numerical issues cause least-squares to produce the last
    # column which is not exactly [0, 0, 1]
    H[:,2] = np.array([0, 0, 1])
    return H


def ransac(keypoints1, keypoints2, matches, n_iters=200, threshold=20):
    """
    Use RANSAC to find a robust affine transformation

        1. Select random set of matches
        2. Compute affine transformation matrix
        3. Compute inliers
        4. Keep the largest set of inliers
        5. Re-compute least-squares estimate on all of the inliers

    Args:
        keypoints1: M1 x 2 matrix, each row is a point
        keypoints2: M2 x 2 matrix, each row is a point
        matches: N x 2 matrix, each row represents a match
            [index of keypoint1, index of keypoint 2]
        n_iters: the number of iterations RANSAC will run
        threshold: the number of threshold to find inliers

    Returns:
        H: a robust estimation of affine transformation from keypoints2 to
        keypoints 1
    """
    N = matches.shape[0]
    n_samples = int(N * 0.2)
    matched1 = pad(keypoints1[matches[:,0]])
    matched2 = pad(keypoints2[matches[:,1]])
    
    max_inliers = np.zeros(N)
    n_inliers = 0

    for i in range(n_iters):
        random_indices = np.random.choice(np.arange(N), n_samples)
        random_samples = matches[random_indices]

        r_matched1 = keypoints1[random_samples[:,0]]
        r_matched2 = keypoints2[random_samples[:,1]]
        H = fit_affine_matrix(r_matched1, r_matched2)

        inlier_indices = []
        x = matched2.dot(H)
        for i in range(N):
            ssd = (np.linalg.norm(x[i] - matched1[i])) ** 2
            if ssd < threshold:
                inlier_indices.append(i)

        if len(inlier_indices) > n_inliers:
            n_inliers = len(inlier_indices)
            max_inliers = inlier_indices
                
    inliers = matches[max_inliers]
    ransac_matched1 = keypoints1[inliers[:,0]]
    ransac_matched2 = keypoints2[inliers[:,1]]
    H = fit_affine_matrix(ransac_matched1, ransac_matched2)
    return H, matches[max_inliers]


def sift_descriptor(patch):
    """
    Implement a simplifed version of Scale-Invariant Feature Transform (SIFT).
    Paper reference: https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf
    
    The implementation does not need exactly match the SIFT reference.
    Here are the key properties of the (baseline) descriptor:
    (1) a 4x4 grid of cells, each length of 16/4=4. It is simply the
        terminology used in the feature literature to describe the spatial
        bins where gradient distributions will be described.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length.

    Args:
        patch: grayscale image patch of shape (h, w)

    Returns:
        feature: 1D array of shape (128, )
    """
    
    dx = filters.sobel_v(patch)
    dy = filters.sobel_h(patch)
    histogram = np.zeros((4,4,8))
    
    for i in range(4):
        for j in range(4):
            row_offset = i * 4
            col_offset = j * 4
            dx_partition = dx[row_offset:row_offset + 4, col_offset:col_offset + 4]
            dy_partition = dy[row_offset:row_offset + 4, col_offset:col_offset + 4]

            cell_hist = np.zeros((8))
            for k in range(4):
                for l in range(4):
                    grad_mag = np.sqrt(np.power(dx_partition[k][l], 2) + np.power(dy_partition[k][l], 2))
                    grad_ori = np.rad2deg(np.arctan2(dy_partition[k][l], dx_partition[k][l])) % 360
                    bin_index = int(grad_ori // 45)
                    cell_hist[bin_index] += grad_mag
            histogram[i][j] = cell_hist
    
    feature = histogram.flatten()
    feature = np.power(feature, 0.65)
    feature = feature / np.linalg.norm(feature)
    return feature
