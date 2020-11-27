"""Helper for evaluation on the Labeled Faces in the Wild dataset 
"""

# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from scipy.sparse import csr_matrix
import facenet
import math
import xclib.evaluation.xc_metrics as xc_metrics
from sklearn.metrics import roc_auc_score

def evaluate(embeddings, actual_issame, nrof_folds=10, distance_metric=0, subtract_mean=False):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy = facenet.calculate_roc(thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), nrof_folds=nrof_folds, distance_metric=distance_metric, subtract_mean=subtract_mean)
    thresholds = np.arange(0, 4, 0.001)
    val, val_std, far = facenet.calculate_val(thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), 1e-3, nrof_folds=nrof_folds, distance_metric=distance_metric, subtract_mean=subtract_mean)
    return tpr, fpr, accuracy, val, val_std, far

def evaluate_auc(embeddings, Yte):
    bsize = 1024
    rows, cols = Yte.nonzero()
    actual_issame = Yte.data

    P = embeddings[0::2][rows]
    Q = embeddings[1::2][cols] 
    
    #assert P.shape[0] == Q.shape[0] and actual_issame.shape[0] == P.shape[0]
    m = P.shape[0]
    segment_m = math.ceil(m/bsize)
    scores = np.zeros(m, dtype=P.dtype)

    for i in range(segment_m):
        i_start, i_end = i*bsize, min((i+1)*bsize, m)
        scores[i_start:i_end] = np.sum(np.multiply(P[i_start:i_end, :], Q[i_start:i_end, :]), axis=-1)

    return roc_auc_score(actual_issame, scores)

def evaluate_patk(embeddings, Yte, k=5):
    bsize = 1024
    m, n = Yte.shape
    sample_idxes = rng.choice(m*n, size=int(0.01*m*n))
    rows = sample_idxes // n
    cols = sample_idxes % n 
    labels = np.ravel(Yte[rows, cols])

    P = embeddings[0::2]
    Q = embeddings[1::2] 
    
    # score_mat = P.dot(Q.T)
    m, n = P.shape[0], Q.shape[0]
    segment_m, segment_n = math.ceil(m/bsize), math.ceil(n/bsize)
    precision, ndcg = np.zeros(k, dtype=P.dtype), np.zeros(k, dtype=P.dtype)
    predictions = list()

    for i in range(segment_m):
        i_start, i_end = i*bsize, min((i+1)*bsize, m)
        score_mat = np.zeros((i_end-i_start, n), dtype=P.dtype)
        for j in range(segment_n):
            j_start, j_end = j*bsize, min((j+1)*bsize, n)
            score_mat[:, j_start:j_end] = P[i_start:i_end].dot(Q[j_start:j_end].T)
        np.fill_diagonal(score_mat[:, i_start:], -np.inf)
        _rows = rows[rows >= i_start and row < i_end ] - i_start
        _cols = cols[rows >= i_start and row < i_end ]
        predictions.append(np.nan_to_num(score_mat[_rows, _cols], neginf=-1))
        _precision = xc_metrics.precision(score_mat, Yte[i_start:i_end], k=k)
        _ndcg = xc_metrics.ndcg(score_mat, Yte[i_start:i_end], k=k)
        precision += (i_end-i_start)*_precision
        ndcg += (i_end-i_start)*_ndcg
    precision /= m
    ndcg /= m
    auc = roc_auc_score(labels, np.hstack(predictions))

    return precision, ndcg, auc

def get_paths(lfw_dir, pairs):
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []
    for pair in pairs:
        if len(pair) == 3:
            path0 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
            path1 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2])))
            issame = True
        elif len(pair) == 4:
            path0 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
            path1 = add_extension(os.path.join(lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3])))
            issame = False
        if os.path.exists(path0) and os.path.exists(path1):    # Only add the pair if both paths exist
            path_list += (path0,path1)
            issame_list.append(issame)
        else:
            nrof_skipped_pairs += 1
    if nrof_skipped_pairs>0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)
    
    return path_list, issame_list
  
def add_extension(path):
    if os.path.exists(path+'.jpg'):
        return path+'.jpg'
    elif os.path.exists(path+'.png'):
        return path+'.png'
    else:
        raise RuntimeError('No file "%s" with extension png or jpg.' % path)

def read_pairs(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
    return np.array(pairs)



