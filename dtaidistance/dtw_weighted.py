# -*- coding: UTF-8 -*-
"""
dtaidistance.dtw_weighted
~~~~~~~~~~~~~~~~~~~~~~~~~

Dynamic Time Warping (DTW) with custom internal distance function.

:author: Wannes Meert
:copyright: Copyright 2018 KU Leuven, DTAI Research Group.
:license: Apache License, Version 2.0, see LICENSE for details.

Weights are represented using a tuple (-x3, -x2, -x1, -x0, x0, x1, x2, x3)

.. code-block::
    |             /
   3|            +
    |           /
    |          /
   1|      +--+
    |     /
   0+----+
    0   x0 x1 x2 x3

"""
import logging
import math
from collections import defaultdict, deque
import numpy as np
from matplotlib import pyplot as plt

from .dtw import best_path


logger = logging.getLogger("be.kuleuven.dtai.distance")

try:
    from tqdm import tqdm
except ImportError:
    logger.info('tqdm library not available')
    tqdm = None


def warping_paths(s1, s2, weights, window=None, **kwargs):
    """
    Input: s1 and s2 are time series of length n/l1 and m/l2

    :param s1:
    :param s2:
    :param weights: Weights on s1
    :return: DTW similarity m between s1 and s2, warping paths matrix
    """
    # TODO: copy original function in DTW to support all options and integrate weights
    # print('warping_paths')
    l1 = len(s1)
    l2 = len(s2)
    # print('l1', l1, 'l2', l2)
    if window is None:
        window = max(l1, l2)
    else:
        window += 1  # TODO: 0 should be diagonal, this is now 1
    # print('window', window)
    paths = np.full((l1 + 1, l2 + 1), np.inf)
    paths[0, 0] = 0
    for i in range(l1):
        # print('i', i)
        # for j in range(max(0, i - max(0, r - c) - window + 1), min(c, i + max(0, c - r) + window)):
        j_start = max(0, i - max(0, l1 - l2) - window + 1)
        j_end = min(l2, i + max(0, l2 - l1) + window)
        # print(j_start, j_end)
        for j in range(j_start, j_end):
            # print('j', j)
            # for j in range(1, l2 + 1):
            d = s1[i] - s2[j]
            # print(f'd[{i},{j}] = {d}')
            if weights is not None:
                # print(weights[i, :])
                # multiplication with LeRu like function
                xn3, xn2, xn1, xn0, xp0, xp1, xp2, xp3 = weights[i, :]
                # print('xn1, xn0, xp0, xp1', xn1, xn0, xp0, xp1)
                if d < 0:
                    x0, x1, x2, x3 = xn0, xn1, xn2, xn3
                    d = -d
                else:
                    x0, x1, x2, x3 = xp0, xp1, xp2, xp3
                if d <= x0:
                    d = 0
                elif x0 < d < x1:
                    d *= (d - x0) / (x1 - x0)
                elif x2 <= d:
                    if np.isinf(x3) or x3 == x1:
                        a = 1
                    else:
                        a = 2 / (x3 - x2)
                    d *= (1 + a * (d - x2))
                else:
                    pass  # keep d
            # print('d\'', d)
            cost = d ** 2
            paths[i + 1, j + 1] = cost + min(paths[i, j + 1], paths[i + 1, j], paths[i, j])
    # s = math.sqrt(paths[l1 - 1, l2 - 1])
    paths = np.sqrt(paths)
    s = paths[l1 - 1, l2 - 1]
    return s, paths


def distance_matrix(s, weights, window=None, show_progress=False, **kwargs):
    dist_opts = {
        'window': window
    }
    dists = np.full((len(s), len(s)), np.inf)
    it_r = range(len(s))
    if show_progress:
        it_r = tqdm(it_r)
    for r in it_r:
        it_c = range(r + 1, len(s))
        for c in it_c:
            # Because of the weights this is not symmetric (?)
            # TODO: make function not hardcoded
            # print(f'{r} -- {c}')
            # print(f's[{r}]', s[r])
            # print('weights', weights.get(r, None))
            # print(f's[{c}]', s[c])
            # print('weights', weights.get(c, None))
            weights_r = weights.get(r, None)
            d1, paths = warping_paths(s[r], s[c], weights_r, **dist_opts)
            # print(f'd1(r)={d1}  -- w=\n{weights_r}')
            # print (paths)
            weights_c = weights.get(c, None)
            if weights_r is None and weights_c is None:
                dists[r, c] = d1
            else:
                d2, paths = warping_paths(s[c], s[r], weights_c, **dist_opts)
                # print(f'd2(c)={d2}  -- w=\n{weights_c}')
                # print(paths)
                dists[r, c] = min(d1, d2)

    return dists


def compute_weights_using_dt(series, labels, prototypeidx, classifier=None, savefig=None, **kwargs):
    ml_values, cl_values, clf = series_to_dt(series, labels, prototypeidx, classifier, savefig, **kwargs)

    logger.debug("------")
    weights = compute_weights_from_mlclvalues(series, ml_values, cl_values, only_max=False, strict_cl=True)
    return weights


def series_to_dt(series, labels, prototypeidx, classifier=None, savefig=None, **kwargs):
    """Compute Decision Tree from series

    :param series:
    :param labels: 0 for cannot-link, 1 for must-link
    :param prototypeidx:
    :param classifier: Classifier instance.
        For example dtw_weighted.DecisionTreeClassifier() or tree.DecisionTreeClassifier().
    :param savefig: Path to filename to save tree Graphviz visualisation
    :param kwargs: Passed to warping_paths
    :return:
    """
    features = [[0] * (len(series[prototypeidx]) * 2)]  # feature per idx, split in positive and negative
    targets = [0]  # Do cluster
    ml_values = defaultdict(lambda: ([], []))

    for idx, label in enumerate(labels):
        cur_features = np.zeros(len(series[prototypeidx]) * 2, dtype=np.double)
        cur_features_cnt = np.zeros(len(series[prototypeidx]) * 2, dtype=np.int)
        s, paths = warping_paths(series[prototypeidx], series[idx], None, **kwargs)
        path = best_path(paths)
        for i_to, i_from in path:
            d = series[prototypeidx][i_to] - series[idx][i_from]
            if label == 0:
                # Cannot-link
                pass
            elif label == 1:
                # Must-link
                if d < 0:
                    ml_values[i_to][0].append(-d)
                elif d > 0:
                    ml_values[i_to][1].append(d)
            if d <= 0:
                cur_features[i_to * 2] += series[idx][i_from]
                cur_features_cnt[i_to * 2] += 1
            if d >= 0:
                cur_features[i_to * 2 + 1] += series[idx][i_from]
                cur_features_cnt[i_to * 2 + 1] += 1

        cur_features_cnt[cur_features_cnt == 0] = 1
        cur_features = np.divide(cur_features, cur_features_cnt)
        features.append([series[prototypeidx][i // 2] - cur_features[i] for i in range(len(cur_features))])
        if label == 0:
            targets.append(1)  # Do not cluster
        elif label == 1:
            targets.append(0)  # Do cluster

    if classifier is None:
        clf = DecisionTreeClassifier()
    else:
        clf = classifier

    features = np.array(features)
    targets = np.array(targets)

    clf.fit(features, targets)

    if savefig is not None:
        try:
            from sklearn import tree
            fn = savefig
            feature_names = ["f{} ({})".format(i // 2, i) for i in range(len(series[prototypeidx]) + 1)]
            tree.export_graphviz(clf, out_file=fn, feature_names=feature_names)
        except ImportError:
            logger.error("No figure generated, sklearn is not installed.")

    cl_values = decisiontree_to_clweights(clf)

    return ml_values, cl_values, clf


def decisiontree_to_clweights(clf):
    """Translate a decision tree to a set of cannot-link weights.

    This is based on the concept that we can represent the false (class 1) case as the
    conjunction of all negated false (class 1) leafs. It will be certainly not class 1
    if it is "not any of the leafs that say class 1" and thus "!(leaf_1 v leaf_2 v ... v leaf_n)"
    which is "!leaf_1 ^ !leaf_2 ^ ... ^ !leaf_n".

    :param clf: DecisionTreeClassifier
    :returns: Weights
    """
    dtnodes = deque([(0, [])])
    cl_values = defaultdict(lambda: ([], []))

    while len(dtnodes) > 0:
        curnode, path = dtnodes.popleft()
        if clf.tree_.children_left[curnode] == -1 and clf.tree_.children_right[curnode] == -1:
            value = clf.tree_.value[curnode][0]
            # logger.debug(f"Leaf - values = {value}")
            if value[0] == 0:
                # Leaf that represents cannot-link
                # logger.debug(f'CL leaf: {curnode}')
                clweights_updatefrompath(cl_values, path)
            # elif value[1] == 0:
            #     logger.debug(f'ML leaf: {curnode}')
            # else:
            #     logger.debug(f'Non pure leaf: {curnode}')
        else:
            threshold = clf.tree_.threshold[curnode]
            feature = clf.tree_.feature[curnode]
            path_left = path + [(feature, threshold, True)]  # true branch (f <= t)
            dtnodes.append((clf.tree_.children_left[curnode], path_left))
            path_right = path + [(feature, threshold, False)]  # false branch (f > t)
            dtnodes.append((clf.tree_.children_right[curnode], path_right))
    return cl_values


def clweights_updatefrompath(cl_values, path):
    logger.debug(f"Path to CL")
    for feature, threshold, leq in path:
        index = feature // 2
        dneg = ((feature % 2) == 0)
        if leq:  # f <= t
            if threshold < 0:
                logger.debug(f"Accept: CL with f{index} <= {threshold} (d is negative={dneg}, feature={feature})")
                cl_values[index][0].append(-threshold)
            else:
                logger.debug(f"Ignore: CL with f{index} <= {threshold} (d is negative={dneg}, feature={feature})")
        else:  # f > t
            if threshold > 0:
                logger.debug(f"Accept: CL with f{index} > {threshold} (d is negative={dneg}, feature={feature})")
                cl_values[index][1].append(threshold)
            else:
                logger.debug(f"Ignore: CL with f{index} > {threshold} (d is negative={dneg}, feature={feature})")


def compute_weights_from_mlclvalues(serie, ml_values, cl_values, only_max=False, strict_cl=True):
    """Compute weights that represent a rectifier type of function.

         |             /
        3|            +
         |           /
         |          /
        1|      +--+
         |     /
        0+----+
         0   x0 x1 x2 x3

    """
    logger.debug('ml_values = ' + 'None' if ml_values is None else str(ml_values.items()))
    logger.debug('cl_values = ' + 'None' if cl_values is None else str(cl_values.items()))

    s1 = serie
    wn = np.zeros((len(s1), 8), dtype=np.double)  # xns, xn2, xn1, xn0, xp0, xp1, xp2, xps
    wn[:, 0:2] = np.inf
    wn[:, 6:8] = np.inf

    # First find ml-max and cl-min values
    ml_maxmin_n = np.zeros((len(s1), 3))
    ml_maxmin_p = np.zeros((len(s1), 3))
    for idx in range(len(s1)):
        try:
            mls = ml_values[idx][0]
        except TypeError:
            mls = []
        try:
            cls = cl_values[idx][0]
        except TypeError:
            cls = []
        ml_max = _clean_max(mls, cls)
        cl_min = _clean_min(cls, mls, strict_cl)
        if np.isinf(cl_min):
            diff = 0
        else:
            diff = cl_min - ml_max
        ml_maxmin_n[idx, :] = [ml_max, cl_min, diff]

        try:
            mls = ml_values[idx][1]
        except TypeError:
            mls = []
        try:
            cls = cl_values[idx][1]
        except TypeError:
            cls = []
        ml_max = _clean_max(mls, cls)
        cl_min = _clean_min(cls, mls, strict_cl)
        if np.isinf(cl_min):
            diff = 0
        else:
            diff = cl_min - ml_max
        ml_maxmin_p[idx, :] = [ml_max, cl_min, diff]

    if only_max:
        # Only retain the cl-min value where the distance between ml_max and cl_min is largest
        maxval = np.max(ml_maxmin_n[:, 2])
        maxidx = ml_maxmin_n[:, 2] == maxval
        # print('maxval', maxval)
        # print('maxidx', maxidx)
        # print('values', ml_maxmin_n)
        vals = ml_maxmin_n[:, 1][maxidx]
        ml_maxmin_n[:, 1] = np.inf
        ml_maxmin_n[:, 1][maxidx] = vals

        maxval = np.max(ml_maxmin_p[:, 2])
        maxidx = ml_maxmin_p[:, 2] == maxval
        # print('maxval', maxval)
        # print('maxidx', maxidx)
        # print('values', ml_maxmin_p)
        vals = ml_maxmin_p[:, 1][maxidx]
        ml_maxmin_p[:, 1] = np.inf
        ml_maxmin_p[:, 1][maxidx] = vals

    for idx in range(len(s1)):
        # vn3, vn2, vn1, vn0, vp0, vp1, vp2, vp3 = wn[idx]
        # Negative
        vn1 = 1.5 * ml_maxmin_n[idx, 0]
        vn3 = ml_maxmin_n[idx, 1]
        if vn1 > vn3:
            vn1 = vn3
        vn0 = 0.5 * vn1
        vn2 = 0.9 * vn3
        if vn2 < vn1:
            vn1 = vn2 = (vn1 + vn2) / 2
        # Positive
        vp1 = 1.5 * ml_maxmin_p[idx, 0]
        vp3 = ml_maxmin_p[idx, 1]
        if vp1 > vp3:
            vp1 = vp3
        vp0 = 0.5 * vp1
        vp2 = 0.9 * vp3
        if vp2 < vp1:
            vp1 = vp2 = (vp1 + vp2) / 2
        wn[idx, :] = [vn3, vn2, vn1, vn0, vp0, vp1, vp2, vp3]
    # logger.debug(f'prototype_weights[{si}]:\n{wn}')
    return wn


def _clean_max(mls, cls):
    mls.sort()
    cls.sort()
    if cls is None or len(cls) == 0:
        min_cls = np.inf
    else:
        min_cls = cls[0]
    max_mls = 0
    for ml in mls:
        if ml > min_cls:
            return max_mls
        if ml > max_mls:
            max_mls = ml
    return max_mls


def _clean_min(cls, mls, keep_largest=True):
    mls.sort()
    cls.sort()
    min_cls = np.inf
    if mls is None or len(mls) == 0:
        max_mls = 0
    else:
        max_mls = mls[-1]
    for cl in reversed(cls):
        if cl < max_mls:
            break
        if cl < min_cls:
            min_cls = cl
    if keep_largest:
        # If None of the cl values are kept, use the largest value.
        if np.isinf(min_cls) and cls is not None and len(cls) > 0:
            min_cls = cls[-1]
    return min_cls


def plot_margins(serie, weights, filename=None, ax=None, origin=(0, 0), scaling=(1, 1), y_limit=None):
    if weights is None:
        return
    if y_limit is None:
        y_limit = (np.min(serie), np.max(serie))
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1)

    logger.debug(f"Weights =\n{weights}")
    #  |             /
    # 3|            +
    #  |           /
    #  |          /
    # 1|      +--+
    #  |     /
    # 0+----+
    #  0   x0 x1 x2 x3
    xn3 = weights[:, 0]
    xn2 = weights[:, 1]
    xn1 = weights[:, 2]
    xn0 = weights[:, 3]
    xp0 = weights[:, 4]
    xp1 = weights[:, 5]
    xp2 = weights[:, 6]
    xp3 = weights[:, 7]

    # Green, between 0 and xn0
    upper_bound_0 = [origin[1] + scaling[1] * (y + w) for y, w in zip(serie, xn0)]
    lower_bound_0 = [origin[1] + scaling[1] * (y - w) for y, w in zip(serie, xp0)]
    ax.fill_between(range(len(serie)), lower_bound_0, upper_bound_0, facecolor='green', alpha=0.1)

    # Blue, between xn0 and xn1
    upper_bound_1 = [origin[1] + scaling[1] * (y + w) for y, w in zip(serie, xn1)]
    lower_bound_1 = [origin[1] + scaling[1] * (y - w) for y, w in zip(serie, xp1)]
    upper_bound_1 = np.array(upper_bound_1)
    lower_bound_1 = np.array(lower_bound_1)
    lower_bound_1[lower_bound_1 < (origin[1] + y_limit[0])] = origin[1] + y_limit[0]
    upper_bound_1[upper_bound_1 > (origin[1] + y_limit[1])] = origin[1] + y_limit[1]
    ax.fill_between(range(len(serie)), lower_bound_0, lower_bound_1, facecolor='blue', alpha=0.1)
    ax.fill_between(range(len(serie)), upper_bound_0, upper_bound_1, facecolor='blue', alpha=0.1)

    # Cyan , between xn1 and xn2
    xn2[np.isinf(xn2)] = 0
    np.maximum(xn1, xn2, xn2)
    xp2[np.isinf(xp2)] = 0
    np.maximum(xp1, xp2, xp2)
    upper_bound_2 = [origin[1] + scaling[1] * (y + w) for y, w in zip(serie, xn2)]
    lower_bound_2 = [origin[1] + scaling[1] * (y - w) for y, w in zip(serie, xp2)]
    upper_bound_2 = np.array(upper_bound_2)
    lower_bound_2 = np.array(lower_bound_2)
    lower_bound_2[lower_bound_2 < (origin[1] + y_limit[0])] = origin[1] + y_limit[0]
    upper_bound_2[upper_bound_2 > (origin[1] + y_limit[1])] = origin[1] + y_limit[1]
    ax.fill_between(range(len(serie)), lower_bound_1, lower_bound_2, facecolor='cyan', alpha=0.1)
    ax.fill_between(range(len(serie)), upper_bound_1, upper_bound_2, facecolor='cyan', alpha=0.1)

    # Red, between xn2 and xn3
    xn3 *= 100
    xp3 *= 100
    xn3[np.isinf(xn3)] = 0
    xp3[np.isinf(xp3)] = 0
    np.maximum(xn2, xn3, xn3)
    np.maximum(xp2, xp3, xp3)
    upper_bound_3 = [origin[1] + scaling[1] * (y + w) for y, w in zip(serie, xn3)]
    lower_bound_3 = [origin[1] + scaling[1] * (y - w) for y, w in zip(serie, xp3)]
    # print(upper_bound_3)
    # print(self.ts_height)
    upper_bound_3 = np.array(upper_bound_3)
    lower_bound_3 = np.array(lower_bound_3)
    lower_bound_3[lower_bound_3 < (origin[1] + y_limit[0])] = origin[1] + y_limit[0]
    upper_bound_3[upper_bound_3 > (origin[1] + y_limit[1])] = origin[1] + y_limit[1]
    ax.fill_between(range(len(serie)), lower_bound_2, lower_bound_3, facecolor='red', alpha=0.1)
    ax.fill_between(range(len(serie)), upper_bound_2, upper_bound_3, facecolor='red', alpha=0.1)

    # Plot series
    ax.plot(origin[1] + scaling[1] * serie)

    if filename:
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    return ax


class DecisionTreeClassifier:
    def __init__(self):
        """Decision Tree that uses a combination of Information Gain and k-distance.
        It also offers the option to use every feature only once.


        Note:
        - This classifier assumes only two classes: 0 and 1.
        """
        self.tree_ = None
        self.criterion = "igkdistance"

    @staticmethod
    def entropy(targets):
        """H(X) = -\sum_{i=0}^{n}(P(x_i))\log P(x_i))

        :param targets: Numpy array with class target (values are 0 and 1)
        :return: entropy
        """
        # print(f'entropy({targets})')
        n = len(targets)
        if n == 0:
            raise Exception("Empty list passed to entropy")
        prob = np.sum(targets) / n
        if prob == 0 or prob == 1:
            h = 0
        else:
            h = - prob * math.log(prob) - (1.0 - prob) * math.log(1.0 - prob)
        # print(f'entropy - targets: {targets} - h: {h}')
        return h

    @staticmethod
    def entropy_continuous(targets, values):
        """Best split based on information gain"""
        # print(f'entropy_continuous:\n{targets}\n{values}')
        h0 = DecisionTreeClassifier.entropy(targets)
        # print(f'h0={h0}')

        thresholds = np.unique(values)
        thresholds = (thresholds[1:] + thresholds[:-1]) / 2
        n = len(values)
        h1_min, th_min = np.inf, None
        for threshold in thresholds:
            mask = (values <= threshold)
            prob = np.sum(mask) / n
            h1 = prob * DecisionTreeClassifier.entropy(targets[mask]) + \
                 (1.0 - prob) * DecisionTreeClassifier.entropy(targets[~mask])
            # print(f'th={threshold} - h1={h1} - prob={prob} - ig={h0 - h1}')
            if h1 < h1_min:
                h1_min = h1
                th_min = threshold
                # No need to do IGR, it always splits in two
                # ig = h0 - h1
                # if prob == 0.0 or prob == 1.0:
                #     iv = 0
                # else:
                #     iv = -prob * math.log(prob, 2) - (1.0 - prob) * math.log(1.0 - prob, 2)
                # if iv > iv_max:
                #     iv_max = iv
                #     th_max = threshold
        if th_min is None:
            ig = 0
        else:
            ig = h0 - h1_min
        return ig, th_min

    @staticmethod
    def kdistance(values, threshold, k=5):
        """k-distance. Can be used as a measure for density."""
        # print(f'kdistance({values}, {threshold})')
        dists = []
        for value in np.nditer(values):
            dist = abs(value - threshold)
            if len(dists) < k:
                dists.append(dist)
                dists.sort()
            elif dist < dists[-1]:
                dists[-1] = dist
                dists.sort()
        # Sometimes k-distance is defined as:
        # dk = 1/k*sum([d**2 for d in dists])
        return dists[-1]

    def fit(self, features, targets, use_feature_once=True):
        """Learn decision tree.

        :param features: Array of dimension #instances x #features, type float
        :param targets: Array of dimension #instances, values 0 or 1
        :param use_feature_once: Use each feature only once in a path
        :return: None

        Fills the tree_ variable with a decision tree representation.
        """
        # print(f'features:\n{features}')
        # print(f'targets:\n{targets}')
        nb_features = features.shape[1]
        nb_instances = features.shape[0]
        self.tree_ = Tree()
        queue = deque([(self.tree_.last(),  # Leaf
                        np.zeros(nb_features, dtype=bool),  # Used features
                        np.ones(nb_instances, dtype=bool))])  # Instances mask
        queue_it = 0
        while len(queue) > 0:
            queue_it += 1
            node, used_ftrs, idxs = queue.popleft()
            # print(f'------ node ({queue_it})\n  {node}\n  {used_ftrs}\n  {idxs}')
            self.tree_.value[node][0, 1] = np.sum(targets[idxs])
            self.tree_.value[node][0, 0] = np.sum(idxs) - self.tree_.value[node][0, 1]
            if np.all(targets[idxs]) or not np.any(targets[idxs]):
                # print('Pure leaf')
                continue
            curvalues = features[idxs, :]
            curtargets = targets[idxs]
            best_gain, best_fi, best_thr = 0, None, None
            for fi in range(nb_features):
                if use_feature_once and used_ftrs[fi]:
                    continue
                ig, thr = self.entropy_continuous(curtargets, curvalues[:, fi])
                if thr is None:
                    kd = None
                    gain = 0.0
                else:
                    kd = self.kdistance(curvalues[:, fi], thr)  # Prefer values in low-density regions (thus large k dist)
                    gain = ig * kd
                # print(f'fi={fi}, thr={thr}, ig={ig}, kd={kd}, gain={gain}')
                if best_gain < gain:
                    best_gain = gain
                    best_fi = fi
                    best_thr = thr
            if best_fi is not None:
                used_ftrs = used_ftrs.copy()
                used_ftrs[best_fi] = True
                self.tree_.feature[node] = best_fi
                self.tree_.threshold[node] = best_thr
                lessorequal = self.tree_.add()
                # print(f'best_fi={best_fi}, best_thr={best_thr}, best_gain={best_gain}')
                leq_idxs = idxs & (features[:, best_fi] <= best_thr)
                queue.append((lessorequal, used_ftrs, leq_idxs))
                self.tree_.children_left[node] = lessorequal
                larger = self.tree_.add()
                larger_idxs = idxs & (features[:, best_fi] > best_thr)
                queue.append((larger, used_ftrs, larger_idxs))
                self.tree_.children_right[node] = larger
                # print('New queue:')
                # print('\n'.join([str(elmt) for elmt in queue]))


class Tree:
    def __init__(self):
        """Tree to represent a Decision Tree.

        This datastructure has the fields required to be compatible with scikit-learn and
        its Graphvic DOT representation.
        """
        self.threshold = []
        self.feature = []
        self.children_right = []
        self.children_left = []
        self.n_outputs = 1
        self.value = []
        self.impurity = []
        self.n_node_samples = []
        self.n_classes = []
        self.add()

    def add(self):
        """Add a new node to the tree and return its index."""
        self.threshold.append(-1)
        self.feature.append(-1)
        self.children_right.append(-1)
        self.children_left.append(-1)
        self.value.append(np.array([[0, 0]]))
        self.impurity.append(-1)
        self.n_node_samples.append(-1)
        self.n_classes.append(-1)
        return len(self.feature) - 1

    def last(self):
        return len(self.feature) - 1

    @property
    def nb_nodes(self):
        return len(self.threshold)
