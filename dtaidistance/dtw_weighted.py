# -*- coding: UTF-8 -*-
"""
dtaidistance.dtw_weighted
~~~~~~~~~~~~~~~~~~~~~~~~~

Dynamic Time Warping (DTW) with custom internal distance function.

:author: Wannes Meert
:copyright: Copyright 2018 KU Leuven, DTAI Research Group.
:license: Apache License, Version 2.0, see LICENSE for details.

Weights are represented using a tuple (-x3, -x2, -x1, -x0, x0, x1, x2, x3).
The distance d, used in DTW, is multiplied with factor w(d):

.. code-block::

    ^
w(d)|
    |             /
   3|            +
    |           /
    |          /
   1|      +--+
    |     /
   0+----+--------------->
    0   x0 x1 x2 x3     d

The negative and positive values are used to make a distinction between negative
and postive distances. Thus to differentiate between the case that the function
compared with is higher or lower than the target function.

"""
import logging
import math
from collections import defaultdict, deque
import io

from .dtw import best_path
from .util_numpy import NumpyException


logger = logging.getLogger("be.kuleuven.dtai.distance")

try:
    import numpy as np
except ImportError:
    np = None

try:
    from tqdm import tqdm
except ImportError:
    logger.info('tqdm library not available')
    tqdm = None


def warping_paths(s1, s2, weights=None, window=None, **_kwargs):
    """
    Input: s1 and s2 are time series of length n/l1 and m/l2

    :param s1:
    :param s2:
    :param weights: Weights on s1
    :return: DTW similarity m between s1 and s2, warping paths matrix
    """
    if np is None:
        raise NumpyException('The function warping_paths requires numpy.')
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


def compute_weights_using_dt(series, labels, prototypeidx, **kwargs):
    """Compute weight array by learning an ensemble of Decision Trees representing
    the differences between the different labels.

    :param series: List of sequences
    :param labels: Labels for series
    :param prototypeidx: The target sequence to learn weights for
    :param kwargs: Arguments to pass to `series_to_dt`
    :return:
    """
    ml_values, cl_values, _clfs, importances = series_to_dt(series, labels, prototypeidx, **kwargs)
    weights = compute_weights_from_mlclvalues(series[prototypeidx], ml_values, cl_values, **kwargs)
    return weights, importances


def series_to_dt(series, labels, prototypeidx, classifier=None, max_clfs=None, min_ig=0,
                 savefig=None, warping_paths_fnc=None, ignore_idxs=None, patternlen=None,
                 min_purity=1.0, **kwargs):
    """Compute Decision Tree from series

    :param series:
    :param labels: 0 for cannot-link, 1 for must-link
    :param prototypeidx:
    :param classifier: Classifier class.
        For example dtw_weighted.DecisionTreeClassifier or tree.DecisionTreeClassifier
    :param max_clfs: Maximum number of classifiers to learn
    :param min_ig: Minimum information gain
    :param savefig: Path to filename to save tree Graphviz visualisation
    :param warping_paths_fnc: Function to compute warping paths
    :param patternlen: Pattern window size (default None)
    :param kwargs: Passed to warping_paths_fnc
    :return:
    """
    if warping_paths_fnc is None:
        warping_paths_fnc = warping_paths
    if ignore_idxs is None:
        ignore_idxs = set()
    features = [[0] * (len(series[prototypeidx]) * 2)]  # feature per idx, split in positive and negative
    targets = [0]  # Do cluster
    ml_values = defaultdict(lambda: ([], []))

    # print(f"prototype idx = {prototypeidx}")
    for idx, label in enumerate(labels):
        if idx in ignore_idxs:
            continue
        cur_features = np.zeros(len(series[prototypeidx]) * 2, dtype=np.double)
        cur_features_cnt = np.zeros(len(series[prototypeidx]) * 2, dtype=np.int)
        wp_params = {key: kwargs[key] for key in {'window', 'max_dist', 'max_step', 'max_length_diff',
                     'penalty', 'psi'}.intersection(kwargs.keys())}
        s, paths = warping_paths_fnc(series[prototypeidx], series[idx], **wp_params)
        path = best_path(paths)
        for i_to, i_from in path:
            d = series[prototypeidx][i_to] - series[idx][i_from]
            # print(f"d{idx}({i_to},{i_from}) = {d}")
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
                cur_features[i_to * 2] += -d
                cur_features_cnt[i_to * 2] += 1
            if d >= 0:
                cur_features[i_to * 2 + 1] += d
                cur_features_cnt[i_to * 2 + 1] += 1

        cur_features_cnt[cur_features_cnt == 0] = 1
        cur_features = np.divide(cur_features, cur_features_cnt)
        features.append(cur_features)
        if label == 0:
            # print(f"{idx:<2}: CL -> {cur_features[0]:0.3f} / {cur_features[1]:0.3f}")
            targets.append(1)  # Do not cluster
        elif label == 1:
            # print(f"{idx:<2}: ML -> {cur_features[0]:0.3f} / {cur_features[1]:0.3f}")
            targets.append(0)  # Do cluster
        else:
            raise Exception("Encountered a label that is not 0 (cannot-link) or 1 (must-link): {}".format(label))

    if classifier is None:
        classifier = DecisionTreeClassifier

    features = np.array(features)
    targets = np.array(targets)

    if savefig is not None:
        try:
            from sklearn import tree
        except ImportError:
            logger.error("No figure generated, sklearn is not installed.")
            savefig, tree, out_string, feature_names = None, None, None, None
        out_string = io.StringIO()

        def args(i):
            if (i % 2) == 0:
                sgn = '-'
                cmp = 's>t'
            else:
                sgn = '+'
                cmp = 's<t'
            return i, sgn, cmp

        feature_names = ["d[{}] ({}, {}, {})".format(i // 2, *args(i))
                         for i in range(2*len(series[prototypeidx]))]
        class_names = ["ML", "CL"]
    else:
        tree, out_string, feature_names, class_names = None, None, None, None

    if patternlen is not None:
        cl_values, clfs, importances = dt_windows(features, targets, classifier, patternlen, max_clfs, min_ig, min_purity)
    else:
        cl_values, clfs, importances = dt_onewindow(features, targets, classifier, max_clfs, min_ig, min_purity)

    if savefig is not None:
        for clf in clfs:
            tree.export_graphviz(clf, out_file=out_string, feature_names=feature_names, class_names=class_names)
            print("\n\n", file=out_string)
        with open(savefig, "w") as ofile:
            print(out_string.getvalue(), file=ofile)

    return ml_values, cl_values, clfs, importances


def dt_windows(features, targets, classifier, patternlen, max_clfs, min_ig, min_purity):
    cl_values = dict()
    clfss = []
    ignore_features = set()
    clf_w = 1.0
    importances = defaultdict(lambda: [0, 0])

    nb_features = features.shape[1]
    nb_indices = int((nb_features / 2))
    max_kd = 0
    for idx in range(0, int(nb_indices - patternlen / 2), int(patternlen / 2)):
        idx_s = idx * 2
        idx_e = idx_s + patternlen * 2
        clf = classifier()
        cur_features = features[:, idx_s:idx_e]
        clf.fit(cur_features, targets, ignore_features=ignore_features, min_ig=min_ig)
        # logger.debug(f"Learned classifier {len(clfss) + 1}: idx = f{idx}/{idx_s}:f{idx+patternlen}/{idx_e}, "
        #              f"nb nodes = {clf.tree_.nb_nodes}, used features = {clf.tree_.used_features}")
        if clf.tree_.nb_nodes <= 1:
            continue
        clf.set_features(list(range(idx_s, idx_e)))
        max_kd = max(max_kd, np.max(clf.tree_.kd))
        clfss.append(clf)
    clfs = []
    for clf_idx, clf in enumerate(clfss):
        score = clf.score(max_kd)
        # logger.debug(f"Clf[{clf_idx:<2}] - Score = {score}, Entropy = {clf.avg_impurity()}, "
        #              f"depth = {clf.tree_.depth}, nbnodes = {clf.tree_.nb_nodes}")
        clfs.append((score, -clf.tree_.nb_nodes, clf))
    clfs.sort(reverse=True)
    min_score = clfs[-1][0]
    max_score = clfs[0][0]
    minallowed_score = max_score - (max_score - min_score) / 5  # TODO: remove this magic number
    clfs_use = 1
    # We expect that the current range includes at least two new classifiers
    while len(clfs) > clfs_use and minallowed_score > clfs[clfs_use][0]:
        max_score = clfs[clfs_use][0]
        minallowed_score = max_score - (max_score - min_score) / 5
        clfs_use += 1
    clfs = [t for t in clfs if t[0] >= minallowed_score]

    if max_clfs is not None:
        clfs = clfs[:max_clfs]
    # logger.debug(f"Kept {len(clfs)} classifiers with score >= {minallowed_score}, clfs_use = {clfs_use}")
    for clf_score, clf_nbnodes, clf in clfs:
        new_cl_values, used_features = decisiontree_to_clweights(clf, min_purity)
        # if len(used_features) == 0:
        #     logger.debug(f"No features used, ignore all features in tree: {clf.tree_.used_features}")
        #     used_features.update(clf.tree_.used_features)
        update_cl_values(cl_values, new_cl_values)
        update_importances(importances, new_cl_values, (clf_score - min_score) / (max_score - min_score))
        # print(f"new_cl_values: {new_cl_values}")
        # print(f"cl_values: {cl_values}")
        # ignore_features.update(clf.tree_.used_features)
        ignore_features.update(used_features)
        # print(f"ignore_features: {ignore_features}")
    return cl_values, clfs, importances


def dt_onewindow(features, targets, classifier, max_clfs, min_ig, min_purity):
    cl_values = dict()
    clfs = []
    ignore_features = set()
    not_empty = True
    clf_w = 1.0
    importances = defaultdict(lambda: [0, 0])
    while not_empty and not (max_clfs is not None and len(clfs) >= max_clfs):
        clf = classifier()
        clf.fit(features, targets, ignore_features=ignore_features, min_ig=min_ig)
        # logger.debug(f"Learned classifier {len(clfs) + 1}: nb nodes = {clf.tree_.nb_nodes}")
        if clf.tree_.nb_nodes <= 1:
            not_empty = False
            continue
        clfs.append(clf)
        new_cl_values, used_features = decisiontree_to_clweights(clf, min_purity)
        if len(used_features) == 0:
            # logger.debug(f"No features used, ignore all features in tree: {clf.tree_.used_features}")
            used_features.update(clf.tree_.used_features)
        update_cl_values(cl_values, new_cl_values)
        update_importances(importances, new_cl_values, clf_w)
        # print(f"new_cl_values: {new_cl_values}")
        # print(f"cl_values: {cl_values}")
        # ignore_features.update(clf.tree_.used_features)
        ignore_features.update(used_features)
        # print(f"ignore_features: {ignore_features}")
        clf_w *= 0.66
    return cl_values, clfs, importances


def update_cl_values(cl_values, new_cl_values):
    for idx, (n, p) in new_cl_values.items():
        if idx not in cl_values:
            cl_values[idx] = [n, p]
        else:
            cl_values[idx][0].extend(n)
            cl_values[idx][1].extend(p)


def update_importances(importances, new_cl_values, weight):
    for idx, (n, p) in new_cl_values.items():
        if len(n) > 0:
            importances[idx][0] = max(weight, importances[idx][0])
        if len(p) > 0:
            importances[idx][1] = max(weight, importances[idx][1])


def decisiontree_to_clweights(clf, min_purity=1.0):
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
    used_features = set()

    while len(dtnodes) > 0:
        curnode, path = dtnodes.popleft()
        if clf.tree_.children_left[curnode] == -1 and clf.tree_.children_right[curnode] == -1:
            value = clf.tree_.value[curnode][0]
            # logger.debug(f"Leaf - values = {value}")
            purity = value[1] / (value[0] + value[1])
            # print(f"purity: {purity}, values: {value}")
            assert((value[0] == 0 and purity == 1.0) or purity != 1.0)
            if purity >= min_purity:
                # Leaf that represents cannot-link with given purity
                # logger.debug(f'CL leaf: {curnode}')
                cur_used_features = clweights_updatefrompath(cl_values, path)
                used_features.update(cur_used_features)
        else:
            threshold = clf.tree_.threshold[curnode]
            feature = clf.tree_.feature[curnode]
            path_left = path + [(feature, threshold, True)]  # true branch (f <= t)
            dtnodes.append((clf.tree_.children_left[curnode], path_left))
            path_right = path + [(feature, threshold, False)]  # false branch (f > t)
            dtnodes.append((clf.tree_.children_right[curnode], path_right))
    return cl_values, used_features


def clweights_updatefrompath(cl_values, path):
    # logger.debug(f"Path to CL: {path}")
    used_features = set()
    for feature, threshold, leq in path:
        index = feature // 2
        dneg = ((feature % 2) == 0)
        if leq:  # f <= t
            # logger.debug(f"Ignore: CL with f{index} <= {threshold} (d is negative={dneg}, feature={feature})")
            pass
        else:  # f > t
            # logger.debug(f"Accept: CL with f{index} >  {threshold} (d is negative={dneg}, feature={feature})")
            cl_values[index][0 if dneg else 1].append(threshold)
            used_features.add(feature)
    return used_features


def compute_weights_from_mlclvalues(serie, ml_values, cl_values, only_max=False, strict_cl=True, **_kwargs):
    """Compute weights that represent a rectifier type of function based on must-link values and cannot-link values.

         |             /
        3|            +
         |           /
         |          /
        1|      +--+
         |     /
        0+----+
         0   x0 x1 x2 x3

    """
    if __debug__ and logger.isEnabledFor(logging.DEBUG):
        logger.debug('ml_values = ' + ('None' if ml_values is None else str(list(ml_values.items()))))
        logger.debug('cl_values = ' + ('None' if cl_values is None else str(list(cl_values.items()))))

    wn = np.zeros((len(serie), 8), dtype=np.double)  # xns, xn2, xn1, xn0, xp0, xp1, xp2, xps
    wn[:, 0:2] = np.inf
    wn[:, 6:8] = np.inf

    # First find ml-max and cl-min values
    ml_maxmin_n = np.zeros((len(serie), 3))
    ml_maxmin_p = np.zeros((len(serie), 3))
    for idx in range(len(serie)):
        if idx in ml_values:
            mls = ml_values[idx][0]
        else:
            mls = []
        if idx in cl_values:
            cls = cl_values[idx][0]
        else:
            cls = []
        ml_max = _clean_max(mls, cls)
        cl_min = _clean_min(cls, mls, strict_cl)
        if np.isinf(cl_min):
            diff = 0
        else:
            diff = cl_min - ml_max
        ml_maxmin_n[idx, :] = [ml_max, cl_min, diff]

        if idx in ml_values:
            mls = ml_values[idx][1]
        else:
            mls = []
        if idx in cl_values:
            cls = cl_values[idx][1]
        else:
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

    for idx in range(len(serie)):
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
    # logger.debug(f'weights:\n{wn}')
    return wn


def _clean_max(mls, cls):
    """Return the maximal value of mls that is smaller than all values in cls."""
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
    """Return the minimal value of cls that is larger than all values in mls."""
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


def plot_margins(serie, weights, filename=None, ax=None, origin=(0, 0), scaling=(1, 1), y_limit=None,
                 importances=None):
    try:
        from matplotlib import pyplot as plt
    except ImportError:
        print("Matplotlib not installed")
        return
    if weights is None:
        return
    if y_limit is None:
        y_limit = (np.min(serie), np.max(serie))
        diff = y_limit[1] - y_limit[0]
        y_limit = (y_limit[0] - diff * 0.2, y_limit[1] + diff * 0.2)
    # logger.debug(f"y_limit = {y_limit}")
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1)
    else:
        fig = None

    # logger.debug(f"Weights =\n{weights}")

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
    ax.fill_between(range(len(serie)), lower_bound_0, upper_bound_0, facecolor='green', alpha=0.2)

    # Dark Green, between x0 and x1
    upper_bound_1 = np.array([origin[1] + scaling[1] * (y + w) for y, w in zip(serie, xn1)])
    lower_bound_1 = np.array([origin[1] + scaling[1] * (y - w) for y, w in zip(serie, xp1)])
    lower_bound_1[lower_bound_1 < (origin[1] + y_limit[0])] = origin[1] + y_limit[0]
    upper_bound_1[upper_bound_1 > (origin[1] + y_limit[1])] = origin[1] + y_limit[1]
    ax.fill_between(range(len(serie)), lower_bound_0, lower_bound_1, facecolor='green', alpha=0.1)
    ax.fill_between(range(len(serie)), upper_bound_0, upper_bound_1, facecolor='green', alpha=0.1)

    # Cyan , between x1 and x2
    # xn2[np.isinf(xn2)] = abs(y_limit[0])
    # xp2[np.isinf(xp2)] = abs(y_limit[1])
    upper_bound_2 = np.array([origin[1] + scaling[1] * (y + w) for y, w in zip(serie, xn2)])
    lower_bound_2 = np.array([origin[1] + scaling[1] * (y - w) for y, w in zip(serie, xp2)])
    upper_bound_2[upper_bound_2 > (origin[1] + y_limit[1])] = origin[1] + y_limit[1]
    lower_bound_2[lower_bound_2 < (origin[1] + y_limit[0])] = origin[1] + y_limit[0]
    # ax.fill_between(range(len(serie)), lower_bound_1, lower_bound_2, facecolor='cyan', alpha=0.1)
    # ax.fill_between(range(len(serie)), upper_bound_1, upper_bound_2, facecolor='cyan', alpha=0.1)

    # Red, between x2 and x3
    # xn3[np.isinf(xn3)] = abs(y_limit[1])
    # xp3[np.isinf(xp3)] = abs(y_limit[0])
    upper_bound_3 = np.array([origin[1] + scaling[1] * (y + w) for y, w in zip(serie, xn3)])
    lower_bound_3 = np.array([origin[1] + scaling[1] * (y - w) for y, w in zip(serie, xp3)])
    upper_bound_3[upper_bound_3 > (origin[1] + y_limit[1])] = origin[1] + y_limit[1]
    lower_bound_3[lower_bound_3 < (origin[1] + y_limit[0])] = origin[1] + y_limit[0]
    ax.fill_between(range(len(serie)), lower_bound_2, lower_bound_3, facecolor='red', alpha=0.1)
    ax.fill_between(range(len(serie)), upper_bound_2, upper_bound_3, facecolor='red', alpha=0.1)

    # Dark Red, between x3 and limit
    lower_bound_4 = np.array([origin[1] + y_limit[0] for y in serie])
    upper_bound_4 = np.array([origin[1] + y_limit[1] for y in serie])
    ax.fill_between(range(len(serie)), lower_bound_3, lower_bound_4, facecolor='red', alpha=0.2)
    ax.fill_between(range(len(serie)), upper_bound_3, upper_bound_4, facecolor='red', alpha=0.2)

    # Importances
    if importances is not None:
        for idx, (imp_n, imp_p) in importances.items():
            if imp_n != 0:
                y = origin[1] + y_limit[1]
                alpha = (imp_n + 0.2) / 1.2
                ax.plot([idx], [y], marker='o', color='red', alpha=alpha)
            if imp_p != 0:
                y = origin[1] + y_limit[0]
                alpha = (imp_p + 0.2) / 1.2
                ax.plot([idx], [y], marker='o', color='red', alpha=alpha)

    # Plot series
    ax.plot(origin[1] + scaling[1] * serie)

    if filename:
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close()
        fig, ax = None, None
    return fig, ax


class DecisionTreeClassifier:
    def __init__(self):
        """Decision Tree that uses a combination of Information Gain and k-distance.
        It also offers the option to use every feature only once.


        Note:
        - This classifier assumes only two classes: 0 and 1.
        """
        self.tree_ = None
        self.criterion = "igkdistance"
        self.n_features_ = None

    @staticmethod
    def entropy(targets):
        """H(X) = -\\sum_{i=0}^{n}(P(x_i))\\log P(x_i))

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
    def informationgain_continuous(targets, values, h0=None):
        """Best split based on information gain.

        :return: (information gain, threshold value, entropy of current node)
        """
        # print(f'entropy_continuous:\n{targets}\n{values}')
        if h0 is None:
            h0 = DecisionTreeClassifier.entropy(targets)
        # print(f'h0={h0}, targets={targets}')

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
        return ig, th_min, h0

    @staticmethod
    def kdistance(values, threshold, k=5):
        """k-distance density measure .

        :param values:
        :param threshold:
        :param k: k-th element to return distance of
        :return: Distances to k nearest neighbours
        """
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

    def fit(self, features, targets, use_feature_once=True, ignore_features=None, min_ig=0):
        """Learn decision tree.

        :param features: Array of dimension #instances x #features, type float
        :param targets: Array of dimension #instances, values 0 or 1
        :param use_feature_once: Use each feature only once in a path
        :param ignore_features: Set of feature indices to ignore
        :param min_ig: Minimal information gain
        :return: DecisionTreeClassifier object (self)

        Fills the tree_ variable with a decision tree representation.
        """
        # print(f'features:\n{features}')
        # print(f'targets:\n{targets}')
        nb_features = features.shape[1]
        self.n_features_ = nb_features
        nb_instances = features.shape[0]
        # print(f'nb_instances: {nb_instances} (targets.shape = {targets.shape}')
        k = int(math.ceil(len(targets) * 0.005))
        k_relative = float(np.max(features))
        self.tree_ = Tree()
        queue = deque([(self.tree_.last(),                  # Leaf
                        np.zeros(nb_features, dtype=bool),  # Used features
                        np.ones(nb_instances, dtype=bool)   # Instances mask
                        )])
        queue_it = 0
        while len(queue) > 0:
            queue_it += 1
            node, used_ftrs, idxs = queue.popleft()
            # print(f'------ node ({queue_it})\n  {node}\n  {used_ftrs}\n  {idxs}')
            nb_samples = np.sum(idxs)
            targetsum = np.sum(targets[idxs])
            self.tree_.value[node][0, 1] = targetsum
            nontargetsum = nb_samples - targetsum
            self.tree_.value[node][0, 0] = nontargetsum
            self.tree_.n_node_samples[node] = nb_samples
            if np.all(targets[idxs]) or not np.any(targets[idxs]):
                # print('Pure leaf')
                self.tree_.impurity[node] = 0
                continue
            curvalues = features[idxs, :]
            curtargets = targets[idxs]
            h0 = DecisionTreeClassifier.entropy(curtargets)
            all_gains = []
            max_kd = 0
            for fi in range(nb_features):
                if (use_feature_once and used_ftrs[fi]) or (ignore_features is not None and fi in ignore_features):
                    continue
                ig, thr, _ = self.informationgain_continuous(curtargets, curvalues[:, fi], h0=h0)
                if thr is None or ig < min_ig:
                    kd = None
                    gain = 0.0
                    # if __debug__ and logger.isEnabledFor(logging.DEBUG):
                    #     if thr is None:
                    #         thr = "None"
                    #     else:
                    #         thr = f"{thr:+.5f}"
                    #     logger.debug(f"Not splitting feature {fi:<3}, ig={ig:.5f}, thr={thr}, "
                    #                  f"k={k}, kd=None, gain=0.0")
                else:
                    # Prefer values in low-density regions (thus large k dist)
                    kd = self.kdistance(curvalues[:, fi], thr, k=k)
                    if kd > max_kd:
                        max_kd = kd
                    all_gains.append((fi, ig, thr, kd))
            best_gain, best_fi, best_thr, best_kd = 0, None, None, None
            for fi, ig, thr, kd in all_gains:
                gain = ig * (1 + (kd / max_kd)**1)
                # if __debug__ and logger.isEnabledFor(logging.DEBUG):
                #     logger.debug(f"Splitting feature {fi:<3}, ig={ig:.5f}, thr={thr:+.5f}, "
                #                  f"k={k}, kd={kd:.5f}, gain={gain:.5f}")
                # print(f'fi={fi}, thr={thr}, ig={ig}, kd={kd}, gain={gain}')
                if best_gain < gain:
                    best_gain = gain
                    best_fi = fi
                    best_thr = thr
                    best_kd = kd
            self.tree_.impurity[node] = h0
            if best_fi is not None:
                self.tree_.kd[node] = best_kd
                leq_idxs = idxs & (features[:, best_fi] <= best_thr)
                used_ftrs = used_ftrs.copy()
                used_ftrs[best_fi] = True
                self.tree_.feature[node] = best_fi
                self.tree_.threshold[node] = best_thr
                lessorequal = self.tree_.add()
                # print(f'best_fi={best_fi}, best_thr={best_thr}, best_gain={best_gain}')
                queue.append((lessorequal, used_ftrs, leq_idxs))
                self.tree_.children_left[node] = lessorequal
                larger = self.tree_.add()
                larger_idxs = idxs & (features[:, best_fi] > best_thr)
                queue.append((larger, used_ftrs, larger_idxs))
                self.tree_.children_right[node] = larger
                # print('New queue:')
                # print('\n'.join([str(elmt) for elmt in queue]))
            else:
                pass
        return self

    def avg_entropy_leafs(self):
        e = 0
        c = 0
        for i in range(len(self.tree_.feature)):
            if self.tree_.feature[i] == -1:
                c += 1
                values = self.tree_.value[i][0]
                prob = values[0] / (values[0] + values[1])
                try:
                    e -= prob * math.log(prob) + (1-prob) * math.log(1-prob)
                except ValueError:
                    pass
        return e / c

    def avg_impurity(self):
        impurity = 0
        count = 0
        for idx in range(len(self.tree_.feature)):
            if self.tree_.feature[idx] == -1:
                count += 1
                impurity += self.tree_.impurity[idx]
        impurity = impurity / count
        return impurity

    def score(self, max_kd):
        # max_kd = np.max(self.tree_.kd)
        sum_wimpurity = 0
        cnt_wimpurity = 0
        queue = deque()
        queue.append((0, 0, 0))  # index, kd max, kd count
        while len(queue) > 0:
            idx, branch_max_kd, cnt = queue.popleft()
            idx_l = self.tree_.children_left[idx]
            idx_r = self.tree_.children_right[idx]
            if idx_l == -1 and idx_r == -1:
                # print(f"impurity = {self.tree_.impurity[idx]}, branch_max_kd = {branch_max_kd}, max_kd = {max_kd}, factor = {1 + (branch_max_kd / max_kd)**2}")
                sum_wimpurity += (1.0 - self.tree_.impurity[idx]) * (1 + (branch_max_kd / max_kd)**1)
                cnt_wimpurity += 1
            else:
                ncsum = max(branch_max_kd, self.tree_.kd[idx])
                ncnt = cnt + 1
                queue.append((idx_l, ncsum, ncnt))
                queue.append((idx_r, ncsum, ncnt))
        return sum_wimpurity / cnt_wimpurity

    def set_features(self, features):
        for i in range(len(self.tree_.feature)):
            if self.tree_.feature[i] != -1:
                self.tree_.feature[i] = features[self.tree_.feature[i]]

    def __eq__(self, other):
        return True

    def __lt__(self, other):
        return False

    def __le__(self, other):
        return True

    def __gt__(self, other):
        return False

    def __ge__(self, other):
        return True


class Tree:
    def __init__(self):
        """Tree to represent a Decision Tree.

        This datastructure has the fields required to be compatible with scikit-learn and
        its Graphvis DOT representation.
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
        self.kd = []
        self.add()

    def add(self):
        """Add a new node to the tree.

        :return: Index of new node
        """
        self.threshold.append(-1)
        self.feature.append(-1)
        self.children_right.append(-1)
        self.children_left.append(-1)
        self.value.append(np.array([[0, 0]]))
        self.impurity.append(-1)
        self.n_node_samples.append(-1)
        self.n_classes.append(-1)
        self.kd.append(-1)
        return len(self.feature) - 1

    def last(self):
        """Last node.

        :return: Index of last added node
        """
        return len(self.feature) - 1

    @property
    def nb_nodes(self):
        return len(self.threshold)

    @property
    def used_features(self):
        return set(self.feature)

    @property
    def depth(self):
        max_depth = 1
        queue = deque()
        queue.append((0, 1))
        while len(queue) > 0:
            idx, depth = queue.popleft()
            if depth > max_depth:
                max_depth = depth
            idx_l = self.children_left[idx]
            idx_r = self.children_right[idx]
            if idx_l != -1:
                queue.append((idx_l, depth + 1))
            if idx_r != -1:
                queue.append((idx_r, depth + 1))
        return max_depth
