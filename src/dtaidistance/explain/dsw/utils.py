import math
import numpy as np


# def split_delta_range(delta_range, i0, i1, p0, p1):
#   """Split segment based on equal relative lengths."""
#     i_nb = math.ceil((i1 - i0) / delta_range)
#     i_len = math.ceil((i1 - i0) / i_nb)
#     i0i1s = []
#     for idx in range(i_nb):
#         i0p, i1p = i0 + idx*i_len, min(i1, i0 + (idx+1)*i_len)
#         perc0 = (i0p-i0) / (i1-i0)
#         perc1 = (i1p-i0) / (i1-i0)
#         p0a = (int(perc0*p1[0]+(1-perc0)*p0[0]), int(perc0*p1[1]+(1-perc0)*p0[1]))
#         p1a = (int(perc1*p1[0]+(1-perc1)*p0[0]), int(perc1*p1[1]+(1-perc1)*p0[1]))
#         i0i1s.append((i0p,i1p, p0a, p1a))
#     # print(f"{i0i1s=}")
#     return i0i1s


def split_segment_implicit(delta_range, i0, i1, p0, p1, points):
    """Split segment based on closest point on linear segment."""
    assert np.allclose(points[i0], p0)
    assert np.allclose(points[i1], p1)

    p0p1norm = np.linalg.norm(p1 - p0)
    p0p1normsqr = p0p1norm ** 2
    i_nb = math.ceil((i1 - i0) / delta_range)
    i_len = math.ceil((i1 - i0) / i_nb)
    prev_pi = i0  # previous path index
    prev_pp = points[i0]  # previous path point
    prev_fp = prev_pp # previous foot point
    i0i1s = []

    for idx in range(i_nb):
        next_pi = min(i1, i0 + (idx+1)*i_len)
        next_pp = points[next_pi]
        # Find point on straight line, closest to current point
        if np.allclose(p0, p1):
            next_fp = p0
        else:
            # Closest distance (point might be beyond line segment
            t = ((next_pp[0] - p0[0]) * (p1[0] - p0[0]) +
                    (next_pp[1] - p0[1]) * (p1[1] - p0[1])) / p0p1normsqr
            if t < 0:
                next_fp = p0
            elif t > 1:
                next_fp = p1
            else:
                next_fp = np.array([int(p0[0] + t * (p1[0] - p0[0])),
                                    int(p0[1] + t * (p1[1] - p0[1]))])
        i0i1s.append((prev_pi,next_pi,prev_fp,next_fp))
        prev_pi = next_pi
        prev_pp = next_pp
        prev_fp = next_fp
    return i0i1s


def get_1stderiv_in_path(series_from, series_to, points, h=1):
    """Compute the first derivative of each point in the cost
    matrix (or self-similarity matrix)."""
    # if deriv_morepoints:
    #     return get_1stderiv_in_path_threepoint(points, h)
    ders = np.zeros(len(points))
    i_of_m = len(series_from) - h - 1
    i_ot_m = len(series_to) - h - 1
    for idx in range(0, len(points)):
        i_of, i_ot = points[idx]
        cost_ft = abs(series_from[i_of] - series_to[i_ot])
        if i_of < h or i_of > i_of_m or i_ot < h or i_ot > i_ot_m:
            # Compute derivatives close to the border
            der = (abs(cost_ft
                            - abs(series_from[i_of]
                                - series_to[max(0,i_ot - h)])
                            ) / h +
                        abs(cost_ft
                            - abs(series_from[i_of]
                                - series_to[min(i_ot_m, i_ot + h)])
                            ) / h +
                        abs(cost_ft
                            - abs(series_from[max(0,i_of-h)]
                                - series_to[i_ot])
                            ) / h +
                        abs(cost_ft
                            - abs(series_from[min(i_of_m,i_of+h)]
                                - series_to[i_ot])
                            ) / h) / 4
        else:
            # Forward and backward differences (1st order approx) in two 
            # directions (horizontal and vertical) and maxed.
            # Use absolute inner distance (thus not the derivative in the
            # self-similarity or cost matrix). This makes it independent of
            # the amplitude of the time series.
            der = (abs(cost_ft - abs(series_from[i_of]   - series_to[i_ot-h])) / h +
                        abs(cost_ft - abs(series_from[i_of]   - series_to[i_ot+h])) / h +
                        abs(cost_ft - abs(series_from[i_of-h] - series_to[i_ot]))   / h +
                        abs(cost_ft - abs(series_from[i_of+h] - series_to[i_ot]))   / h) / 4
        ders[idx] = abs(der)
    # If the first derivative is zero, then distance has little impact
    # in the max_deriv_deviation approach. Force a minimal value.
    min_ders = np.max(ders) * 0.1
    ders[ders < min_ders] = min_ders
    return ders


def get_1stderiv_in_path_threepoint(series_from, series_to, points, h=1):
    """Compute the first derivative of each point in the cost
    matrix (or self-similarity matrix)."""
    # if not deriv_morepoints:
    #     return get_1stderiv_in_path(points, h)
    ders = np.zeros(len(points))
    i_of_m = len(series_from) - 2*h - 1
    i_ot_m = len(series_to) - 2*h - 1
    for idx in range(0, len(points)):
        i_of, i_ot = points[idx]
        cost_ft = abs(series_from[i_of] - series_to[i_ot])
        if i_of < 2*h or i_of > i_of_m or i_ot < 2*h or i_ot > i_ot_m:
            # Compute derivatives close to the border
            der = (abs(cost_ft
                            - abs(series_from[i_of]
                                - series_to[max(0,i_ot - h)])
                            ) / h +
                        abs(cost_ft
                            - abs(series_from[i_of]
                                - series_to[min(i_ot_m, i_ot + h)])
                            ) / h +
                        abs(cost_ft
                            - abs(series_from[max(0,i_of-h)]
                                - series_to[i_ot])
                            ) / h +
                        abs(cost_ft
                            - abs(series_from[min(i_of_m,i_of+h)]
                                - series_to[i_ot])
                            ) / h) / 4
        else:
            # Three-point forward and backward differences (1st order approx) in two 
            # directions (horizontal and vertical) and maxed.
            # Use absolute inner distance (thus not the derivative in the
            # self-similarity or cost matrix). This makes it independent of
            # the amplitude of the time series.
            der = (abs(-3*cost_ft + 4*abs(series_from[i_of]   - series_to[i_ot-h]) - abs(series_from[i_of]     - series_to[i_ot-2*h])) / 2*h +
                        abs(-3*cost_ft + 4*abs(series_from[i_of]   - series_to[i_ot+h]) - abs(series_from[i_of]     - series_to[i_ot+2*h])) / 2*h +
                        abs(-3*cost_ft + 4*abs(series_from[i_of-h] - series_to[i_ot])   - abs(series_from[i_of-2*h] - series_to[i_ot]))     / 2*h +
                        abs(-3*cost_ft + 4*abs(series_from[i_of+h] - series_to[i_ot])   - abs(series_from[i_of+2*h] - series_to[i_ot]))     / 2*h) / 4
        ders[idx] = abs(der)
    # If the first derivative is zero, then distance has little impact
    # in the max_deriv_deviation approach. Force a minimal value.
    min_ders = np.max(ders) * 0.1
    ders[ders < min_ders] = min_ders
    return ders


def get_2ndderiv_in_path(series_from, series_to, points, h=1):
    """Compute the second derivative of each point in the cost
    matrix (or self-similarity matrix).

    This method ignores the direction and computes the max
    absolute value of the derivative in the vertical and horizontal direction.
    The goal is to select points that have a rapidly changing point
    in the cost matrix.

    The centered difference approximations (along the diagonals)
    method is used.
    """
    # if deriv_morepoints:
    #     return get_2ndderiv_in_path_fivepoint(points, h)
    ders = np.zeros(len(points))
    i_of_m = len(series_from) - h - 1
    i_ot_m = len(series_to) - h - 1
    for idx in range(0, len(points)):
        i_of, i_ot = points[idx]
        cost_ft = abs(series_from[i_of] - series_to[i_ot])
        if i_of < h or i_of > i_of_m or i_ot < h or i_ot > i_ot_m:
            # Compute derivatives close to the border
            der = (abs(abs(series_from[i_of] - series_to[max(0, i_ot - h)]) +
                            abs(series_from[i_of] - series_to[min(i_ot_m, i_ot + h)]) -
                            2 * cost_ft
                            ) / (h ** 2) +
                        abs(abs(series_from[min(i_of_m, i_of + h)] - series_to[i_ot]) +
                            abs(series_from[max(0, i_of - h)] - series_to[i_ot]) -
                            2 * cost_ft
                            ) / (h ** 2) / 2
                        )
        else:
            # Centered difference approximations (along the diagonals)
            # Could also have been five-point stencil 2nd derivative?
            der = (abs(abs(series_from[i_of] - series_to[i_ot-h])+
                            abs(series_from[i_of] - series_to[i_ot+h])-
                            2 * cost_ft
                            ) / (h**2) +
                        abs(abs(series_from[i_of+h] - series_to[i_ot])+
                            abs(series_from[i_of-h] - series_to[i_ot])-
                            2 * cost_ft
                            ) / (h**2)
                        ) / 2
        ders[idx] = abs(der)
    return ders


def get_2ndderiv_in_path_fivepoint(series_from, series_to, points, h=1):
    """Compute the second derivative of each point in the cost
    matrix (or self-similarity matrix).

    This method ignores the direction and computes the max
    absolute value of the derivative in the vertical and horizontal direction.
    The goal is to select points that have a rapidly changing point
    in the cost matrix.

    The centered difference approximations (along the diagonals)
    method is used.
    """
    # if not deriv_morepoints:
    #     return get_2ndderiv_in_path(points, h)
    ders = np.zeros(len(points))
    i_of_m = len(series_from) - 2*h - 1
    i_ot_m = len(series_to) - 2*h - 1
    for idx in range(0, len(points)):
        i_of, i_ot = points[idx]
        cost_ft = abs(series_from[i_of] - series_to[i_ot])
        sf_i_of = series_from[i_of]
        st_i_ot = series_to[i_ot]
        if i_of < 2*h or i_of > i_of_m or i_ot < 2*h or i_ot > i_ot_m:
            # Compute derivatives close to the border
            der = (abs(abs(series_from[i_of] - series_to[max(0, i_ot - h)]) +
                            abs(series_from[i_of] - series_to[min(i_ot_m, i_ot + h)]) -
                            2 * cost_ft
                            ) / (h ** 2) +
                        abs(abs(series_from[min(i_of_m, i_of + h)] - st_i_ot) +
                            abs(series_from[max(0, i_of - h)] - st_i_ot) -
                            2 * cost_ft
                            ) / (h ** 2)
                        ) / 2
        else:
            # Five-point central difference formula difference approximations
            # Could also have been five-point stencil 2nd derivative?
            der = (abs(-   abs(sf_i_of - series_to[i_ot-2*h])
                            +16*abs(sf_i_of - series_to[i_ot-  h])
                            -30*cost_ft
                            +16*abs(sf_i_of - series_to[i_ot+  h])
                            -   abs(sf_i_of - series_to[i_ot+2*h])
                            ) / (12*h**2) +
                        abs(-   abs(series_from[i_of-2*h] - st_i_ot)
                            +16*abs(series_from[i_of-  h] - st_i_ot)
                            -30*cost_ft
                            +16*abs(series_from[i_of+  h] - st_i_ot)
                            -   abs(series_from[i_of+2*h] - st_i_ot)
                            ) / (12*h**2)
                        ) / 2
        ders[idx] = abs(der)
    return ders


def plot_arrow(x, y, arrowstyle, delta, delta_min, delta_max, ax, color):
    """Custom arrow plotting. Allows for inwards pointing arrows.

    :param x: A pair of x coordinates
    :param y: A pair of y coordinates
    :param arrowstyle: One of '<->', '<-<', '>->', '>-<'
    :param delta: Amount of change. Used to determine the color transparency.
    :param delta_min: Minimum amount of change. Used to determine the color transparency.
    :param delta_max: Maximum amount of change. Used to determine the color transparency.
    :param ax: Matplotlib Axes
    :return:
    """
    import matplotlib.patches as mpatches
    import matplotlib.path as mpath

    t = ax.transAxes.transform([(0, 0), (1, 1)])
    t = ax.get_figure().get_dpi() / (t[1, 1] - t[0, 1]) / 23
    mdx, mdy = 144 * t, 0.10
    ml, _, mr = arrowstyle

    if ml == ">":
        sl = [
            (x[0], y[0] + mdy),
            (x[0] + mdx, y[0]),
            (x[0], y[0] - mdy),
            (x[0] + mdx, y[0]),
        ]
    elif ml == "<":
        sl = [
            (x[0] + mdx, y[0] - mdy),
            (x[0], y[0]),
            (x[0] + mdx, y[0] + mdy),
            (x[0], y[0]),
        ]
    else:
        raise ValueError(f'Unknown ml: {ml}')
    if mr == ">":
        sr = [
            (x[1], y[1]),
            (x[1] - mdx, y[1] + mdy),
            (x[1] - mdx, y[1] - mdy),
            (x[1], y[1]),
        ]
    elif mr == "<":
        sr = [
            (x[1] - mdx, y[1]),
            (x[1], y[1] + mdy),
            (x[1] - mdx, y[1]),
            (x[1], y[1] - mdy),
        ]
    else:
        raise ValueError(f'Unknown mr: {mr}')

    alpha = 0.5 * (delta - delta_min) / (delta_max - delta_min) + 0.2
    # ax.plot(x, y, color=color_shade, alpha=alpha)
    # ax.plot(x[0], y[0], color=color_shade, marker=ml, alpha=alpha, markersize=marker_size)
    # ax.plot(x[1], y[1], color=color_shade, marker=mr, alpha=alpha, markersize=marker_size)
    pp1 = mpatches.PathPatch(
        mpath.Path(
            sl + sr,
            [
                mpath.Path.MOVETO,
                mpath.Path.LINETO,
                mpath.Path.LINETO,
                mpath.Path.MOVETO,
                mpath.Path.LINETO,
                mpath.Path.LINETO,
                mpath.Path.MOVETO,
                mpath.Path.LINETO,
            ],
        ),
        fc="none",
        transform=ax.transData,
        color=color,
        alpha=alpha,
        linewidth=1,
        clip_on=False,
    )
    ax.add_patch(pp1)


def plot_arrow2(x, y, arrowstyle, height, fig, color):
    """Custom arrow plotting. Allows for inwards pointing arrows.

    :param x: A pair of x coordinates
    :param y: A pair of y coordinates
    :param arrowstyle: One of '<->', '<-<', '>->', '>-<', '---'
    :param delta: Amount of change. Used to determine the color transparency.
    :param epsilon: Amount of ignored change. Used to determine the color transparency.
    :param ax: Matplotlib Axes
    :return:
    """
    import matplotlib.patches as mpatches
    import matplotlib.path as mpath

    # t = ax.transAxes.transform([(0, 0), (1, 1)])
    # t = ax.get_figure().get_dpi() / (t[1, 1] - t[0, 1]) / 23
    mdx, mdy = height, height
    ml, _, mr = arrowstyle

    if ml == ">":
        sl = [
            (x[0], y[0] + mdy),
            (x[0] + mdx, y[0]),
            (x[0], y[0] - mdy),
            (x[0] + mdx, y[0]),
        ]
    elif ml == "<":
        sl = [
            (x[0] + mdx, y[0] - mdy),
            (x[0], y[0]),
            (x[0] + mdx, y[0] + mdy),
            (x[0], y[0]),
        ]
    elif ml == "-":
        sl = [
            (x[0], y[0]),
            (x[0], y[0]),
            (x[0], y[0]),
            (x[0], y[0]),
        ]
    else:
        raise ValueError(f'Unknown ml: {ml}')
    if mr == ">":
        sr = [
            (x[1], y[1]),
            (x[1] - mdx, y[1] + mdy),
            (x[1] - mdx, y[1] - mdy),
            (x[1], y[1]),
        ]
    elif mr == "<":
        sr = [
            (x[1] - mdx, y[1]),
            (x[1], y[1] + mdy),
            (x[1] - mdx, y[1]),
            (x[1], y[1] - mdy),
        ]
    elif mr == "-":
        sr = [
            (x[1], y[1]),
            (x[1], y[1]),
            (x[1], y[1]),
            (x[1], y[1]),
        ]
    else:
        raise ValueError(f'Unknown mr: {mr}')

    alpha = 0.6
    # ax.plot(x, y, color=color_shade, alpha=alpha)
    # ax.plot(x[0], y[0], color=color_shade, marker=ml, alpha=alpha, markersize=marker_size)
    # ax.plot(x[1], y[1], color=color_shade, marker=mr, alpha=alpha, markersize=marker_size)
    pp1 = mpatches.PathPatch(
        mpath.Path(
            sl + sr,
            [
                mpath.Path.MOVETO,
                mpath.Path.LINETO,
                mpath.Path.LINETO,
                mpath.Path.MOVETO,
                mpath.Path.LINETO,
                mpath.Path.LINETO,
                mpath.Path.MOVETO,
                mpath.Path.LINETO,
            ],
        ),
        fc="none",
        transform=fig.transFigure,
        color=color,
        alpha=alpha,
        linewidth=1,
        clip_on=False,
    )
    return pp1

