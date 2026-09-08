########################################################################################################
#
# Use function TTC(samples, 'dataframe') or TTC(samples, 'values') to compute two-dimensional Time-To-Collision.
#
# The first input is a pandas dataframe of vehicle pair samples, which should include the following columns.
# -----------------------------------------------------------------------------------
# x_i      :  x coordinate of the ego vehicle (usually assumed to be centroid)      |
# y_i      :  y coordinate of the ego vehicle (usually assumed to be centroid)      |
# vx_i     :  x coordinate of the velocity of the ego vehicle                       |
# vy_i     :  y coordinate of the velocity of the ego vehicle                       |
# hx_i     :  x coordinate of the heading direction of the ego vehicle              |
# hy_i     :  y coordinate of the heading direction of the ego vehicle              |
# length_i :  length of the ego vehicle                                             |
# width_i  :  width of the ego vehicle                                              |
# x_j      :  x coordinate of another vehicle (usually assumed to be centroid)      |
# y_j      :  y coordinate of another vehicle (usually assumed to be centroid)      |
# vx_j     :  x coordinate of the velocity of another vehicle                       |
# vy_j     :  y coordinate of the velocity of another vehicle                       |
# hx_j     :  x coordinate of the heading direction of another vehicle              |
# hy_j     :  y coordinate of the heading direction of another vehicle              |
# length_j :  length of another vehicle                                             |
# width_j  :  width of another vehicle                                              |
#------------------------------------------------------------------------------------
# The second input allows outputing a dataframe with inputed samples plus a new column named 'TTC', or mere TTC values.
#
# If TTC==np.inf, the ego vehicle and another vehicle will never collide if they keep current speed.
# A negative TTC means the bounding boxes of the ego vehicle and another vehicle are overlapping.
# This is due to approximating the space occupied by a vehicle with a rectangular.
# In other words, TTC<0 in this computation means the collision between the two vehicles almost (or although seldom, already) occurred.
#
# *** Note that mere TTC computation can give an extreme small positive value even when the vehivles are overlapping a bit.
#     In order to improve the accuracy, please use function CurrentD(samples, 'dataframe') or CurrentD(samples, 'values') to further
#     exclude overlapping vehicles.
#
########################## Copyright (c) 2022 Yiru Jiao <y.jiao-1@tudelft.nl> ###########################

########################################################################################################
# Refactored to use Numpy insted of pandas dataframe
# Input 'samples' format: 2D NumPy array of shape (N, 16) or 1D array of shape (16,)
# Output: NumPy 1D array of TTC values (or scalar if input is 1D)
########################################################################################################

import warnings
import numpy as np

def line(point0, point1):
    x0, y0 = point0
    x1, y1 = point1
    a = y0 - y1
    b = x1 - x0
    c = x0 * y1 - x1 * y0
    return a, b, c


def intersect(line0, line1):
    a0, b0, c0 = line0
    a1, b1, c1 = line1
    D = a0 * b1 - a1 * b0  # D==0 then two lines overlap[cite: 1]
    D[D == 0] = np.nan
    x = (b0 * c1 - b1 * c0) / D
    y = (a1 * c0 - a0 * c1) / D
    return np.array([x, y])


def ison(line_start, line_end, point):
    crossproduct = (point[1] - line_start[1]) * (
        line_end[0] - line_start[0]
    ) - (point[0] - line_start[0]) * (line_end[1] - line_start[1])
    dotproduct = (point[0] - line_start[0]) * (
        line_end[0] - line_start[0]
    ) + (point[1] - line_start[1]) * (line_end[1] - line_start[1])
    squaredlength = (line_end[0] - line_start[0]) ** 2 + (
        line_end[1] - line_start[1]
    ) ** 2
    return (
        (np.absolute(crossproduct) <= 1e-5)
        & (dotproduct >= 0)
        & (dotproduct <= squaredlength)
    )


def dist_p2l(point, line_start, line_end):
    return np.absolute(
        (line_end[0] - line_start[0]) * (line_start[1] - point[1])
        - (line_start[0] - point[0]) * (line_end[1] - line_start[1])
    ) / np.sqrt(
        (line_end[0] - line_start[0]) ** 2 + (line_end[1] - line_start[1]) ** 2
    )


def getpoints(samples):
    # vehicle i
    heading_i = samples[:, [4, 5]]
    perp_heading_i = np.array([-heading_i[:, 1], heading_i[:, 0]]).T
    heading_scale_i = np.tile(
        np.sqrt(heading_i[:, 0] ** 2 + heading_i[:, 1] ** 2), (2, 1)
    ).T
    length_i = np.tile(samples[:, 6], (2, 1)).T
    width_i = np.tile(samples[:, 7], (2, 1)).T

    point_up = (
        samples[:, [0, 1]] + heading_i / heading_scale_i * length_i / 2
    )
    point_down = (
        samples[:, [0, 1]] - heading_i / heading_scale_i * length_i / 2
    )
    point_i1 = (point_up + perp_heading_i / heading_scale_i * width_i / 2).T
    point_i2 = (point_up - perp_heading_i / heading_scale_i * width_i / 2).T
    point_i3 = (point_down + perp_heading_i / heading_scale_i * width_i / 2).T
    point_i4 = (point_down - perp_heading_i / heading_scale_i * width_i / 2).T

    # vehicle j
    heading_j = samples[:, [12, 13]]
    perp_heading_j = np.array([-heading_j[:, 1], heading_j[:, 0]]).T
    heading_scale_j = np.tile(
        np.sqrt(heading_j[:, 0] ** 2 + heading_j[:, 1] ** 2), (2, 1)
    ).T
    length_j = np.tile(samples[:, 14], (2, 1)).T
    width_j = np.tile(samples[:, 15], (2, 1)).T

    point_up = (
        samples[:, [8, 9]] + heading_j / heading_scale_j * length_j / 2
    )
    point_down = (
        samples[:, [8, 9]] - heading_j / heading_scale_j * length_j / 2
    )
    point_j1 = (point_up + perp_heading_j / heading_scale_j * width_j / 2).T
    point_j2 = (point_up - perp_heading_j / heading_scale_j * width_j / 2).T
    point_j3 = (point_down + perp_heading_j / heading_scale_j * width_j / 2).T
    point_j4 = (point_down - perp_heading_j / heading_scale_j * width_j / 2).T

    return (
        point_i1,
        point_i2,
        point_i3,
        point_i4,
        point_j1,
        point_j2,
        point_j3,
        point_j4,
    )


def CurrentD(samples):
    samples = np.atleast_2d(samples)
    (
        point_i1,
        point_i2,
        point_i3,
        point_i4,
        point_j1,
        point_j2,
        point_j3,
        point_j4,
    ) = getpoints(samples)

    dist_mat = []
    count_i = 0
    for point_i_start, point_i_end in zip(
        [point_i1, point_i4, point_i3, point_i2],
        [point_i2, point_i3, point_i1, point_i4],
    ):
        count_j = 0
        for point_j_start, point_j_end in zip(
            [point_j1, point_j4, point_j3, point_j2],
            [point_j2, point_j3, point_j1, point_j4],
        ):
            if count_i < 2 and count_j < 2:
                # Distance from point to point
                dist_mat.append(
                    np.sqrt(
                        (point_i_start[0] - point_j_start[0]) ** 2
                        + (point_i_start[1] - point_j_start[1]) ** 2
                    )
                )
                dist_mat.append(
                    np.sqrt(
                        (point_i_start[0] - point_j_end[0]) ** 2
                        + (point_i_start[1] - point_j_end[1]) ** 2
                    )
                )
                dist_mat.append(
                    np.sqrt(
                        (point_i_end[0] - point_j_start[0]) ** 2
                        + (point_i_end[1] - point_j_start[1]) ** 2
                    )
                )
                dist_mat.append(
                    np.sqrt(
                        (point_i_end[0] - point_j_end[0]) ** 2
                        + (point_i_end[1] - point_j_end[1]) ** 2
                    )
                )

            # Distance from point to edge
            ist = intersect(
                line(
                    point_i_start,
                    point_i_start
                    + np.array([
                        -(point_j_start - point_j_end)[1],
                        (point_j_start - point_j_end)[0],
                    ]),
                ),
                line(point_j_start, point_j_end),
            )
            ist[:, ~ison(point_j_start, point_j_end, ist)] = np.nan
            dist_mat.append(
                np.sqrt(
                    (ist[0] - point_i_start[0]) ** 2
                    + (ist[1] - point_i_start[1]) ** 2
                )
            )

            # Overlapped bounding boxes
            ist = intersect(
                line(point_i_start, point_i_end),
                line(point_j_start, point_j_end),
            )
            dist = np.ones(samples.shape[0]) * np.nan
            dist[
                ison(point_i_start, point_i_end, ist)
                & ison(point_j_start, point_j_end, ist)
            ] = 0
            dist[
                np.isnan(ist[0])
                & (
                    ison(point_i_start, point_i_end, point_j_start)
                    | ison(point_i_start, point_i_end, point_j_end)
                )
            ] = 0
            dist_mat.append(dist)
            count_j += 1
        count_i += 1

    cdist = np.nanmin(np.array(dist_mat), axis=0)
    return cdist


def TTC_ij(samples):
    (
        point_i1,
        point_i2,
        point_i3,
        point_i4,
        point_j1,
        point_j2,
        point_j3,
        point_j4,
    ) = getpoints(samples)
    direct_v = (samples[:, [2, 3]] - samples[:, [10, 11]]).T

    dist_mat = []
    leaving_mat = []
    for point_line_start in [point_i1, point_i2, point_i3, point_i4]:
        for edge_start, edge_end in zip(
            [point_j1, point_j3, point_j1, point_j2],
            [point_j2, point_j4, point_j3, point_j4],
        ):
            point_line_end = point_line_start + direct_v
            # intersection point
            ist = intersect(
                line(point_line_start, point_line_end),
                line(edge_start, edge_end),
            )
            ist[:, ~ison(edge_start, edge_end, ist)] = np.nan
            # distance from point to intersection point
            dist_ist = np.sqrt(
                (ist[0] - point_line_start[0]) ** 2
                + (ist[1] - point_line_start[1]) ** 2
            )
            dist_ist[np.isnan(dist_ist)] = np.inf
            dist_mat.append(dist_ist)
            leaving = direct_v[0] * (ist[0] - point_line_start[0]) + direct_v[
                1
            ] * (ist[1] - point_line_start[1])
            leaving[leaving >= 0] = 10
            leaving[leaving < 0] = 1
            leaving_mat.append(leaving)

    dist2overlap = np.array(dist_mat).min(axis=0)
    relative_speed = np.sqrt(
        (samples[:, 2] - samples[:, 10]) ** 2
        + (samples[:, 3] - samples[:, 11]) ** 2
    )
    TTC = dist2overlap / relative_speed
    leaving = np.nansum(np.array(leaving_mat), axis=0)
    TTC[leaving < 10] = np.inf
    TTC[(leaving > 10) & (leaving % 10 != 0)] = -1

    return TTC


def _TTC_single_pair(samples_1d):
    """Executes single-pair TTC calculation on a (16,) vector."""
    samples_2d = samples_1d[np.newaxis, :]
    ttc_ij = TTC_ij(samples_2d)

    swap_indices = [8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7]
    samples_swapped = samples_2d[:, swap_indices]
    ttc_ji = TTC_ij(samples_swapped)

    return float(np.minimum(ttc_ij, ttc_ji)[0])


def TTC(ego, other):
    """Computes 2D TTC by executing the algorithm N times for each ego-target pair.

    Parameters
    ----------
    ego : np.ndarray
        1D array of shape (8,) -> [x, y, vx, vy, hx, hy, length, width]
    other : np.ndarray
        2D array of shape (N, 8) -> N rows of [x, y, vx, vy, hx, hy, length, width]

    Returns
    -------
    np.ndarray
        1D NumPy array of shape (N,) containing TTC for each target.
    """
    ego = np.asarray(ego)
    other = np.asarray(other)

    if other.ndim == 1:
        other = other[np.newaxis, :]

    num_targets = other.shape[0]
    ttc_array = np.empty(num_targets, dtype=float)

    for i in range(num_targets):
        pair_sample = np.hstack([ego, other[i]])
        ttc_array[i] = _TTC_single_pair(pair_sample)

    return ttc_array