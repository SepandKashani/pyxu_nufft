import itertools

import numpy as np
import pyxu.info.ptype as pxt

import pyxu_nufft.math.numba as pxm_nb

__all__ = [
    "grid_cluster",
    "bisect_cluster",
    "fuse_cluster",
]

BBoxDim = pxt.NDArray  # np.float32/64
PointIndex = pxt.NDArray  # np.int64
ClusterMapping = pxt.NDArray  # np.int64


def grid_cluster(
    x: pxt.NDArray,
    bbox_dim: BBoxDim,
) -> tuple[PointIndex, ClusterMapping]:
    """
    Split D-dimensional points onto lattice-aligned clusters.
    Each cluster may contain arbitrary-many points.

    Parameters
    ----------
    x: NDArray
        (M, D) point cloud.
    bbox_dim: BBoxDim
        (D,) box dimensions.

    Returns
    -------
    x_idx: PointIndex
        (M,) indices that sort `x` along axis 0.
    cl_info: ClusterMapping
        (Q+1,) cluster start/stop indices.

        ``x[x_idx][cl_info[q] : cl_info[q+1]]`` contains all points in the q-th cluster.
    """
    (M, D), dtype = x.shape, x.dtype
    bbox_dim = np.array(bbox_dim, dtype=dtype)
    assert (len(bbox_dim) == D) and (bbox_dim > 0).all()

    # Quick exit if only one point.
    if M == 1:
        x_idx = np.r_[0]
        cl_info = np.r_[0, 1]
    else:
        # Compute cluster index of each point
        c_idx, lattice_shape = pxm_nb.digitize(x, bbox_dim)

        # Re-order & count points
        cl_count, x_idx = pxm_nb.count_sort(c_idx, k=lattice_shape.prod())

        # Encode `cl_info`
        Q = len(cl_count)
        cl_info = np.zeros(Q + 1, dtype=np.int64)
        cl_info[1:] = cl_count.cumsum()
    return x_idx, cl_info


def bisect_cluster(
    cl_info: ClusterMapping,
    N_max: int,
) -> ClusterMapping:
    """
    Hierarchically split clusters until each contains at most `N_max` points.

    Parameters
    ----------
    cl_info: ClusterMapping
        (Q+1,) cluster start/stop indices.
    N_max: int
        Maximum number of points allocated per cluster.

    Returns
    -------
    clB_info: ClusterMapping
        (L+1,) bisected cluster start/stop indices.
    """
    M = cl_info[-1]
    Q = len(cl_info) - 1
    assert N_max > 0

    cl_size = cl_info[1:] - cl_info[:-1]
    cl_chunks = np.ceil(cl_size / N_max).astype(int)
    L = sum(cl_chunks)

    clB_info = np.full(L + 1, fill_value=M, dtype=np.int64)
    _l = 0
    for q in range(Q):
        for c in range(cl_chunks[q]):
            clB_info[_l] = cl_info[q] + min(c * N_max, cl_size[q])
            _l += 1
    return clB_info


def fuse_cluster(
    x: pxt.NDArray,
    x_idx: PointIndex,
    cl_info: ClusterMapping,
    bbox_dim: BBoxDim,
) -> tuple[PointIndex, ClusterMapping]:
    """
    Fuse neighboring clusters until aggregate bounding-boxes have at most size `bbox_dim`.

    It is assumed all clusters passed in already satisfy this pre-condition.

    Parameters
    ----------
    x: NDArray
        (M, D) point cloud.
    x_idx: PointIndex
        (M,) indices which sort `x` into clusters.
    cl_info: ClusterMapping
        (Q+1,) cluster start/stop indices.

        ``x[x_idx][cl_info[q] : cl_info[q+1]]`` contains all points in the q-th cluster.
    bbox_dim: BBoxDim
        (D,) maximum (fused) box dimensions.

    Returns
    -------
    xF_idx: PointIndex
        (M,) indices which sort `x` into *fused* clusters.
    clF_info: ClusterMapping
        (L+1,) fused cluster start/stop indices.

        ``x[xF_idx][clF_info[l] : clF_info[l+1]]`` contains all points in the l-th cluster.
    """
    (M, D), dtype = x.shape, x.dtype
    bbox_dim = np.array(bbox_dim, dtype=dtype)
    assert (len(bbox_dim) == D) and (bbox_dim > 0).all()

    Q = len(cl_info) - 1

    # Quit exit if only one cluster.
    if Q == 1:
        xF_idx = x_idx
        clF_info = cl_info
    else:
        # Initialize state variables:
        # * cl_LL, cl_UR: dict[int64, float32/64(D,)]
        # * cl_group: dict[int64, set]
        cl_LL, cl_UR = pxm_nb.group_minmax(x, x_idx, cl_info)
        cl_LL = {q: cl_LL[q] for q in range(Q)}
        cl_UR = {q: cl_UR[q] for q in range(Q)}
        cl_group = {q: {q} for q in range(Q)}

        # Fuse clusters until completion
        clusters = set(range(Q))
        q = Q  # fused cluster index
        candidates_available = True
        while (len(clusters) > 1) and candidates_available:
            for i, j in itertools.combinations(clusters, 2):
                bbox_LL = np.fmin(cl_LL[i], cl_LL[j])
                bbox_UR = np.fmax(cl_UR[i], cl_UR[j])

                fuseable = np.all(bbox_UR - bbox_LL <= bbox_dim)
                if fuseable:
                    cl_LL[q] = bbox_LL
                    cl_LL.pop(i), cl_LL.pop(j)

                    cl_UR[q] = bbox_UR
                    cl_UR.pop(i), cl_UR.pop(j)

                    cl_group[q] = cl_group[i] | cl_group[j]
                    cl_group.pop(i), cl_group.pop(j)

                    clusters.add(q)
                    clusters.remove(i), clusters.remove(j)
                    q += 1

                    # clusters have changed -> skip rest of the for loop
                    break
            else:
                # no clusters could be fused -> terminate while loop
                candidates_available = False

        # Encode (xF_idx, clF_info)
        L = len(cl_group)
        clF_info = np.full(L + 1, fill_value=M, dtype=np.int64)
        xF_idx = np.empty(M, dtype=np.int64)
        offset = 0
        for _l, cluster_ids in enumerate(cl_group.values()):
            clF_info[_l] = offset
            for q in cluster_ids:
                a, b = cl_info[q], cl_info[q + 1]
                xF_idx[offset : offset + (b - a)] = x_idx[a:b]
                offset += b - a
    return xF_idx, clF_info
