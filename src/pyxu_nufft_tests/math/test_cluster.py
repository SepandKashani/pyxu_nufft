import numpy as np
import pytest
import pyxu.info.deps as pxd
import pyxu_tests.conftest as ct
import pyxu_tests.operator.conftest as cto

import pyxu_nufft.math as pxm

# Reference some Pyxu fixtures.
width = ct.width


# Supported array backends for all clustering routines.
@pytest.fixture(
    params=[
        pxd.NDArrayInfo.NUMPY,
        pxd.NDArrayInfo.CUPY,  # not supported (yet)
    ]
)
def ndi(request) -> pxd.NDArrayInfo:
    _ndi = request.param
    cto.MapT._skip_if_unsupported(_ndi)
    return _ndi


class TestGridCluster:
    def test_value(self, space_dim, bbox_dim, _data):
        # * (in, out) shapes are consistent
        # * all points accounted for
        # * cl_info well formed
        # * cluster max extent <= bbox_dim
        # * #clusters <= max allowed count

        # (in, out) shapes are consistent
        x = _data["x"]
        x_idx = _data["x_idx"]
        cl_info = _data["cl_info"]
        assert (x.ndim == 2) and (x.shape[1] == space_dim)
        assert cl_info.ndim == 1
        assert (x_idx.ndim == 1) and (len(x) == len(x_idx))

        ndi = pxd.NDArrayInfo.from_obj(x)
        xp = ndi.module()
        M = len(x)
        Q = len(cl_info) - 1

        # All points accounted for
        assert M == len(x_idx)
        assert M == len(xp.unique(x_idx))

        # cl_info well formed, i.e.
        # * cl_info is strictly monotonic -> no empty clusters
        # * cl_info[-1] goes beyond len(x)
        assert xp.all(cl_info[1:] > cl_info[:-1])
        assert M == cl_info[-1]

        # cluster max extent <= bbox_dim
        bbox_dim = xp.array(bbox_dim)
        for q in range(Q):
            select = slice(cl_info[q], cl_info[q + 1])
            x_cl = x[x_idx[select]]
            ptp_cl = xp.ptp(x_cl, axis=0)  # (D,)
            assert xp.all(ptp_cl <= bbox_dim)

        # #clusters <= max allowed count
        N_cl_max = 5**space_dim  # 5 hard-coded in Fixture[_data]
        assert Q <= N_cl_max

    def test_backend(self, ndi, _data):
        # (input, output) backends match
        get_backend = lambda _: pxd.NDArrayInfo.from_obj(_)

        assert get_backend(_data["x_idx"]) == ndi
        assert get_backend(_data["cl_info"]) == ndi

    def test_prec(self, _data):
        # outputs are integer arrays
        get_dtype = lambda _: _.dtype

        assert get_dtype(_data["x_idx"]) == np.int64
        assert get_dtype(_data["cl_info"]) == np.int64

    # Fixtures ----------------------------------------------------------------
    @pytest.fixture(params=[1, 2, 3])
    def space_dim(self, request) -> int:
        return request.param

    @pytest.fixture
    def bbox_dim(self, space_dim) -> tuple[float]:
        rng = np.random.default_rng()
        dim = rng.uniform(0.3, 2, size=space_dim)
        return tuple(dim.astype(float))

    @pytest.fixture(params=[1, 1_000_000])  # M
    def _data(self, space_dim, bbox_dim, ndi, width, request) -> dict:
        # (input, output) pairs obtained when calling grid_cluster().
        xp = ndi.module()
        rng = xp.random.default_rng()

        bbox_max = 5 * xp.r_[bbox_dim]
        M = request.param

        offset = rng.standard_normal(space_dim)
        x = rng.uniform(0, 1, size=(M, space_dim)).astype(width.value)
        x *= bbox_max
        x += offset

        x_idx, cl_info = pxm.grid_cluster(x, bbox_dim)

        return dict(
            x=x,
            x_idx=x_idx,
            cl_info=cl_info,
        )


class TestFuseCluster:
    def test_value(self, space_dim, bbox_dim, _data):
        # * (in, out) shapes are consistent
        # * all points accounted for
        # * clF_info well formed
        # * cluster max extent <= bbox_dim
        # * #clusters <= max allowed count
        # * #clusters <= initial #clusters

        # (in, out) shapes are consistent
        x = _data["x"]
        xF_idx = _data["xF_idx"]
        clF_info = _data["clF_info"]
        assert (x.ndim == 2) and (x.shape[1] == space_dim)
        assert clF_info.ndim == 1
        assert (xF_idx.ndim == 1) and (len(x) == len(xF_idx))

        ndi = pxd.NDArrayInfo.from_obj(x)
        xp = ndi.module()
        M = len(x)
        L = len(clF_info) - 1

        # All points accounted for
        assert M == len(xF_idx)
        assert M == len(xp.unique(xF_idx))

        # clF_info well formed, i.e.
        # * clF_info is strictly monotonic -> no empty clusters
        # * clF_info[-1] goes beyond len(x)
        assert xp.all(clF_info[1:] > clF_info[:-1])
        assert M == clF_info[-1]

        # cluster max extent <= bbox_dim
        bbox_dim = xp.array(bbox_dim)
        for q in range(L):
            select = slice(clF_info[q], clF_info[q + 1])
            x_cl = x[xF_idx[select]]
            ptp_cl = xp.ptp(x_cl, axis=0)  # (D,)
            assert xp.all(ptp_cl <= bbox_dim)

        # #clusters <= max allowed count
        N_cl_max = 5**space_dim  # 5 hard-coded in Fixture[_data]
        assert L <= N_cl_max

        # #clusters <= initial #clusters
        assert len(clF_info) <= len(_data["cl_info"])

    def test_backend(self, ndi, _data):
        # (input, output) backends match
        get_backend = lambda _: pxd.NDArrayInfo.from_obj(_)

        assert get_backend(_data["xF_idx"]) == ndi
        assert get_backend(_data["clF_info"]) == ndi

    def test_prec(self, _data):
        # outputs are integer arrays
        get_dtype = lambda _: _.dtype

        assert get_dtype(_data["xF_idx"]) == np.int64
        assert get_dtype(_data["clF_info"]) == np.int64

    # Fixtures ----------------------------------------------------------------
    @pytest.fixture(params=[1, 2, 3])
    def space_dim(self, request) -> int:
        return request.param

    @pytest.fixture
    def bbox_dim(self, space_dim) -> tuple[float]:
        rng = np.random.default_rng()
        dim = rng.uniform(0.3, 2, size=space_dim)
        return tuple(dim.astype(float))

    @pytest.fixture(params=[1, 1_000_000])  # M
    def _data(self, space_dim, bbox_dim, ndi, width, request) -> dict:
        # (input, output) pairs obtained when calling fuse_cluster().
        xp = ndi.module()
        rng = xp.random.default_rng()

        bbox_max = 5 * xp.r_[bbox_dim]
        M = request.param

        offset = rng.standard_normal(space_dim)
        x = rng.uniform(0, 1, size=(M, space_dim)).astype(width.value)
        x *= bbox_max
        x += offset

        x_idx, cl_info = pxm.grid_cluster(x, bbox_dim)
        xF_idx, clF_info = pxm.fuse_cluster(x, x_idx, cl_info, bbox_dim)

        return dict(
            x=x,
            x_idx=x_idx,
            cl_info=cl_info,
            xF_idx=xF_idx,
            clF_info=clF_info,
        )


class TestBisectCluster:
    def test_value(self, _data, N_max):
        # * clB_info well formed
        # * #clusters >= input #clusters
        # * cluster size <= N_max

        cl_info = _data["cl_info"]
        clB_info = _data["clB_info"]

        ndi = pxd.NDArrayInfo.from_obj(cl_info)
        xp = ndi.module()
        Q = len(cl_info) - 1
        L = len(clB_info) - 1
        M = cl_info[-1]

        # clB_info well formed, i.e.
        # * clB_info is strictly monotonic -> no empty clusters
        # * clB_info[-1] goes beyond len(x)
        assert xp.all(clB_info[1:] > clB_info[:-1])
        assert 0 == clB_info[0]
        assert M == clB_info[-1]

        # #clusters >= input #clusters
        assert Q <= L

        # cluster size <= N_max
        cl_size = clB_info[1:] - clB_info[:-1]
        assert xp.all(cl_size <= N_max)

    def test_backend(self, ndi, _data):
        # (input, output) backends match
        get_backend = lambda _: pxd.NDArrayInfo.from_obj(_)

        assert get_backend(_data["clB_info"]) == ndi

    def test_prec(self, _data):
        # outputs are integer arrays
        get_dtype = lambda _: _.dtype

        assert get_dtype(_data["clB_info"]) == np.int64

    # Fixtures ----------------------------------------------------------------
    @pytest.fixture(params=[30, 5_001])
    def N_max(self, request) -> int:
        return request.param

    @pytest.fixture(params=[1, 500])  # Q
    def _data(self, ndi, N_max, request) -> dict:
        # (input, output) pairs obtained when calling bisect_cluster().
        xp = ndi.module()
        rng = xp.random.default_rng()

        Q = request.param  # Number of clusters

        cl_info = xp.zeros(Q + 1, dtype=np.int64)
        cl_info[1:] = rng.integers(low=1, high=5_000, size=Q).cumsum()
        clB_info = pxm.bisect_cluster(cl_info, N_max)

        return dict(
            cl_info=cl_info,
            clB_info=clB_info,
        )
