import itertools

import numpy as np
import pytest
import pyxu.abc as pxa
import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.runtime as pxrt
import pyxu_tests.operator.conftest as conftest
import scipy.integrate as spi

import pyxu_nufft.math as pxm
from pyxu_nufft.math.kernel._kernel import _FSSPulse


# _FSSPulse instances look very similar to Operator instances, except:
#    - Only NUMPY (and later) CUPY inputs are supported.
#    - we only ever call [apply,applyF,argscale]().
#    - [apply,applyF]() do not take `runtime.getPrecision()` into account: output dtypes always match input dtypes.
#
# Due to this strong similarity, we piggy-back on the Operator test suite to validate the implementations.
# All meaningless tests are disabled by default.
#
# We do not test _FSSPulse.[support,supportF]() which are assumed correct from manual tests.
class OperatorMixin(conftest.MapT):
    disable_test = conftest.MapT.disable_test | {
        "test_chunk_apply",
        "test_chunk_call",
        "test_codim_rank",
        "test_codim_shape",
        "test_codim_size",
        "test_dim_rank",
        "test_dim_shape",
        "test_dim_size",
        "test_interface_asop",
        "test_interface_lipschitz",
        "test_interface",
        "test_interface2_lipschitz",
        "test_math_lipschitz",
        "test_precCM_apply",
        "test_precCM_call",
        "test_precCM_lipschitz",
        "test_value_asop",
    }
    forward_fields = (
        # Fields to be forwarded to wrapper Operator instance; see spec().
        "apply",
        "applyF",
        "_nb_apply",
        "_nb_applyF",
        "support",
        "supportF",
        "argscale",
    )

    @pytest.fixture(
        params=itertools.product(
            [
                pxd.NDArrayInfo.NUMPY,
                # pxd.NDArrayInfo.CUPY,  # not currently supported
            ],
            pxrt.Width,
        )
    )
    def spec(
        self,
        dim_shape,
        codim_shape,
        operator,
        request,
    ) -> tuple[pxt.OpT, pxd.NDArrayInfo, pxrt.Width]:
        ndi, width = request.param

        # Wrap `operator` into an Operator instance to run the test suite.
        op = pxa.Map(
            dim_shape=dim_shape,
            codim_shape=codim_shape,
        )
        for name in self.forward_fields:
            setattr(op, name, getattr(operator, name))

        return op, ndi, width

    # Internal helpers --------------------------------------------------------
    @pytest.fixture
    def dim_shape(self) -> pxt.NDArrayShape:
        return (5, 3, 4)

    @pytest.fixture
    def codim_shape(self, dim_shape) -> pxt.NDArrayShape:
        return dim_shape

    @pytest.fixture
    def data_math_lipschitz(self, dim_shape):
        pass

    @pytest.fixture
    def operator(self) -> _FSSPulse:
        # Override in sub-classes to instantiate the _FSSPulse object to test.
        raise NotImplementedError

    def test_math_fourier(self, op, ndi):
        # \int_{-s}^{s} op.apply(x) dx == op.applyF(0)
        #
        # One could also check
        #     \int_{-sF}^{sF} op.applyF(v) dv ~= op.apply(0)
        # but we do not do this since op.supportF() is only approximate.
        self._skip_if_disabled()
        self._skip_unless_NUMPY_CUPY(ndi)

        func = op._nb_apply()
        funcF = op._nb_applyF()
        s = op.support()

        z0, *_ = spi.quad(func, -s, s)
        z0_gt = funcF(0)
        assert np.allclose(z0, z0_gt)


class TestBox(OperatorMixin):
    @pytest.fixture
    def operator(self) -> _FSSPulse:
        return pxm.Box(support=1)

    @pytest.fixture
    def data_apply(self, dim_shape):
        rng = np.random.default_rng()
        dim_size = np.prod(dim_shape)

        x = rng.uniform(-1.5, 1.5, dim_size)
        y = np.zeros_like(x)
        y[np.fabs(x) <= 1] = 1

        return dict(
            in_=dict(arr=x.reshape(dim_shape)),
            out=y.reshape(dim_shape),
        )


class TestTriangle(OperatorMixin):
    @pytest.fixture
    def operator(self) -> _FSSPulse:
        return pxm.Triangle(support=1)

    @pytest.fixture
    def data_apply(self, dim_shape):
        rng = np.random.default_rng()
        dim_size = np.prod(dim_shape)

        x = rng.uniform(-1.5, 1.5, dim_size)
        y = np.clip(1 - np.fabs(x), 0, None)

        return dict(
            in_=dict(arr=x.reshape(dim_shape)),
            out=y.reshape(dim_shape),
        )


class TestKaiserBessel(OperatorMixin):
    forward_fields = OperatorMixin.forward_fields + ("_beta",)

    @pytest.fixture(params=[0.3, 1])
    def kb_beta(self, request) -> float:
        return request.param

    @pytest.fixture
    def operator(self, kb_beta) -> _FSSPulse:
        return pxm.KaiserBessel(
            beta=kb_beta,
            support=1,
        )

    @pytest.fixture
    def data_apply(self, kb_beta, dim_shape):
        rng = np.random.default_rng()
        dim_size = np.prod(dim_shape)

        x = rng.uniform(-1.5, 1.5, dim_size)
        y = np.i0(kb_beta * np.sqrt(np.clip(1 - x**2, 0, None)))
        y /= np.i0(kb_beta)
        y[np.fabs(x) > 1] = 0

        return dict(
            in_=dict(arr=x.reshape(dim_shape)),
            out=y.reshape(dim_shape),
        )
