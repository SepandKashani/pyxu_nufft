import collections
import collections.abc as cabc
import concurrent.futures as cf
import operator
import warnings

import numpy as np
import pyxu.abc as pxa
import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.info.warning as pxw
import pyxu.runtime as pxrt
import pyxu.util as pxu
import scipy.optimize as sopt

import pyxu_nufft.math as pxm
from pyxu_nufft.math.cluster import ClusterMapping, PointIndex

isign_default = 1
spp_default = 5
upsampfac_default = 2
T_default = 2 * np.pi
Tc_default = 0
enable_warnings_default = True

__all__ = [
    "NUFFT1",
    "NUFFT2",
    "NUFFT3",
]


class NUFFT1(pxa.LinOp):
    r"""
    Type-1 Non-Uniform FFT :math:`\mathbb{A}: \mathbb{C}^{M} \to \mathbb{C}^{L_{1} \times\cdots\times L_{D}}`.

    NUFFT1 approximates, up to a requested relative accuracy :math:`\varepsilon > 0`, the following exponential sum:

      .. math::

         v_{\mathbf{n}} = (\mathbf{A} \mathbf{w})_{n} = \sum_{m=1}^{M} w_{m} e^{j \cdot s \cdot 2\pi \langle \mathbf{n},
         \mathbf{x}_{m} / \mathbf{T} \rangle},

      where

      * :math:`s \in \pm 1` defines the sign of the transform;
      * :math:`\mathbf{n} \in \{ -N_{1}, \ldots N_{1} \} \times\cdots\times \{ -N_{D}, \ldots, N_{D} \}`, with
        :math:`L_{d} = 2 * N_{d} + 1`;
      * :math:`\{\mathbf{x}_{m}\}_{m=1}^{M} \in [T_{c_{1}} - \frac{T_{1}}{2}, T_{c_{1}} + \frac{T_{1}}{2}]
        \times\cdots\times [T_{c_{D}} - \frac{T_{D}}{2}, T_{c_{D}} + \frac{T_{D}}{2}]` are non-uniform support points;
      * :math:`\mathbf{w} \in \mathbb{C}^{M}` are weights associated with :math:`\{\mathbf{x}\}_{m=1}^{M}`;
      * :math:`\mathbf{T} \in \mathbb{R}_{+}^{D}` and :math:`\mathbf{T_{c}} \in \mathbb{R}^{D}`.

      Concretely NUFFT1 computes approximations to the (scaled) Fourier Series coefficients of the :math:`T`-periodic
      function:

      .. math::

         \tilde{f}(\mathbf{x}) = \sum_{\mathbf{q} \in \mathbb{Z}^{D}} \sum_{m=1}^{M} w_{m} \delta(\mathbf{x} -
         \mathbf{x}_{m} - \mathbf{q} \odot \mathbf{T}),

         v_{\mathbf{n}} = \left( \prod_{d} T_{d} \right) \tilde{f}_{\mathbf{n}}^{FS}.

    .. rubric:: Implementation Notes

    * :py:class:`~pyxu_nufft.operator.NUFFT1` is not **precision-agnostic**: it will only work on NDArrays with the
      same dtype as `x`.  A warning is emitted if inputs must be cast to the support dtype.
    * :py:class:`~pyxu_nufft.operator.NUFFT1` instances are **not arraymodule-agnostic**: they will only work with
      NDArrays belonging to the same array module as `x`. Currently only NUMPY.
    """

    def __init__(
        self,
        x: pxt.NDArray,
        N: tuple[int],
        *,
        isign: int = isign_default,
        spp: tuple[int] = spp_default,
        upsampfac: tuple[float] = upsampfac_default,
        T: tuple[float] = T_default,
        Tc: tuple[float] = Tc_default,
        enable_warnings: bool = enable_warnings_default,
        fft_kwargs: dict = None,
        spread_kwargs: dict = None,
        **kwargs,
    ):
        r"""
        Parameters
        ----------
        x: NDArray
            (M, D) support points :math:`\mathbf{x}_{m} \in [T_{c_{1}} - \frac{T_{1}}{2}, T_{c_{1}} + \frac{T_{1}}{2}]
            \times\cdots\times [T_{c_{D}} - \frac{T_{D}}{2}, T_{c_{D}} + \frac{T_{D}}{2}]`.
        N: int, tuple[int]
            Number of coefficients [-N,...,N] to compute per dimension.
        isign: 1, -1
            Sign :math:`s` of the transform.
        spp: int, tuple[int]
            Samples-per-pulse, i.e. the width of the spreading kernel in each dimension.  Must be odd-valued.
        upsampfac: float, tuple[float]
            NUFFT upsampling factors :math:`\sigma_{d} > 1`.
        T: float, tuple[float]
            (D,) scalar factors :math:`\mathbf{T} \in \mathbb{R}_{+}^{D}`, i.e. the periodicity of
            :math:`f(\mathbf{x})`.
        Tc: float, tuple[float]
            (D,) center of one period :math:`T_{c} \in \mathbb{R}^{D}`.
        enable_warnings: bool
            If ``True``, emit a warning in case of precision mis-match issues.
        fft_kwargs: dict
            kwargs forwarded to :py:class:`~pyxu.operator.FFT`.
        spread_kwargs: dict
            kwargs forwarded to :py:class:`~pyxu_nufft.operator.UniformSpread`.
        """
        # Put all variables in canonical form & validate ----------------------
        #   x: (M, D) array (NUMPY)
        #   N: (D,) int
        #   isign: {-1, +1}
        #   spp: (D,) int
        #   upsampfac: (D,) float
        #   T: (D,) float
        #   Tc: (D,) float
        #   fft_kwargs: dict
        #   spread_kwargs: dict
        if x.ndim == 1:
            x = x[:, np.newaxis]
        M, D = x.shape
        N = self._as_seq(N, D, int)
        isign = isign // abs(isign)
        upsampfac = self._as_seq(upsampfac, D, float)
        T = self._as_seq(T, D, float)
        Tc = self._as_seq(Tc, D, float)
        spp = self._as_seq(spp, D, int)
        if fft_kwargs is None:
            fft_kwargs = dict()
        if spread_kwargs is None:
            spread_kwargs = dict()

        assert (N > 0).all()
        assert (spp > 0).all() & (spp % 2 == 1).all()
        assert (upsampfac > 1).all()
        assert (T > 0).all()

        # Initialize Operator -------------------------------------------------
        ndi = pxd.NDArrayInfo.from_obj(x)
        if ndi != pxd.NDArrayInfo.NUMPY:
            raise NotImplementedError

        self.cfg = self._init_metadata(N, isign, upsampfac, T, Tc, spp)
        super().__init__(
            dim_shape=(M, 2),
            codim_shape=(*self.cfg.L, 2),
        )
        self._x = pxrt.coerce(x)
        self._enable_warnings = bool(enable_warnings)
        self.lipschitz = np.sqrt(self.cfg.L.prod() * M)

        self._init_ops(fft_kwargs, spread_kwargs)

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            (..., M,2) weights :math:`\mathbf{w} \in \mathbb{C}^{M}` viewed as a real array. (See
            :py:func:`~pyxu.util.view_as_real`.)

        Returns
        -------
        out: NDArray
            (..., L1,...,LD,2) weights :math:`\mathbf{v} \in \mathbb{C}^{L_{1} \times\cdots\times L_{D}}` viewed as a
            real array. (See :py:func:`~pyxu.util.view_as_real`.)
        """
        arr = self._cast_warn(arr)
        xp = pxu.get_array_module(arr)

        # 1. Spread over periodic boundaries
        arr = xp.moveaxis(arr, -1, 0)  # (2,..., M)
        g = self._spread.apply(arr)  # (2,..., fft1+2P1,...,fftD+2PD)
        g = xp.moveaxis(g, 0, -1)  # (..., fft1+2P1,...,fftD+2PD,2)

        # 2. Remove periodic excess from lattice, but apply periodic effect beforehand
        g = self._pad.adjoint(g)  # (..., fft1,...,fftD,2)

        # 3. FFS of up-sampled gridded data
        scale = xp.array([1, -self.cfg.isign], dtype=arr.dtype)
        g *= scale
        g_FS = self._ffs.apply(g)  # (..., fft1,...,fftD,2)
        g_FS *= scale

        # 4. Remove up-sampled sections
        g_FS = self._trim.apply(g_FS)  # (..., L1,...,LD,2)

        # 5. Correct for spreading effect
        psi_FS = self._kernelFS(xp, g_FS.dtype, True)
        out = pxm.hadamard_outer(g_FS, *psi_FS)
        return out

    def capply(self, arr: pxt.NDArray) -> pxt.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            (..., M) weights :math:`\mathbf{w} \in \mathbb{C}^{M}`.

        Returns
        -------
        out: NDArray
            (..., L1,...,LD) weights :math:`\mathbf{v} \in \mathbb{C}^{L_{1} \times\cdots\times L_{D}}`.
        """
        x = pxu.view_as_real(pxu.require_viewable(arr))  # (..., M,2)
        y = self.apply(x)  # (..., L1,...,LD,2)
        out = pxu.view_as_complex(pxu.require_viewable(y))  # (..., L1,...,LD)
        return out

    @pxrt.enforce_precision(i="arr")
    def adjoint(self, arr: pxt.NDArray) -> pxt.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            (..., L1,...,LD,2) weights :math:`\mathbf{v} \in \mathbb{C}^{L_{1} \times\cdots\times L_{D}}` viewed as a
            real array. (See :py:func:`~pyxu.util.view_as_real`.)

        Returns
        -------
        out: NDArray
            (..., M,2) weights :math:`\mathbf{w} \in \mathbb{C}^{M}` viewed as a real array. (See
            :py:func:`~pyxu.util.view_as_real`.)
        """
        arr = self._cast_warn(arr)
        xp = pxu.get_array_module(arr)

        # 1. Correct  for spreading effect
        psi_FS = self._kernelFS(xp, arr.dtype, True)
        g_FS = pxm.hadamard_outer(arr, *psi_FS)  # (..., L1,...,LD,2)

        # 2. Go to up-sampled grid
        g_FS = self._trim.adjoint(g_FS)  # (..., fft1,...,fftD,2)

        # 3. FFS of up-sampled gridded data
        scale = xp.array([1, -self.cfg.isign], dtype=arr.dtype)
        g_FS *= scale
        g = self._ffs.adjoint(g_FS)  # (..., fft1,...,fftD,2)
        g *= scale

        # 4. Extend FFS mesh with periodic border effects
        g = self._pad.apply(g)  # (..., fft1+2P1,...,fftD+2PD,2)

        # 5. Interpolate over periodic boundaries
        g = xp.moveaxis(g, -1, 0)  # (2,..., fft1+2P1,...,fftD+2PD)
        out = self._spread.adjoint(g)  # (2,..., M)
        out = xp.moveaxis(out, 0, -1)  # (..., M,2)
        return out

    def cadjoint(self, arr: pxt.NDArray) -> pxt.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            (..., L1,...,LD) weights :math:`\mathbf{v} \in \mathbb{C}^{L_{1} \times\cdots\times L_{D}}`.

        Returns
        -------
        out: NDArray
            (..., M) weights :math:`\mathbf{w} \in \mathbb{C}^{M}`.
        """
        x = pxu.view_as_real(pxu.require_viewable(arr))  # (..., L1,...,LD,2)
        y = self.adjoint(x)  # (..., M,2)
        out = pxu.view_as_complex(pxu.require_viewable(y))  # (..., M)
        return out

    def asarray(self, **kwargs) -> pxt.NDArray:
        # Perform computation in `x`-backend ... ------------------------------
        xp = pxu.get_array_module(self._x)

        A = xp.stack(  # (L1,...,LD, D)
            xp.meshgrid(
                *[xp.arange(-n, n + 1) for n in self.cfg.N],
                indexing="ij",
            ),
            axis=-1,
        )
        B = xp.exp(  # (L1,...,LD, M)
            (2j * self.cfg.isign * np.pi)
            * xp.tensordot(
                A,
                self._x / self.cfg.T,
                axes=[[-1], [-1]],
            )
        )

        # ... then abide by user's backend/precision choice. ------------------
        xp = kwargs.get("xp", pxd.NDArrayInfo.NUMPY.module())
        dtype = kwargs.get("dtype", pxrt.getPrecision().value)
        C = xp.array(
            pxu.as_real_op(B, dim_rank=1),
            dtype=pxrt.Width(dtype).value,
        )
        return C

    # Internal Helpers --------------------------------------------------------
    def _cast_warn(self, arr: pxt.NDArray) -> pxt.NDArray:
        if arr.dtype == self._x.dtype:
            out = arr
        else:
            if self._enable_warnings:
                msg = "Computation may not be performed at the requested precision."
                warnings.warn(msg, pxw.PrecisionWarning)
            out = arr.astype(dtype=self._x.dtype)
        return out

    @staticmethod
    def _as_seq(x, N, _type=None) -> np.ndarray:
        if isinstance(x, cabc.Iterable):
            _x = tuple(x)
        else:
            _x = (x,)
        if len(_x) == 1:
            _x *= N  # broadcast
        assert len(_x) == N

        if _type is None:
            return _x
        else:
            return np.r_[tuple(map(_type, _x))]

    @staticmethod
    def _init_metadata(N, isign, upsampfac, T, Tc, spp) -> collections.namedtuple:
        # Compute all NUFFT1 parameters & store in namedtuple with (sub-)fields:
        # [All sequences are NumPy arrays]
        #
        # * D: int                    [Transform Dimensionality]
        # * N: (D,) int               [One-sided FS count /w upsampling]
        # * Ns: (D,) int              [One-sided FS count w/ upsampling]
        # * L: (D,) int               [Two-sided FS size  /w upsampling]
        # * Ls: (D,) int              [Two-sided FS size  w/ upsampling]
        # * T: (D,) float             [Function period]
        # * Tc: (D,) float            [Mid-point of period of interest]
        # * isign: int                [Sign of the exponent]
        # * upsampfac: (D,) float     [Upsampling factor \sigma]
        # * fft_shape: (D,) int       [FFT dimensions]
        # * kernel_spp: (D,) int      [Kernel sample count]
        # * kernel_alpha: (D,) float  [Kernel arg-scale factor]
        # * kernel_beta: (D,) float   [Kernel bandwidth (before arg-scaling)]
        # * z_start: (D,) float       [Lattice start coordinate /w padding]
        # * z_stop: (D,) float        [Lattice stop  coordinate /w padding]
        # * z_num: (D,) int           [Lattice node-count       /w padding]
        # * z_step: (D,) float        [Lattice pitch; useful to have explicitly]
        # * z_pad: (D,) int           [Padding size to add to lattice head/tail for periodic boundary conditions.]
        from pyxu.operator import FFT

        from pyxu_nufft.operator import FFS

        # FFT parameters
        D = len(N)
        L = 2 * N + 1
        Ns = np.ceil(upsampfac * N).astype(int)  # N^{\sigma}
        Ls = 2 * Ns + 1  # N_{FS}^{\sigma}
        fft_shape = np.r_[FFT.next_fast_len(Ls)]

        # Kernel parameters
        kernel_spp = spp
        kernel_alpha = (2 / T) * (fft_shape / kernel_spp)
        kernel_beta = (np.pi * kernel_spp) * (Ns / fft_shape)

        # Lattice parameters
        ffs = FFS(T=T, Tc=Tc, Nfs=Ls, Ns=fft_shape)
        nodes = ffs.sample_points(xp=np, dtype=pxrt.Width.DOUBLE.value)
        z_start = np.array([n[0] for n in nodes])
        z_stop = np.array([n[-1] for n in nodes])
        z_num = fft_shape
        z_step = T / fft_shape
        z_pad = kernel_spp // 2

        CONFIG = collections.namedtuple(
            "CONFIG",
            field_names=[
                "D",
                "N",
                "Ns",
                "L",
                "Ls",
                "T",
                "Tc",
                "isign",
                "upsampfac",
                "fft_shape",
                "kernel_spp",
                "kernel_alpha",
                "kernel_beta",
                "z_start",
                "z_stop",
                "z_num",
                "z_step",
                "z_pad",
            ],
        )
        return CONFIG(
            D=D,
            N=N,
            Ns=Ns,
            L=L,
            Ls=Ls,
            T=T,
            Tc=Tc,
            isign=isign,
            upsampfac=upsampfac,
            fft_shape=fft_shape,
            kernel_spp=kernel_spp,
            kernel_alpha=kernel_alpha,
            kernel_beta=kernel_beta,
            z_start=z_start,
            z_stop=z_stop,
            z_num=z_num,
            z_step=z_step,
            z_pad=z_pad,
        )

    def _init_ops(self, fft_kwargs, spread_kwargs):
        from pyxu.operator import Pad, Trim

        from pyxu_nufft.operator import FFS, UniformSpread

        self._spread = UniformSpread(  # spreads support points onto uniform lattice
            x=self._x,
            z=dict(
                start=self.cfg.z_start - self.cfg.z_step * self.cfg.z_pad,
                stop=self.cfg.z_stop + self.cfg.z_step * self.cfg.z_pad,
                num=self.cfg.z_num + 2 * self.cfg.z_pad,
            ),
            kernel=[
                pxm.KaiserBessel(support=1 / a, beta=b)
                for (a, b) in zip(
                    self.cfg.kernel_alpha,
                    self.cfg.kernel_beta,
                )
            ],
            enable_warnings=self._enable_warnings,
            **spread_kwargs,
        )
        self._pad = Pad(  # applies periodic border effects after spreading
            dim_shape=(*self.cfg.z_num, 2),
            pad_width=(*self.cfg.z_pad, 0),
            mode="wrap",
        )
        self._ffs = FFS(  # FFS transform on up-sampled gridded data
            T=self.cfg.T,
            Tc=self.cfg.Tc,
            Nfs=self.cfg.Ls,
            Ns=self.cfg.fft_shape,
            **fft_kwargs,
        )
        self._trim = Trim(  # removes up-sampled FFS sections
            dim_shape=(*self.cfg.fft_shape, 2),
            trim_width=[
                (ns - n, ns - n + tot - ls)
                for (n, ns, ls, tot) in zip(
                    self.cfg.N,
                    self.cfg.Ns,
                    self.cfg.Ls,
                    self.cfg.fft_shape,
                )
            ]
            + [(0, 0)],
        )

    def _kernelFS(self, xp: pxt.ArrayModule, dtype: pxt.DType, invert: bool) -> list[pxt.NDArray]:
        # Returns
        # -------
        # psi_FS: list[NDArray]
        #     (D+1,) kernel FS coefficients (1D), or their reciprocal.
        #     The trailing dimension is just there to operate on real-valued views directly.
        psi_FS = [None] * self.cfg.D + [xp.ones(2, dtype=dtype)]
        f = xp.reciprocal if invert else lambda _: _
        for d in range(self.cfg.D):
            psi = self._spread._kernel[d]
            T = self.cfg.T[d]
            N = self.cfg.N[d]

            pFS = psi.applyF(xp.arange(-N, N + 1) / T) / T
            psi_FS[d] = f(pFS).astype(dtype)
        return psi_FS


def NUFFT2(
    x: pxt.NDArray,
    N: tuple[int],
    *,
    isign: int = isign_default,
    spp: tuple[int] = spp_default,
    upsampfac: tuple[float] = upsampfac_default,
    T: tuple[float] = T_default,
    Tc: tuple[float] = Tc_default,
    enable_warnings: bool = enable_warnings_default,
    fft_kwargs: dict = None,
    spread_kwargs: dict = None,
    **kwargs,
) -> pxt.OpT:
    r"""
    Type-2 Non-Uniform FFT :math:`\mathbb{A}: \mathbb{C}^{L_{1} \times\cdots\times L_{D}} \to \mathbb{C}^{M}`.

    NUFFT2 approximates, up to a requested relative accuracy :math:`\varepsilon > 0`, the following exponential sum:

      .. math::

         \mathbf{w}_{m} = (\mathbf{A} \mathbf{v})_{m} = \sum_{\mathbf{n}} v_{\mathbf{n}} e^{j \cdot s \cdot 2\pi \langle \mathbf{n}, \mathbf{x}_{m} / \mathbf{T} \rangle},

      where

      * :math:`s \in \pm 1` defines the sign of the transform;
      * :math:`\mathbf{n} \in \{ -N_{1}, \ldots N_{1} \} \times\cdots\times \{ -N_{D}, \ldots, N_{D} \}`, with
        :math:`L_{d} = 2 * N_{d} + 1`;
      * :math:`\{\mathbf{x}_{m}\}_{m=1}^{M} \in [T_{c_{1}} - \frac{T_{1}}{2}, T_{c_{1}} + \frac{T_{1}}{2}]
        \times\cdots\times [T_{c_{D}} - \frac{T_{D}}{2}, T_{c_{D}} + \frac{T_{D}}{2}]` are non-uniform support points;
      * :math:`\mathbf{v} \in \mathbb{C}^{L_{1} \times\cdots\times L_{D}}` are weights;
      * :math:`\mathbf{T} \in \mathbb{R}_{+}^{D}` and :math:`\mathbf{T_{c}} \in \mathbb{R}^{D}`.

      Concretely NUFFT2 can be interpreted as computing approximately non-uniform samples of a :math:`T`-periodic
      function from its Fourier Series coefficients.  It is the adjoint of a type-1 NUFFT.

    .. rubric:: Implementation Notes

    * :py:func:`~pyxu_nufft.operator.NUFFT2` is not **precision-agnostic**: it will only work on NDArrays with the
      same dtype as `x`.  A warning is emitted if inputs must be cast to the support dtype.
    * :py:func:`~pyxu_nufft.operator.NUFFT2` instances are **not arraymodule-agnostic**: they will only work with
      NDArrays belonging to the same array module as `x`. Currently only NUMPY backends are supported.


    Parameters
    ----------
    x: NDArray
        (M, D) support points :math:`\mathbf{x}_{m} \in [T_{c_{1}} - \frac{T_{1}}{2}, T_{c_{1}} + \frac{T_{1}}{2}]
        \times\cdots\times [T_{c_{D}} - \frac{T_{D}}{2}, T_{c_{D}} + \frac{T_{D}}{2}]`.
    N: int, tuple[int]
        Number of coefficients [-N,...,N] to compute per dimension.
    isign: 1, -1
        Sign :math:`s` of the transform.
    spp: int, tuple[int]
        Samples-per-pulse, i.e. the width of the spreading kernel in each dimension.  Must be odd-valued.
    upsampfac: float, tuple[float]
        NUFFT upsampling factors :math:`\sigma_{d} > 1`.
    T: float, tuple[float]
        (D,) scalar factors :math:`\mathbf{T} \in \mathbb{R}_{+}^{D}`, i.e. the periodicity of :math:`f(\mathbf{x})`.
    Tc: float, tuple[float]
        (D,) center of one period :math:`T_{c} \in \mathbb{R}^{D}`.
    enable_warnings: bool
        If ``True``, emit a warning in case of precision mis-match issues.
    fft_kwargs: dict
        kwargs forwarded to :py:class:`~pyxu.operator.FFT`.
    spread_kwargs: dict
        kwargs forwarded to :py:class:`~pyxu_nufft.operator.UniformSpread`.
    """
    op1 = NUFFT1(
        x=x,
        N=N,
        isign=-isign,
        spp=spp,
        upsampfac=upsampfac,
        T=T,
        Tc=Tc,
        enable_warnings=enable_warnings,
        fft_kwargs=fft_kwargs,
        spread_kwargs=spread_kwargs,
        **kwargs,
    )
    op2 = op1.T
    op2._name = "NUFFT2"

    # Expose c[apply,adjoint]()
    op2.capply = op1.cadjoint
    op2.cadjoint = op1.capply

    return op2


class NUFFT3(pxa.LinOp):
    r"""
    Type-3 Non-Uniform FFT :math:`\mathbb{A}: \mathbb{C}^{M} \to \mathbb{C}^{N}`.

    NUFFT3 approximates, up to a requested relative accuracy :math:`\varepsilon > 0`, the following exponential sum:

    .. math::

       z_{n} = (\mathbf{A} \mathbf{w})_{n} = \sum_{m=1}^{M} w_{m} e^{j \cdot s \cdot 2\pi \langle \mathbf{v}_{n},
       \mathbf{x}_{m} \rangle},

    where

    * :math:`s \in \pm 1` defines the sign of the transform;
    * :math:`\{ \mathbf{x}_{m} \in \mathbb{R}^{D} \}_{m=1}^{M} are non-uniform support points;
    * :math:`\{ \mathbf{v}_{n} \in \mathbb{R}^{D} \}_{n=1}^{N} are non-uniform frequencies;
    * :math:`\mathbf{w} \in \mathbb{C}^{M}` are weights associated with :math:`\{\mathbf{x}\}_{m=1}^{M}`.

    Concretely NUFFT3 computes approximately samples of the Fourier Transform of the function:

    .. math::

       f(\mathbf{x}) = \sum_{m=1}^{M} w_{m} \delta(\mathbf{x} - \mathbf{x}_{m}).

    Notes
    -----
    .. rubric:: Implementation Notes

    * :py:class:`~pyxu_nufft.operator.NUFFT3` is not **precision-agnostic**: it will only work on NDArrays with the
      same dtype as (`x`,`v`).  A warning is emitted if inputs must be cast to their dtype.
    * :py:class:`~pyxu_nufft.operator.NUFFT3` instances are **not arraymodule-agnostic**: they will only work with
      NDArrays belonging to the same array module as (`x`,`v`). Currently only NUMPY backends are supported.
    * The complexity and memory footprint of the type-3 NUFFT can be arbitrarily large for poorly-centered data, or for
      data with a large spread.  Memory consumption can be significantly reduced by evaluating the type-3 summation in
      chunks:

      .. math::

         \begin{align}
             (2)\;\; &z_{k}
             =
             \sum_{p=1}^{P}
             \sum_{j \in \mathcal{M}_{p}} w_{j}
             e^{j 2\pi \cdot s \cdot \langle \mathbf{v}_{k}, \mathbf{x}_{j} \rangle },
             \quad &
             k\in \mathcal{N}_{q},
             \quad q=1,\ldots, Q, \quad & \text{Type 3 (chunked)}
         \end{align}

      where :math:`\{\mathcal{M}_1, \ldots, \mathcal{M}_P\}` and :math:`\{\mathcal{N}_1, \ldots, \mathcal{N}_Q\}` are
      *partitions* of the sets :math:`\{1, \ldots, M\}` and  :math:`\{1, \ldots, N\}` respectively.
    """

    def __init__(
        self,
        x: pxt.NDArray,
        v: pxt.NDArray,
        *,
        isign: int = isign_default,
        spp: tuple[int] = spp_default,
        upsampfac: tuple[float] = upsampfac_default,
        enable_warnings: bool = enable_warnings_default,
        fft_kwargs: dict = None,
        spread_kwargs: dict = None,
        chunked: bool = False,
        **kwargs,
    ):
        r"""
        Parameters
        ----------
        x: NDArray
            (M, D) support points :math:`\mathbf{x}_{m} \in \mathbb{R}^{D}`.
        v: NDArray
            (N, D) frequencies :math:`\mathbf{v}_{n} \in \mathbb{R}^{D}`.
        isign: 1, -1
            Sign :math:`s` of the transform.
        spp: int, tuple[int]
            Samples-per-pulse, i.e. the width of the spreading kernel in each dimension.  Must be odd-valued.
        upsampfac: float, tuple[float]
            NUFFT upsampling factors :math:`\sigma_{d}^{v} > 1`.
        enable_warnings: bool
            If ``True``, emit a warning in case of precision mis-match issues.
        fft_kwargs: dict
            kwargs forwarded to :py:class:`~pyxu.operator.FFT`.
        spread_kwargs: dict
            kwargs forwarded to :py:class:`~pyxu_nufft.operator.UniformSpread`.
        chunked: bool
            Evaluate NUFFT3 by partitioning (x, v) domains. (See notes.)
        kwargs: dict
            Extra kwargs passed to ``NUFFT3._partition_domain()``.
            Supported parameters are:

                * domain: str = "xv"
                    When performing auto-chunking, determines which domains are partitioned.
                    Must be one of:

                    * "x":  partition X-domain;
                    * "v":  partition V-domain;
                    * "xv": partition X/V-domains.
                * max_fft_mem: float = 10
                    Max FFT memory (MiB) allowed.
                * max_anisotropy: float = 5
                    Max tolerated (normalized) anisotropy ratio >= 1.

                    * Setting close to 1 favors cubeoid-shaped partitions of x/v space.
                    * Setting large allows x/v-partitions to be highly asymmetric.

            Default values are chosen if unspecified.


        Notes
        -----
        Setting `chunk=True` evaluates the NUFFT by partitioning (x,v) domains.  The partitioning is done automatically
        using ALGORITHM, and is affected by values of `domain`, `max_fft_mem` and `max_anisotropy`.
        """
        # Put all variables in canonical form & validate ----------------------
        #   x: (M, D) array (NUMPY)
        #   v: (N, D) array (NUMPY)
        #   isign: {-1, +1}
        #   spp: (D,) int
        #   upsampfac: (D,) float
        #   fft_kwargs: dict
        #   spread_kwargs: dict
        if x.ndim == 1:
            x = x[:, np.newaxis]
        M, D = x.shape
        if v.ndim == 1:
            v = v[:, np.newaxis]
        N, _ = v.shape
        isign = isign // abs(isign)
        upsampfac = self._as_seq(upsampfac, D, float)
        spp = self._as_seq(spp, D, _type=int)
        if fft_kwargs is None:
            fft_kwargs = dict()
        if spread_kwargs is None:
            spread_kwargs = dict()
        spread_kwargs["enable_warnings"] = bool(enable_warnings)
        kwargs = dict(
            domain=kwargs.get("domain", "xv"),
            max_fft_mem=kwargs.get("max_fft_mem", 10),
            max_anisotropy=kwargs.get("max_anisotropy", 5),
        )

        assert (spp > 0).all() & (spp % 2 == 1).all()
        assert (upsampfac > 1).all()
        assert operator.eq(
            pxd.NDArrayInfo.from_obj(x),
            pxd.NDArrayInfo.from_obj(v),
        ), "[x,v] Must belong to the same array backend."

        # Initialize Operator -------------------------------------------------
        ndi = pxd.NDArrayInfo.from_obj(x)
        if ndi != pxd.NDArrayInfo.NUMPY:
            raise NotImplementedError

        super().__init__(
            dim_shape=(M, 2),
            codim_shape=(N, 2),
        )
        self._x = pxrt.coerce(x)
        self._v = pxrt.coerce(v)
        self._enable_warnings = bool(enable_warnings)
        self.lipschitz = np.sqrt(N * M)

        self._x_idx, self._x_info, self._v_idx, self._v_info = self._partition_domain(
            x=self._x,
            v=self._v,
            upsampfac=upsampfac,
            chunked=chunked,
            **kwargs,
        )
        self.cfg = self._init_metadata(
            x=self._x,
            x_idx=self._x_idx,
            x_info=self._x_info,
            v=self._v,
            v_idx=self._v_idx,
            v_info=self._v_info,
            isign=isign,
            spp=spp,
            upsampfac=upsampfac,
        )
        self._init_ops(fft_kwargs, spread_kwargs)

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            (..., M,2) weights :math:`\mathbf{w} \in \mathbb{C}^{M}` viewed as a real array. (See
            :py:func:`~pyxu.util.view_as_real`.)

        Returns
        -------
        out: NDArray
            (..., N,2) weights :math:`\mathbf{z} \in \mathbb{C}^{N}` viewed as a real array. (See
            :py:func:`~pyxu.util.view_as_real`.)
        """
        x = pxu.view_as_complex(pxu.require_viewable(arr))  # (..., M)
        y = self.capply(x)  # (..., N)
        out = pxu.view_as_real(pxu.require_viewable(y))  # (..., N,2)
        return out

    def capply(self, arr: pxt.NDArray) -> pxt.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            (..., M) weights :math:`\mathbf{w} \in \mathbb{C}^{M}`.

        Returns
        -------
        out: NDArray
            (..., N) weights :math:`\mathbf{z} \in \mathbb{C}^{N}`.
        """
        arr = self._cast_warn(arr)
        xp = pxu.get_array_module(arr)

        c_width = pxrt.CWidth(arr.dtype)
        c_dtype = c_width.value
        r_width = c_width.real
        r_dtype = r_width.value

        sh = arr.shape[:-1]  # (...,)
        N_stack = len(sh)

        w = arr.conj() if (self.cfg.isign == 1) else arr

        # Spread signal: w -> gBE
        gBE = xp.zeros(  # (Nx_blk, ..., Nv_blk, fft1,...,fftD)
            shape=(self.cfg.Nx_blk, *sh, self.cfg.Nv_blk, *self.cfg.fft_shape),
            dtype=c_dtype,
        )
        with cf.ThreadPoolExecutor() as executor:
            fs = [None] * self.cfg.Nx_blk
            for nx in range(self.cfg.Nx_blk):
                fs[nx] = executor.submit(self._fw_spread, w=w, out=gBE, Nx_idx=nx)
            cf.wait(fs)  # guarantee all sub-blocks have been spread

        # Window signal: gBE -> h
        window = self._window(xp, r_dtype, True)
        h = pxm.hadamard_outer(gBE, *window)  # (Nx_blk, ..., Nv_blk, fft1,...,fftD)

        # FFS: h -> h_FS
        select = [slice(_l) for _l in self.cfg.L]
        h_FS = self._ffs.capply(h)[..., *select]  # (..., Nv_blk, L1,...,LD)

        # Transpose (x,v) sub-block order
        h_FS = h_FS.swapaxes(0, N_stack + 1)  # (Nv_blk, ..., Nx_blk, L1,...,LD)

        # Interpolate signal: h_FS -> f_F
        f_F = xp.zeros((*sh, self.codim_shape[0]), dtype=c_dtype)  # (..., N)
        with cf.ThreadPoolExecutor() as executor:
            fs = [None] * self.cfg.Nv_blk
            for nv in range(self.cfg.Nv_blk):
                fs[nv] = executor.submit(self._fw_interpolate, h_FS=h_FS, out=f_F, Nv_idx=nv)
            cf.wait(fs)  # guarantee all sub-blocks have been interpolated

        out = f_F.conj() if (self.cfg.isign == 1) else f_F
        return out

    @pxrt.enforce_precision(i="arr")
    def adjoint(self, arr: pxt.NDArray) -> pxt.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            (..., N,2) weights :math:`\mathbf{z} \in \mathbb{C}^{N}` viewed as a real array. (See
            :py:func:`~pyxu.util.view_as_real`.)

        Returns
        -------
        out: NDArray
            (..., M,2) weights :math:`\mathbf{w} \in \mathbb{C}^{M}` viewed as a real array. (See
            :py:func:`~pyxu.util.view_as_real`.)
        """
        x = pxu.view_as_complex(pxu.require_viewable(arr))  # (..., N)
        y = self.cadjoint(x)  # (..., M)
        out = pxu.view_as_real(pxu.require_viewable(y))  # (..., M,2)
        return out

    def cadjoint(self, arr: pxt.NDArray) -> pxt.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            (..., N) weights :math:`\mathbf{z} \in \mathbb{C}^{N}`.

        Returns
        -------
        out: NDArray
            (..., M) weights :math:`\mathbf{w} \in \mathbb{C}^{M}`.
        """
        arr = self._cast_warn(arr)
        xp = pxu.get_array_module(arr)

        c_width = pxrt.CWidth(arr.dtype)
        c_dtype = c_width.value
        r_width = c_width.real
        r_dtype = r_width.value

        sh = arr.shape[:-1]  # (...,)
        N_stack = len(sh)

        z = arr.conj() if (self.cfg.isign == 1) else arr

        # [Note: Architecture]
        # The chain of operations in capply() is
        #    w -> _fw_spread() -> gBE -> h -> h_FS -> trim() -> _fw_interpolate() -> z (A.K.A. f_F)
        # The chain of operations in cadjoint() is
        #    z -> _bw_spread() -> pad() -> h_FS -> h -> gBE -> _bw_interpolate() -> w
        #
        # capply.trim() can be done by creating a view of h_FS; it is hence fast.
        # cadjoint.pad() on the other hand requires allocation of an array, hence is comparatively slow.
        #
        # Solution: instead of padding h_FS(..., L1,...,LD) to size (..., fft1,...,fftD), we allocate an
        # (..., fft1,...,fftD)-sized buffer upfront and just fill the (..., L1,...,LD) sub-region in _bw_spread().
        buffer = xp.zeros(  # (Nv_blk, ..., Nx_blk, fft1,...,fftD)
            shape=(self.cfg.Nv_blk, *sh, self.cfg.Nx_blk, *self.cfg.fft_shape),
            dtype=c_dtype,
        )

        # Spread signal: z -> h_FS
        select = [slice(_l) for _l in self.cfg.L]
        h_FS = buffer[..., *select]  # (Nv_blk, ..., Nx_blk, L1,...,LD)
        assert not h_FS.flags.owndata  # ensure buffer trick works
        with cf.ThreadPoolExecutor() as executor:
            fs = [None] * self.cfg.Nv_blk
            for nv in range(self.cfg.Nv_blk):
                fs[nv] = executor.submit(self._bw_spread, z=z, out=h_FS, Nv_idx=nv)
            cf.wait(fs)  # guarantee all sub-blocks have been spread

        # Transpose (v,x) sub-block order
        h_FS = buffer.swapaxes(0, N_stack + 1)  # (Nx_blk, ..., Nv_blk, fft1,...,fftD)

        # FFS: h_FS -> h
        h = self._ffs.cadjoint(h_FS)  # (Nx_blk, ..., Nv_blk, fft1,...,fftD)

        # Window signal: h -> gBE
        window = self._window(xp, r_dtype, True)
        gBE = pxm.hadamard_outer(h, *window)  # (Nx_blk, ..., Nv_blk, fft1,...,fftD)

        # Interpolate signal: gBE -> w
        w = xp.zeros((*sh, self.dim_shape[0]), dtype=c_dtype)  # (..., M)
        with cf.ThreadPoolExecutor() as executor:
            fs = [None] * self.cfg.Nx_blk
            for nx in range(self.cfg.Nx_blk):
                fs[nx] = executor.submit(self._bw_interpolate, gBE=gBE, out=w, Nx_idx=nx)
            cf.wait(fs)  # guarantee all sub-blocks have been interpolated

        out = w.conj() if (self.cfg.isign == 1) else w
        return out

    def asarray(self, **kwargs) -> pxt.NDArray:
        # Perform computation in `xv`-backend ... -----------------------------
        xp = pxu.get_array_module(self._x)
        isign = self.cfg.isign
        A = xp.exp((2j * np.pi * isign) * (self._v @ self._x.T))

        # ... then abide by user's backend/precision choice. ------------------
        xp = kwargs.get("xp", pxd.NDArrayInfo.NUMPY.module())
        dtype = kwargs.get("dtype", pxrt.getPrecision().value)
        B = xp.array(
            pxu.as_real_op(A, dim_rank=1),
            dtype=pxrt.Width(dtype).value,
        )
        return B

    # Internal Helpers --------------------------------------------------------
    def _cast_warn(self, arr: pxt.NDArray) -> pxt.NDArray:
        r_width = pxrt.Width(self._x.dtype)
        x_cdtype = r_width.complex.value
        if arr.dtype == x_cdtype:
            out = arr
        else:
            if self._enable_warnings:
                msg = "Computation may not be performed at the requested precision."
                warnings.warn(msg, pxw.PrecisionWarning)
            out = arr.astype(dtype=x_cdtype)
        return out

    @staticmethod
    def _as_seq(x, N, _type=None) -> np.ndarray:
        if isinstance(x, cabc.Iterable):
            _x = tuple(x)
        else:
            _x = (x,)
        if len(_x) == 1:
            _x *= N  # broadcast
        assert len(_x) == N

        if _type is None:
            return _x
        else:
            return np.r_[tuple(map(_type, _x))]

    @staticmethod
    def _partition_domain(
        x: pxt.NDArray,
        v: pxt.NDArray,
        upsampfac: tuple[float],
        chunked: bool,
        domain: str,
        max_fft_mem: float,
        max_anisotropy: float,
    ):
        # Partition (x,v) into distinct clusters such that the FFTs performed are capped at a maximum size.
        #
        # This method assumes (x,v) are NUMPY arrays.
        #
        # Parameters
        # ----------
        # x: NDArray
        #     (M, D) point cloud.
        # v: NDArray
        #     (N, D) point cloud.
        # upsampfac: tuple[float]
        #     Upsampling factors \sigma_{d}^{v}.
        # chunked: bool
        #     Perform chunked evaluation.
        # domain: "x", "v", "xv"
        #     Domains to partition.
        # max_fft_mem: float [MiB]
        #     Maximum allowed FFT size.
        # max_anisotropy: float
        #     Maximum tolerated chunk anisotropy.
        #
        # Returns
        # -------
        # x_idx: PointIndex
        #     (M,) indices which sort `x` into cluster order.
        # x_info: ClusterMapping
        #     (Q+1,) `x` cluster start/stop indices.
        #
        #     ``x[x_idx][x_info[q] : x_info[q+1]]`` contains all `x` points in the q-th cluster.
        # v_idx: PointIndex
        #     (N,) indices which sort `v` into cluster order.
        # v_info: ClusterMapping
        #     (L+1,) `v` cluster start/stop indices.
        #
        #     ``v[v_idx][v_info[l] : v_info[l+1]]`` contains all `v` points in the l-th cluster.
        M, N = len(x), len(v)
        x_idx = np.arange(M, dtype=np.int64)
        x_info = np.array([0, M], dtype=np.int64)
        v_idx = np.arange(N, dtype=np.int64)
        v_info = np.array([0, N], dtype=np.int64)

        if chunked and (M > 1) and (N > 1):
            x_min, x_max = pxm.minmax(x)
            v_min, v_max = pxm.minmax(v)
            x_bbox_dim, v_bbox_dim = NUFFT3._infer_bbox_dims(
                x_ptp=x_max - x_min,
                v_ptp=v_max - v_min,
                upsampfac=upsampfac,
                domain=domain,
                max_fft_mem=max_fft_mem,
                max_anisotropy=max_anisotropy,
            )
            if "x" in domain:
                x_idx, x_info = pxm.grid_cluster(x, x_bbox_dim)
                x_idx, x_info = pxm.fuse_cluster(x, x_idx, x_info, x_bbox_dim)
            if "v" in domain:
                v_idx, v_info = pxm.grid_cluster(v, v_bbox_dim)
                v_idx, v_info = pxm.fuse_cluster(v, v_idx, v_info, v_bbox_dim)
        return x_idx, x_info, v_idx, v_info

    @staticmethod
    def _infer_bbox_dims(
        x_ptp: tuple[float],
        v_ptp: tuple[float],
        upsampfac: tuple[float],
        domain: str,
        max_fft_mem: float,
        max_anisotropy: float,
    ) -> tuple[tuple[float]]:
        # Find X box dimensions (X_1,...,X_D) and V box dimensions (V_1,...,V_D) such that:
        #
        # * number of NUFFT sub-problems is minimized;
        # * NUFFT sub-problems limited to user-specified memory budget;
        # * box dimensions are not too rectangular, i.e. anisotropic.
        #
        # Parameters
        # ----------
        # x_ptp, v_ptp: float[tuple]
        #     (D,) spread of the data in each domain.
        # upsampfac: tuple[float]
        #     Upsampling factors \sigma_{d}^{v}.
        # domain: "x", "v", "xv"
        #     Domain(s) to partition.
        #     Some constraints below are dropped if only (x,) or (v,) is to be sharded.
        # max_fft_mem: float
        #     Max FFT memory (MiB) allowed per sub-block.
        # max_anisotropy: float
        #     Max tolerated (normalized) anisotropy ratio >= 1.
        #
        # Returns
        # -------
        # X: tuple[float]
        #     (D,) X-box dimensions.
        # V: tuple[float]
        #     (D,) V-box dimensions.
        #
        #
        # Notes
        # -----
        # Given that
        #
        #     FFX_memory / element_itemsize
        #     \approx
        #     \prod_{k=1..D} (\sigma_k X_k V_k),
        #
        # we can solve an optimization problem to find the optimal (X_k, V_k) values.
        #
        #
        # Mathematical Formulation
        # ------------------------
        #
        # User input:
        #     1. FFT_memory: max memory budget per sub-problem
        #     2. alpha: max anisotropy >= 1
        #
        # minimize (objective_func)
        #     \prod_{k=1..D} X_k^{tot} / X_k                                                    # X-domain box-count
        #     *                                                                                 #      \times
        #     \prod_{k=1..D} V_k^{tot} / V_k                                                    # V-domain box-count
        # subject to
        #     1. \prod_{k=1..D} s_k X_k V_k <= FFX_mem / elem_size                              # sub-problem memory limit
        #     2. X_k <= X_k^{tot}                                                               # X-domain box size limited to X_k's spread
        #     3. V_k <= V_k^{tot}                                                               # V-domain box size limited to V_k's spread
        #     4. objective_func >= 1                                                            # at least 1 NUFFT sub-problem necessary
        #     5. 1/alpha <= (X_k / X_k^{tot}) / (X_q / X_q^{tot}) <= alpha                      # X-domain box size anisotropy limited
        #     6. 1/alpha <= (V_k / V_k^{tot}) / (V_q / V_q^{tot}) <= alpha                      # V-domain box size anisotropy limited
        #     7. 1/alpha <= (X_k / X_k^{tot}) / (V_q / V_q^{tot}) <= alpha                      # XV-domain box size anisotropy limited
        #
        # The problem above can be recast as a small LP and easily solved:
        # * Constraints (5,6) are dropped for 1D problems.
        # * Constraints (5,6,7) are [partially] dropped if only X|V is sharded.
        #
        #
        # Mathematical Formulation (LinProg)
        # ----------------------------------
        #
        # minimize
        #     c^{T} x
        # subject to
        #        A x <= b
        #    lb <= x <= ub
        # where
        #     x = [ln(X_1) ... ln(X_D), ln(V_1) ... ln(V_D)] \in \bR^{2D}
        #     c = [-1 ... -1]
        #     ub = [ln(X_1^{tot}) ... ln(X_D^{tot}), ln(V_1^{tot}) ... ln(V_D^{tot})]
        #     lb = [-inf ... -inf, -inf ... -inf]                   (domain = "xv")
        #          [-inf ... -inf, ln(V_1^{tot}) ... ln(V_D^{tot})] (domain = "x")
        #          [ln(X_1^{tot}) ... ln(X_D^{tot}), -inf ... -inf] (domain = "v")
        #     [A | b] = [  -c   | b1 = ln(FFX_mem / elem_size) - \sum_{k=1..D} ln(s_k)           ],  # sub-problem memory limit
        #               [  -c   | b2 = \sum_{k=1..D} ln(X_k^{tot}) + \sum_{k=1..D} ln(V_k^{tot}) ],  # at least 1 NUFFT sub-problem necessary
        #          (L1) [ M1, Z | b3 = ln(alpha) + ln(X_k^{tot}) - ln(X_q^{tot})                 ],  # X-domain box size anisotropy limited (upper limit, vector form)
        #          (L2) [-M1, Z | b4 = ln(alpha) - ln(X_k^{tot}) + ln(X_q^{tot})                 ],  # X-domain box size anisotropy limited (lower limit, vector form)
        #          (L3) [ Z, M1 | b5 = ln(alpha) + ln(V_k^{tot}) - ln(V_q^{tot})                 ],  # Z-domain box size anisotropy limited (upper limit, vector form)
        #          (L4) [ Z,-M1 | b6 = ln(alpha) - ln(V_k^{tot}) + ln(V_q^{tot})                 ],  # Z-domain box size anisotropy limited (lower limit, vector form)
        #          (L5) [   M2  | b7 = ln(alpha) + ln(V_k^{tot}) - ln(X_q^{tot})                 ],  # cross-domain box size anisotropy limited (upper limit, vector form)
        #          (L6) [  -M2  | b8 = ln(alpha) - ln(V_k^{tot}) + ln(X_q^{tot})                 ],  # cross-domain box size anisotropy limited (lower limit, vector form)
        #     ]
        #     Z = zeros(D_choose_2, D)
        #     M1 = (D_choose_2, D) (M)ask containing [-1, 1] per row
        #     M2 = (D**2, 2D) (M)ask containing [-1, 1] per row

        # Validate inputs ---------------------------------
        assert np.all((x_ptp := np.r_[x_ptp]) > 0)
        assert np.all((v_ptp := np.r_[v_ptp]) > 0)
        assert np.all((upsampfac := np.r_[upsampfac]) > 1)
        domain = domain.lower().strip()
        assert domain in ("x", "v", "xv")
        assert max_fft_mem > 0
        assert max_anisotropy >= 1

        # Build (c, lb, ub) -------------------------------
        D = len(x_ptp)
        c = -np.ones(2 * D)
        ub = np.log(np.r_[x_ptp, v_ptp])
        lb = np.log(np.r_[x_ptp, v_ptp])
        if "x" in domain:
            lb[:D] = -np.inf
        if "v" in domain:
            lb[-D:] = -np.inf

        # Build (A, b) ------------------------------------
        D_choose_2 = D * (D - 1) // 2
        D_pow_2 = D**2
        Z = np.zeros((D_choose_2, D))
        M1 = np.zeros((D_choose_2, D))
        M2 = np.zeros((D_pow_2, 2 * D))
        i, j = np.triu_indices(D, k=1)
        q, k = np.meshgrid(np.arange(D), np.arange(D), indexing="ij")
        q, k = q.ravel(), k.ravel()
        for _r, (_i, _j) in enumerate(zip(i, j)):
            M1[_r, _i] = -1
            M1[_r, _j] = 1
        for _r, (_q, _k) in enumerate(zip(q, k)):
            M2[_r, _q] = -1
            M2[_r, D + _k] = 1
        A = np.block(
            [
                [-c],  # memory limit
                [-c],  # at least 1 box
                [M1, Z],  # X_k anisotropy upper-bound
                [-M1, Z],  # X_k anisotropy lower-bound
                [Z, M1],  # V_k anisotropy upper-bound
                [Z, -M1],  # V_k anisotropy lower-bound
                [M2],  # XV_k anisotropy upper-bound
                [-M2],  # XV_k anisotropy lower-bound
            ]
        )
        Mx = np.log(x_ptp[j]) - np.log(x_ptp[i])
        Mv = np.log(v_ptp[j]) - np.log(v_ptp[i])
        Mxv = np.log(v_ptp[k]) - np.log(x_ptp[q])
        b = np.r_[
            np.log(max_fft_mem * (2**20) / x_ptp.dtype.itemsize) - np.log(upsampfac).sum(),  # b1: memory limit
            np.log(x_ptp).sum() + np.log(v_ptp).sum(),  # b2: at least 1 box
            np.log(max_anisotropy) + Mx,  # X_k anisotropy upper-bound
            np.log(max_anisotropy) - Mx,  # X_k anisotropy lower-bound
            np.log(max_anisotropy) + Mv,  # V_k anisotropy upper-bound
            np.log(max_anisotropy) - Mv,  # V_k anisotropy lower-bound
            np.log(max_anisotropy) + Mxv,  # XV_k anisotropy upper-bound
            np.log(max_anisotropy) - Mxv,  # XV_k anisotropy lower-bound
        ]

        # Filter anisotropy constraints -------------------
        if domain in ("x", "v"):
            # Drop XV anisotropy constraints
            idx = slice(0, -2 * D_pow_2)
            A, b = A[idx], b[idx]
        if D > 1:
            # Drop self-domain anisotropy constraints if needed
            idx = dict(
                x=np.r_[0, 1, 2 + np.arange(2 * D_choose_2)],
                v=np.r_[0, 1, np.arange(-2 * D_choose_2, 0)],
                xv=slice(None),
            )[domain]
            A, b = A[idx], b[idx]

        # Solve LinProg -----------------------------------
        res = sopt.linprog(
            c=c,
            A_ub=A,
            b_ub=b,
            bounds=np.stack([lb, ub], axis=1),
            method="highs",
        )
        if res.success:
            X = np.exp(res.x[:D])
            V = np.exp(res.x[-D:])
            return tuple(X), tuple(V)
        else:
            msg = "Auto-chunking failed given memory/anisotropy constraints."
            raise ValueError(msg)

    @staticmethod
    def _init_metadata(
        x: pxt.NDArray,
        x_idx: PointIndex,
        x_info: ClusterMapping,
        v: pxt.NDArray,
        v_idx: PointIndex,
        v_info: ClusterMapping,
        isign: int,
        spp: tuple[int],
        upsampfac: tuple[int],
    ) -> collections.namedtuple:
        # Compute all NUFFT3 parameters & store in namedtuple with (sub-)fields:
        # [All sequences are NumPy arrays]
        #
        # * D: int                    [Transform Dimensionality]
        # * N: (D,) int               [One-sided FS count w/ upsampling]
        # * L: (D,) int               [Two-sided FS size  w/ upsampling]
        # * Nx_blk: int               [Number of x-domain clusters]
        # * Xd: (D,) float            [Maximum x-domain cluster spread]
        # * Xc: (Nx_blk, D) float     [Cluster centroids; x-domain]
        # * Nv_blk: int               [Number of v-domain clusters]
        # * Vd: (D,) float            [Maximum v-domain cluster spread]
        # * Vc: (Nv_blk, D) float     [Cluster centroids; v-domain]
        # * s_max: (D,) float         [Kernel absolute width; upper bound]
        # * isign: int                [Sign of the exponent]
        # * upsampfac: (D,) float     [Upsampling factor \sigma_{v}]
        # * sigma_x: (D,) float       [Upsampling factor \sigma_{x}]
        # * fft_shape: (D,) int       [FFT dimensions]
        # * kernel_spp: (D,) int      [Kernel sample count]
        # * kernel1_alpha: (D,) float [Kernel arg-scale factor]
        # * kernel1_beta: (D,) float  [Kernel bandwidth (before arg-scaling)]
        # * kernel2_alpha: (D,) float [Kernel arg-scale factor]
        # * kernel2_beta: (D,) float  [Kernel bandwidth (before arg-scaling)]
        # * z_start: (D,) float       [Lattice start coordinate (absolute, centered at origin)]
        # * z_stop: (D,) float        [Lattice stop  coordinate (absolute, centered at origin)]
        # * z_num: (D,) int           [Lattice node-count]
        # * z_step: (D,) float        [Lattice pitch; useful to have explicitly]
        from pyxu.operator import FFT

        from pyxu_nufft.operator import FFS

        _, D = x.shape
        Xd_min = Vd_min = 1e-6

        # Kernel parameters (part 1)
        kernel_spp = np.r_[spp]

        # V-domain parameters
        Nv_blk = len(v_info) - 1
        V_min, V_max = pxm.group_minmax(v, v_idx, v_info)
        Vc = (V_max + V_min) / 2
        Vd = (V_max - V_min).max(axis=0)  # (D,)
        Vd += Vd_min

        # X-domain parameters
        Nx_blk = len(x_info) - 1
        s_max = kernel_spp / (2 * Vd * upsampfac)
        X_min, X_max = pxm.group_minmax(x, x_idx, x_info)
        Xc = (X_max + X_min) / 2
        Xd = (X_max - X_min).max(axis=0) + 2 * s_max  # (D,)
        Xd += Xd_min

        # FFT parameters
        sigma_x = 1.05 * np.ones(D)  # sufficient; all effort should be placed instead into sigma_v, i.e. `upsampfac`
        N = np.ceil(sigma_x * Xd * upsampfac * Vd / 2).astype(int)
        L = 2 * N + 1
        fft_shape = np.r_[FFT.next_fast_len(L)]

        # Kernel parameters (part 2)
        kernel1_alpha = (2 * fft_shape) / (kernel_spp * sigma_x * Xd)
        kernel1_beta = np.pi * kernel_spp * N / fft_shape
        kernel2_alpha = 2 * sigma_x * Xd / kernel_spp
        kernel2_beta = np.pi * kernel_spp * (1 - 0.5 / sigma_x)

        # Lattice parameters
        ffs = FFS(T=sigma_x * Xd, Tc=0, Nfs=L, Ns=fft_shape)
        nodes = ffs.sample_points(xp=np, dtype=pxrt.Width.DOUBLE.value)
        z_start = np.array([n[0] for n in nodes])
        z_stop = np.array([n[-1] for n in nodes])
        z_num = fft_shape
        z_step = (sigma_x * Xd) / fft_shape

        CONFIG = collections.namedtuple(
            "CONFIG",
            field_names=[
                "D",
                "N",
                "L",
                "Nx_blk",
                "Xd",
                "Xc",
                "Nv_blk",
                "Vd",
                "Vc",
                "s_max",
                "isign",
                "upsampfac",
                "sigma_x",
                "fft_shape",
                "kernel_spp",
                "kernel1_alpha",
                "kernel1_beta",
                "kernel2_alpha",
                "kernel2_beta",
                "z_start",
                "z_stop",
                "z_num",
                "z_step",
            ],
        )
        return CONFIG(
            D=D,
            N=N,
            L=L,
            Nx_blk=Nx_blk,
            Xd=Xd,
            Xc=Xc,
            Nv_blk=Nv_blk,
            Vd=Vd,
            Vc=Vc,
            s_max=s_max,
            isign=isign,
            upsampfac=upsampfac,
            sigma_x=sigma_x,
            fft_shape=fft_shape,
            kernel_spp=kernel_spp,
            kernel1_alpha=kernel1_alpha,
            kernel1_beta=kernel1_beta,
            kernel2_alpha=kernel2_alpha,
            kernel2_beta=kernel2_beta,
            z_start=z_start,
            z_stop=z_stop,
            z_num=z_num,
            z_step=z_step,
        )

    def _init_ops(self, fft_kwargs, spread_kwargs):
        from pyxu_nufft.operator import FFS, UniformSpread

        chunked = (self.cfg.Nx_blk > 1) or (self.cfg.Nv_blk > 1)
        if chunked:
            # When performing chunking, (x,v) sub-domains are already bbox-limited, so no need to restrict window ratios.
            # In this context spread/interp cluster formation just relies on domain bisection.
            spread_kwargs = spread_kwargs.copy()
            spread_kwargs["max_window_ratio"] = np.inf

        self._spread1 = dict()
        kernel1 = [
            pxm.KaiserBessel(support=1 / alpha, beta=beta)
            for (alpha, beta) in zip(self.cfg.kernel1_alpha, self.cfg.kernel1_beta)
        ]
        for nx in range(self.cfg.Nx_blk):
            a = self._x_info[nx]
            b = self._x_info[nx + 1]
            x_idx = self._x_idx[a:b]

            self._spread1[nx] = UniformSpread(
                x=self._x[x_idx],
                z=dict(
                    start=self.cfg.Xc[nx] + self.cfg.z_start,
                    stop=self.cfg.Xc[nx] + self.cfg.z_stop,
                    num=self.cfg.z_num,
                ),
                kernel=kernel1,
                **spread_kwargs,
            )

        self._ffs = FFS(
            T=self.cfg.sigma_x * self.cfg.Xd,
            Tc=0,
            Nfs=self.cfg.L,
            Ns=self.cfg.fft_shape,
            **fft_kwargs,
        )

        self._spread2 = dict()
        kernel2 = [
            pxm.KaiserBessel(support=1 / alpha, beta=beta)
            for (alpha, beta) in zip(self.cfg.kernel2_alpha, self.cfg.kernel2_beta)
        ]
        for nv in range(self.cfg.Nv_blk):
            a = self._v_info[nv]
            b = self._v_info[nv + 1]
            v_idx = self._v_idx[a:b]

            self._spread2[nv] = UniformSpread(
                x=self._v[v_idx] - self.cfg.Vc[nv],
                z=dict(
                    start=-self.cfg.N / (self.cfg.sigma_x * self.cfg.Xd),
                    stop=self.cfg.N / (self.cfg.sigma_x * self.cfg.Xd),
                    num=self.cfg.L,
                ),
                kernel=kernel2,
                **spread_kwargs,
            )

    def _window(self, xp, dtype, invert: bool) -> list[pxt.NDArray]:
        # Compute coefficients of the window function.
        #
        # Returns
        # -------
        # window: list[NDArray]
        #     (D,) window coefficients (1D), or their reciprocal.
        f = xp.reciprocal if invert else lambda _: _
        mesh = self._ffs.sample_points(xp, dtype)

        window = [None] * self.cfg.D
        for d in range(self.cfg.D):
            w_func = self._spread2[0]._kernel[d].applyF
            w = w_func(mesh[d])
            window[d] = f(w)
        return window

    def _kernel_F(
        self,
        v: pxt.NDArray,
        xp: pxt.ArrayModule,
        dtype: pxt.DType,
    ) -> pxt.NDArray:
        # Compute Fourier coefficients of the spreading kernel.
        #
        # Parameters
        # ----------
        # v: NDArray
        #     (N, D) frequencies.
        # xp: ArrayModule
        # dtype: DType
        #
        # Returns
        # -------
        # kernel_F: NDArray[real]
        #     (N,) Fourier coefficients.
        kernel_F = xp.ones(len(v), dtype=dtype)
        for d in range(self.cfg.D):
            psi_F = self._spread1[0]._kernel[d]
            kernel_F *= psi_F.applyF(v[:, d])  # (N,)
        return kernel_F

    def _fw_spread(self, w: pxt.NDArray, out: pxt.NDArray, Nx_idx: int) -> None:
        # For a given x-domain cluster, spread weights for all v-domain clusters.
        #
        # Parameters
        # ----------
        # w: NDArray[complex]
        #     (..., M) support weights. [NUMPY]
        # out: NDArray[complex]
        #     (Nx_blk, ..., Nv_blk, fft1,...,fftD) pre-allocated buffer in which to store the result.
        # Nx_idx: int
        #     X-domain cluster identifier.
        xp = pxu.get_array_module(w)

        # Sub-sample (w,)
        a = self._x_info[Nx_idx]
        b = self._x_info[Nx_idx + 1]
        x_idx = self._x_idx[a:b]  # (Mq,)
        w = w[..., x_idx]  # (..., Mq)

        # Modulate to baseband: w -> wBE
        spreadder = self._spread1[Nx_idx]
        mod = xp.exp((-2j * np.pi) * (self.cfg.Vc @ spreadder._x.T))  # (Nv_blk, Mq)
        wBE = w[..., np.newaxis, :] * mod  # (..., Nv_blk, Mq)

        # Spread signal onto FFT mesh: wBE -> gBE
        wBE = xp.stack([wBE.real, wBE.imag], axis=0)  # (2,..., Nv_blk, Mq)
        gBE = spreadder.apply(wBE)  # (2,..., Nv_blk, fft1,...,fftD)
        out.real[Nx_idx] = gBE[0]  # (..., Nv_blk, fft1,...,fftD)
        out.imag[Nx_idx] = gBE[1]
        return None

    def _fw_interpolate(self, h_FS: pxt.NDArray, out: pxt.NDArray, Nv_idx: int) -> None:
        # For a given v-domain cluster, interpolate FS coefficients for all x-domain clusters.
        #
        # Parameters
        # ----------
        # h_FS: NDArray[complex]
        #     (Nv_blk, ..., Nx_blk, L1,...,LD) FS coefficients. [NUMPY]
        # out: NDArray[complex]
        #     (..., N) pre-allocated buffer in which to store the result.
        # Nv_idx: int
        #     V-domain cluster identifier.
        xp = pxu.get_array_module(h_FS)
        r_dtype = pxrt.CWidth(h_FS.dtype).real.value

        # Sub-sample (h_FS,)
        g_FS = h_FS[Nv_idx]  # (..., Nx_idx, L1,...,LD)

        # Interpolate signal: h_FS -> gBE_F
        spreadder = self._spread2[Nv_idx]
        g_FS = xp.stack([g_FS.real, g_FS.imag], axis=0)  # (2,..., Nx_blk, L1,...,LD)
        gBE_F = spreadder.adjoint(g_FS)  # (2,..., Nx_blk, Nk)
        gBE_F = gBE_F[0] + 1j * gBE_F[1]  # (..., Nx_blk, Nk)

        # Modulate to passband: gBE_F -> g_F
        _v = spreadder._x  # (Nk, D)
        mod = xp.exp((-2j * np.pi) * (self.cfg.Xc @ _v.T))  # (Nx_blk, Nk)
        g_F = gBE_F
        g_F *= mod  # (..., Nx_blk, Nk)

        # Correct for spreading kernel effect (after reducing across sub-blocks): g_F -> f_F
        kernel_F = self._kernel_F(_v, xp, r_dtype)  # (Nk,)
        f_F = g_F.sum(axis=-2) / kernel_F  # (..., Nk)

        a = self._v_info[Nv_idx]
        b = self._v_info[Nv_idx + 1]
        v_idx = self._v_idx[a:b]
        out[..., v_idx] = f_F
        return None

    def _bw_spread(self, z: pxt.NDArray, out: pxt.NDArray, Nv_idx: int) -> None:
        # For a given v-domain cluster, spread weights for all x-domain clusters.
        #
        # Parameters
        # ----------
        # z: NDArray[complex]
        #     (..., N) spectral weights. [NUMPY]
        # out: NDArray[complex]
        #     (Nv_blk, ..., Nx_blk, L1,...,LD) pre-allocated buffer in which to store the result.
        # Nv_idx: int
        #     V-domain cluster identifier.
        xp = pxu.get_array_module(z)
        r_dtype = pxrt.CWidth(z.dtype).real.value

        # Sub-sample (z,)
        a = self._v_info[Nv_idx]
        b = self._v_info[Nv_idx + 1]
        v_idx = self._v_idx[a:b]  # (Nk,)
        z = z[..., v_idx]  # (..., Nk)

        # Correct for spreading kernel effect: z -> g_F
        spreadder = self._spread2[Nv_idx]
        _v = spreadder._x  # (Nk, D)
        kernel_F = self._kernel_F(_v, xp, r_dtype)  # (Nk,)
        g_F = z / kernel_F  # (..., Nk)

        # Modulate to baseband: g_F -> gBE_F
        mod = xp.exp((2j * np.pi) * (self.cfg.Xc @ _v.T))  # (Nx_blk, Nk)
        gBE_F = g_F[..., np.newaxis, :] * mod  # (..., Nx_blk, Nk)

        # Spread signal: gBE_F -> h_FS
        gBE_F = xp.stack([gBE_F.real, gBE_F.imag], axis=0)  # (2,..., Nx_blk, Nk)
        h_FS = spreadder.apply(gBE_F)  # (2,..., Nx_blk, L1,...,LD)
        out.real[Nv_idx] = h_FS[0]  # (..., Nx_blk, L1,...,LD)
        out.imag[Nv_idx] = h_FS[1]
        return None

    def _bw_interpolate(self, gBE: pxt.NDArray, out: pxt.NDArray, Nx_idx: int) -> None:
        # For a given x-domain cluster, interpolate spatial coefficients for all v-domain clusters.
        #
        # Parameters
        # ----------
        # gBE: NDArray[complex]
        #     (Nx_blk, ..., Nv_blk, fft1,...,fftD) FFS spatial coefficients. [NUMPY]
        # out: NDArray[complex]
        #     (..., M) pre-allocated buffer in which to store the result.
        # Nx_idx: int
        #     X-domain cluster identifier.
        xp = pxu.get_array_module(gBE)

        # Sub-sample (gBE,)
        gBE = gBE[Nx_idx]  # (..., Nv_blk, fft1,...,fftD)

        # Interpolate signal: gBE -> wBE
        spreadder = self._spread1[Nx_idx]
        gBE = xp.stack([gBE.real, gBE.imag], axis=0)  # (2,..., Nv_blk, fft1,...,fftD)
        wBE = spreadder.adjoint(gBE)  # (2,..., Nv_blk, Mq)
        wBE = wBE[0] + 1j * wBE[1]  # (..., Nv_blk, Mq)

        # Modulate to passband (before reducing across sub-blocks): wBE -> w
        mod = xp.exp((2j * np.pi) * (self.cfg.Vc @ spreadder._x.T))  # (Nv_blk, Mq)
        w = xp.sum(wBE * mod, axis=-2)  # (..., Mq)

        a = self._x_info[Nx_idx]
        b = self._x_info[Nx_idx + 1]
        x_idx = self._x_idx[a:b]
        out[..., x_idx] = w
        return None


class HVOXv1_FINUFFT(NUFFT3):
    # Compute NUFFT3 using HVOXv1 method, FINUFFT-backed

    def __init__(
        self,
        x: pxt.NDArray,
        v: pxt.NDArray,
        *,
        isign: int = isign_default,
        spp: tuple[int] = spp_default,
        upsampfac: tuple[float] = upsampfac_default,
        enable_warnings: bool = enable_warnings_default,
        fft_kwargs: dict = None,
        spread_kwargs: dict = None,
        chunked: bool = False,
        **kwargs,
    ):
        super().__init__(
            x=x,
            v=v,
            isign=isign,
            spp=spp,
            upsampfac=upsampfac,
            enable_warnings=enable_warnings,
            fft_kwargs=fft_kwargs,
            spread_kwargs=spread_kwargs,
            chunked=chunked,
            **kwargs,
        )

    def _init_ops(self, fft_kwargs, spread_kwargs):
        # Sub-tranforms are created at runtime to support batch-dimensions.
        self._nworkers=spread_kwargs.get("workers")

    def capply(self, arr: pxt.NDArray) -> pxt.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            (..., M) weights :math:`\mathbf{w} \in \mathbb{C}^{M}`.

        Returns
        -------
        out: NDArray
            (..., N) weights :math:`\mathbf{z} \in \mathbb{C}^{N}`.
        """
        arr = self._cast_warn(arr)
        xp = pxu.get_array_module(arr)

        c_width = pxrt.CWidth(arr.dtype)
        c_dtype = c_width.value

        sh = arr.shape[:-1]  # (...,)
        N_stack = len(sh)

        with cf.ThreadPoolExecutor() as executor:
            fs = [None] * (self.cfg.Nx_blk * self.cfg.Nv_blk)

            i = 0
            for nx in range(self.cfg.Nx_blk):
                for nv in range(self.cfg.Nv_blk):
                    fs[i] = executor.submit(
                        self._capply_part,
                        nx,
                        nv,
                        N_stack,
                        arr,
                    )
                    i += 1

        out = xp.zeros((*sh, self.codim_shape[0]), dtype=c_dtype)
        for future in cf.as_completed(fs):
            nx, nv, v = future.result()

            c = self._v_info[nv]
            d = self._v_info[nv + 1]
            v_idx = self._v_idx[c:d]
            out[..., v_idx] += v
        return out

    def _capply_part(
        self,
        Nx_idx: int,
        Nv_idx: int,
        N_stack: int,
        arr: np.ndarray,
    ) -> np.ndarray:
        from pyxu_finufft.operator import NUFFT3 as FINUFFT3

        # Compute equivalent `eps` based on FINUFFT paper
        #     eps = 10**(-p)
        #     p = min(spp) - 1
        p = min(self.cfg.kernel_spp) - 1.
        eps = float(10 ** (-p))

        a = self._x_info[Nx_idx]
        b = self._x_info[Nx_idx + 1]
        x_idx = self._x_idx[a:b]
        x = self._x[x_idx] * (2 * np.pi)  # FINUFFT computes /w 2\pi scale.

        c = self._v_info[Nv_idx]
        d = self._v_info[Nv_idx + 1]
        v_idx = self._v_idx[c:d]
        v = self._v[v_idx]

        op = FINUFFT3(
            x=x,
            v=v,
            isign=self.cfg.isign,
            eps=eps,
            enable_warnings=False,
            n_trans=N_stack,
            nthreads=self._nworkers,
        )

        w = arr[..., x_idx]
        v = op.capply(w)
        return (Nx_idx, Nv_idx, v)
