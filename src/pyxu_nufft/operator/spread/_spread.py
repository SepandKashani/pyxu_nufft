import collections.abc as cabc
import concurrent.futures as cf
import pathlib as plib
import string
import types
import warnings

import numpy as np
import pyxu.abc as pxa
import pyxu.info.config as pxcfg
import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.info.warning as pxw
import pyxu.runtime as pxrt
import pyxu.util as pxu

import pyxu_nufft.math as pxm

__all__ = [
    "UniformSpread",
]


class UniformSpread(pxa.LinOp):
    r"""
    :math:`D`-dimensional spreading operator :math:`A: \mathbb{R}^{M} \to \mathbb{R}^{N_{1} \times
    \cdots \times N_{D}}`.

    .. math::

       (A \, \mathbf{w})[n_{1}, \ldots, n_{D}]
       =
       \sum_{m = 1}^{M} w_{m} \phi(z_{n_{1}, \ldots, n_{D}} - x_{m}),

    .. math::

       (A^{*} \mathbf{v})_{m}
       =
       \sum_{n_{1}, \ldots, n_{D} = 1}^{N_{1}, \ldots, N_{D}}
       v[n_{1}, \ldots, n_{D}] \phi(z_{n_{1}, \ldots, n_{D}} - x_{m}),

    .. math::

       \mathbf{w} \in \mathbb{R}^{M},
       \quad
       \mathbf{v} \in \mathbb{R}^{N_{1} \times\cdots\times N_{D}},
       \quad
       z_{n_{1},\ldots,n_{D}} \in \mathcal{D},
       \quad
       \phi: \mathcal{K} \to \mathbb{R},

    where
    :math:`\mathcal{D} = [\alpha_{1}, \beta_{1}] \times\cdots\times [\alpha_{D}, \beta_{D}]` and
    :math:`\mathcal{K} = [-s_{1}, s_{1}] \times\cdots\times [-s_{D}, s_{D}]`,
    :math:`s_{d} > 0`.


    .. rubric:: Implementation Notes

    * :py:class:`~pyxu_nufft.operator.UniformSpread` is not **precision-agnostic**: it will only work on NDArrays with
      the same dtype as `x`.  A warning is emitted if inputs must be cast to the support dtype.
    * :py:class:`~pyxu_nufft.operator.UniformSpread` instances are **not arraymodule-agnostic**: they will only work
      with NDArrays belonging to the same array module as `x`. Only NUMPY backends are currently supported.
    * Spread/interpolation are performed efficiently via the algorithm described in [FINUFFT]_, i.e. partition
      :math:`\{\mathbf{x}_{m}\}` into sub-grids, spread onto each sub-grid, then add the results to the global grid.
      This approach works best when the kernel is *localized*. For kernels with huge support (w.r.t the full grid), spreading via a tensor contraction is preferable.
    """

    def __init__(
        self,
        x: pxt.NDArray,
        z: dict,
        kernel: cabc.Sequence,
        enable_warnings: bool = True,
        **kwargs,
    ):
        r"""
        Parameters
        ----------
        x: NDArray
            (M, D) support points :math:`\{x_{1},\ldots,x_{M}\}`.
        z: dict
            Lattice specifier, with keys:

            * `start`: (D,) values :math:`\{\alpha_{1}, \ldots, \alpha_{D}\} \in \mathbb{R}`.
            * `stop` : (D,) values :math:`\{\beta_{1}, \ldots, \beta_{D}\} \in \mathbb{R}`.
            * `num`  : (D,) values :math:`\{N_{1}, \ldots, N_{D}\} \in \mathbb{N}^{*}`.

            Scalars are broadcasted to all dimensions.

            The lattice is defined as:

            .. math::

               \left[z_{n_{1}, \ldots, n_{D}}\right]_{d}
               =
               \alpha_{d} + \frac{\beta_{d} - \alpha_{d}}{N_{d} - 1} n_{d},
               \quad
               n_{d} \in \{0, \ldots, N_{d}-1\}

        kernel: Sequence[obj]
            (D,) seperable kernel specifiers :math:`\phi_{d}: \mathcal{K}_{d} \to \mathbb{R}` such that

            .. math::

               \phi(\mathbf{x}) = \prod_{d=1}^{D} \phi_{d}(x_{d}).

            Functions should be objects with the following fields:

            * ``apply(np.ndarray) -> np.ndarray``, a ufunc to evaluate the kernel at any point :math:`x \in \mathbb{R}`.

            * ``support() -> float``, encoding the kernel's :math:`[-s, s]` support.
              Note that kernels must have symmetric support, but the kernel itself need not be symmetric.

        enable_warnings: bool
            If ``True``, emit a warning in case of precision mis-match issues.

        kwargs: dict
            Extra kwargs to configure :py:class:`~pyxu_nufft.operator.UniformSpread`.
            Supported parameters are:

            * `max_cluster_size`: int = 10_000

              Maximum number of support points per sub-grid/cluster.

            * `max_window_ratio`: float = 10

              Maximum size of the sub-grids, expressed as multiples of the kernel's support.

            * `workers`: int = 2 (# virtual cores)

              Number of threads used to spread sub-grids.
              Specifying `None` uses all cores.

            Default values are chosen if unspecified.

            Some guidelines to set these parameters:

            * The pair (`max_window_ratio`, `max_cluster_size`) determines the maximum memory requirements per
              sub-grid.
            * `workers` sub-grids are processed in parallel.
              Due to the Python GIL, the speedup is not linear with the number of workers.
              Choosing a small value (ex 2-4) seems to offer the best parallel efficiency.
            * `max_cluster_size` should be chosen large enough for there to be meaningful work done by each thread.
              If chosen too small, then many sub-grids need to be written to the global grid, which may introduce
              overheads.
            * `max_window_ratio` should be chosen based on the point distribution. Set it to `inf` if only cluster
              size matters.
        """
        # Put all internal variables in canonical form ------------------------
        #   x: (M, D) array (NUMPY)
        #   z: start: (D,)-float,
        #      stop : (D,)-float,
        #      num  : (D,)-int,
        #   kernel: tuple[obj]
        assert x.ndim in {1, 2}
        if x.ndim == 1:
            x = x[:, np.newaxis]
        M, D = x.shape

        kernel = self._as_seq(kernel, D)
        for k in kernel:
            assert hasattr(k, "apply"), "[Kernel] Missing apply()."
            assert hasattr(k, "support"), "[Kernel] Missing support()."
            s = k.support()
            assert s > 0, "[Kernel] Support must be non-zero."

        z["start"] = self._as_seq(z["start"], D, float)
        z["stop"] = self._as_seq(z["stop"], D, float)
        z["num"] = self._as_seq(z["num"], D, int)
        msg_lattice = "[z] Degenerate lattice detected."
        for d in range(D):
            alpha, beta, N = z["start"][d], z["stop"][d], z["num"][d]
            assert alpha <= beta, msg_lattice
            if alpha < beta:
                assert N >= 1, msg_lattice
            else:
                # Lattices with overlapping nodes are not allowed.
                assert N == 1, msg_lattice

        kwargs = {
            "max_cluster_size": kwargs.get("max_cluster_size", 10_000),
            "max_window_ratio": kwargs.get("max_window_ratio", 10),
            "workers": kwargs.get("workers", 2),
        }
        assert kwargs["max_cluster_size"] > 0
        assert kwargs["max_window_ratio"] >= 3

        # Object Initialization -----------------------------------------------
        super().__init__(
            dim_shape=M,
            codim_shape=z["num"],
        )
        self._x = pxrt.coerce(x)
        self._z = z
        self._kernel = kernel
        self._enable_warnings = bool(enable_warnings)
        self._kwargs = kwargs

        # Acceleration metadata -----------------------------------------------
        ndi = pxd.NDArrayInfo.from_obj(self._x)
        if ndi == pxd.NDArrayInfo.NUMPY:
            self._nb = self._gen_code(dim_rank=D, dtype=self._x.dtype)
            self._cl_info = self._build_cl_info(
                x=self._x,
                z=self._z,
                kernel=self._kernel,
                **self._kwargs,
            )
        else:
            raise NotImplementedError

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            (..., M) input weights :math:`\mathbf{w} \in \mathbb{R}^{M}`.

        Returns
        -------
        out: NDArray
            (..., N1,...,ND) lattice values :math:`\mathbf{v} \in \mathbb{R}^{N_{1} \times\cdots\times N_{D}}`.
        """
        xp = pxu.get_array_module(arr)
        arr = self._cast_warn(arr)
        sh = arr.shape[: -self.dim_rank]

        # Re-order/shape (x, w)
        Ns = int(np.prod(sh))
        x_idx = self._cl_info["x_idx"]
        x = self._x[x_idx]
        w = arr[..., x_idx].reshape(Ns, -1)  # (Ns, M)

        # Spread each cluster onto its own sub-grid
        Q = len(self._cl_info["cl_bound"]) - 1
        with cf.ThreadPoolExecutor(max_workers=self._kwargs["workers"]) as executor:
            fs = [executor.submit(self._spread, x=x, w=w, q=q) for q in range(Q)]

        # Update global grid
        out = xp.zeros((*sh, *self.codim_shape), dtype=arr.dtype)
        for f in cf.as_completed(fs):
            q, v = f.result()

            z_anchor = self._cl_info["z_anchor"][q]
            z_num = self._cl_info["z_num"][q]
            roi = [slice(n0, n0 + num) for (n0, num) in zip(z_anchor, z_num)]

            out[..., *roi] += v.reshape(*sh, *z_num)
        return out

    @pxrt.enforce_precision(i="arr")
    def adjoint(self, arr: pxt.NDArray) -> pxt.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            (..., N1,...,ND) lattice values :math:`\mathbf{v} \in \mathbb{R}^{N_{1} \times\cdots\times N_{D}}`.

        Returns
        -------
        out: NDArray
            (..., M) non-uniform weights :math:`\mathbf{w} \in \mathbb{R}^{M}`.
        """
        xp = pxu.get_array_module(arr)
        arr = self._cast_warn(arr)
        sh = arr.shape[: -self.codim_rank]

        # Re-order/shape (x, v)
        Ns = int(np.prod(sh))
        x_idx = self._cl_info["x_idx"]
        x = self._x[x_idx]
        v = arr.reshape(Ns, *self.codim_shape)  # (Ns, N1,...,ND)

        # Interpolate each sub-grid onto support points within.
        Q = len(self._cl_info["cl_bound"]) - 1
        with cf.ThreadPoolExecutor(max_workers=self._kwargs["workers"]) as executor:
            fs = [executor.submit(self._interpolate, x=x, v=v, q=q) for q in range(Q)]

        # Update global support
        out = xp.zeros((*sh, self.dim_size), dtype=arr.dtype)
        for f in cf.as_completed(fs):
            q, w = f.result()

            a = self._cl_info["cl_bound"][q]
            b = self._cl_info["cl_bound"][q + 1]
            idx = x_idx[a:b]

            out[..., idx] = w.reshape(*sh, b - a)
        return out

    def asarray(self, **kwargs) -> pxt.NDArray:
        # Perform computation in `x`-backend/precision ... --------------------
        xp = pxu.get_array_module(self._x)
        dtype = self._x.dtype

        lattice = self._lattice(xp, dtype, flatten=False)

        A = xp.ones((*self.codim_shape, *self.dim_shape), dtype=dtype)  # (N1,...,ND, M)
        for d in range(self.codim_rank):
            _l = lattice[d]  # (1,...,1,Nd,1,...,1)
            _x = self._x[:, d]  # (M,)
            _phi = self._kernel[d].apply
            _A = _phi(_l[..., np.newaxis] - _x)  # (1,...,1,Nd,1,...,1, M)
            A *= _A

        # ... then abide by user's backend/precision choice. ------------------
        xp = kwargs.get("xp", pxd.NDArrayInfo.NUMPY.module())
        dtype = kwargs.get("dtype", pxrt.getPrecision().value)
        B = xp.array(pxu.to_NUMPY(A), dtype=dtype)
        return B

    # Helper routines (internal) ----------------------------------------------
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
    def _as_seq(x, N, _type=None) -> tuple:
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
            return tuple(map(_type, _x))

    @staticmethod
    def _build_cl_info(
        x: np.ndarray,
        z: dict[str, tuple],
        kernel: tuple[pxt.OpT],
        **kwargs,
    ) -> dict[int, dict]:
        # Build acceleration metadata.
        #
        # * Partitions the support points into Q clusters.
        # * Identifies the sub-grids onto which each cluster is spread.
        #
        # Parameters
        # ----------
        # x: np.ndarray[float]
        #     (M, D) support points.
        # z: dict[str, tuple]
        #     Lattice (start, stop, num) specifier.
        # kernel: tuple[obj]
        #     (D,) axial kernels.
        # kwargs: dict
        #     Spreadder config info.
        #
        # Returns
        # -------
        # info: dict[str, np.ndarray]
        #     (Q,) cluster metadata, with fields:
        #
        #     * x_idx: np.ndarray[int64]
        #         (M,) indices to re-order `x` s.t. points in each cluster are sequential.
        #         Its length may be smaller than M if points do not contribute to the lattice in any way.
        #     * cl_bound: np.ndarray[int64]
        #         (Q+1,) indices into `x_idx` indicating where the q-th cluster's support points start/end.
        #         Cluster `q` contains support points `x[x_idx][cl_bound[q] : cl_bound[q+1]]`.
        #     * z_anchor: np.ndarray[int64]
        #         (Q, D) lower-left coordinate of each sub-grid w.r.t. the global grid.
        #     * z_num: np.ndarray[int64]
        #         (Q, D) sub-grid sizes.

        # Get kernel/lattice parameters.
        s = np.array([k.support() for k in kernel])
        alpha = np.array(z["start"])
        beta = np.array(z["stop"])
        N = np.array(z["num"])

        # Restrict clustering to support points which contribute to the lattice.
        active = pxm.filter_to_bbox(x, alpha, beta, s)  # (M,)
        active2global = np.flatnonzero(active)
        x = x[active]

        # Quick exit if no support points.
        _, D = x.shape
        if len(x) == 0:
            zeros = lambda sh: np.zeros(sh, dtype=np.int64)
            info = dict(
                x_idx=zeros(0),
                cl_bound=zeros(1),
                z_anchor=zeros((0, D)),
                z_num=zeros((0, D)),
            )
        else:
            # Group support points into clusters to match max window size.
            max_window_ratio = kwargs.get("max_window_ratio")
            bbox_dim = (2 * s) * max_window_ratio
            x_idx, cl_info = pxm.grid_cluster(x, bbox_dim)
            x_idx, cl_info = pxm.fuse_cluster(x, x_idx, cl_info, bbox_dim)

            # Split clusters to match max cluster size.
            N_max = kwargs.get("max_cluster_size")
            cl_info = pxm.bisect_cluster(cl_info, N_max)

            # Gather metadata per cluster
            Q = len(cl_info) - 1
            info = dict(
                x_idx=active2global[x_idx],  # indices w.r.t input `x`
                cl_bound=cl_info,  # (Q+1,)
                z_anchor=np.zeros((Q, D), dtype=np.int64),
                z_num=np.zeros((Q, D), dtype=np.int64),
            )
            cl_min, cl_max = pxm.group_minmax(x, x_idx, cl_info)  # (Q, D)
            for q in range(Q):
                # 1) Compute off-grid lattice boundaries after spreading.
                LL = cl_min[q] - s  # lower-left lattice coordinate
                UR = cl_max[q] + s  # upper-right lattice coordinate

                # 2) Get gridded equivalents.
                #
                # Note: using `ratio` safely handles the problematic (alpha==beta) case.
                ratio = N - 1.0
                ratio[N > 1] /= (beta - alpha)[N > 1]
                LL_idx = np.floor((LL - alpha) * ratio)
                UR_idx = np.ceil((UR - alpha) * ratio)

                # 3) Clip LL/UR to lattice boundaries.
                LL_idx = np.fmax(0, LL_idx).astype(int)
                UR_idx = np.fmin(UR_idx, N - 1).astype(int)

                # 4) Store metadata
                info["z_anchor"][q] = LL_idx
                info["z_num"][q] = UR_idx - LL_idx + 1
        return info

    def _lattice(
        self,
        xp: pxt.ArrayModule,
        dtype: pxt.DType,
        roi: tuple[slice] = None,
        flatten: bool = True,
    ) -> tuple[pxt.NDArray]:
        # Create sparse lattice mesh.
        #
        # Parameters
        # ----------
        # xp: ArrayModule
        #     Which array module to use to represent the mesh.
        # dtype: DType
        #     Precision of the arrays.
        # roi: tuple[slice]
        #     If provided, the lattice is restricted to a specific region-of-interest.
        #     The full lattice is returned by default.
        # flatten: bool
        #
        # Returns
        # -------
        # lattice: tuple[NDArray]
        #     * flatten=True : (D,) 1D lattice nodes.
        #     * flatten=False: (D,) sparse ND-meshgrid of lattice nodes.
        D = len(self._z["start"])
        if roi is None:
            roi = (slice(None),) * D

        lattice = [None] * D
        for d in range(D):
            alpha = self._z["start"][d]
            beta = self._z["stop"][d]
            N = self._z["num"][d]
            step = 0 if (N == 1) else (beta - alpha) / (N - 1)
            _roi = roi[d]
            lattice[d] = (alpha + xp.arange(N)[_roi] * step).astype(dtype)
        if not flatten:
            lattice = xp.meshgrid(
                *lattice,
                indexing="ij",
                sparse=True,
            )
        return lattice

    def _spread(self, x: np.ndarray, w: np.ndarray, q: int) -> tuple[int, np.ndarray]:
        # Spread (support, weight) pairs onto sub-lattice of specific cluster.
        #
        # Parameters
        # ----------
        # x: NDArray[float]
        #     (M, D) support points. [NUMPY, canonical order]
        # w: NDArray[float]
        #     (Ns, M) support weights. [NUMPY, canonical order]
        # q: int
        #     Cluster index.
        #
        # Returns
        # -------
        # q: int
        #     Cluster index.
        # v: NDArray[float]
        #     (Ns, S1,...,SD) q-th spreaded sub-lattice.
        dtype = w.dtype

        # Extract relevant fields from `_cl_info`
        a = self._cl_info["cl_bound"][q]
        b = self._cl_info["cl_bound"][q + 1]
        Mq, D = b - a, x.shape[1]
        z_anchor = self._cl_info["z_anchor"][q]  # (D,)
        z_num = S = self._cl_info["z_num"][q]  # (S1,...,SD)
        roi = [slice(n0, n0 + num) for (n0, num) in zip(z_anchor, z_num)]

        # Sub-sample (x, w)
        x = x[a:b]  # (Mq, D)
        w = w[:, a:b]  # (Ns, Mq)

        # Evaluate 1D kernel weights per support point
        lattice = self._lattice(np, dtype, roi)  # (S1,),...,(SD,)
        kernel = np.zeros((Mq, max(S), D), dtype=dtype)  # (Mq, S_max, D)
        for d, kern in enumerate(self._kernel):
            kernel[:, : S[d], d] = kern.apply(lattice[d] - x[:, [d]])

        # Spread onto sub-lattice
        v = np.zeros((*S, w.shape[0]), dtype=dtype)  # (S1,...,SD, Ns)
        self._nb.f_spread(
            w=np.require(w.T, requirements="C"),
            kernel=kernel,
            out=v,
        )

        axes = (-1, *range(self.codim_rank))
        return (q, v.transpose(axes))

    def _interpolate(self, x: np.ndarray, v: np.ndarray, q: int) -> tuple[int, np.ndarray]:
        # Interpolate (lattice, weight) pairs onto support points within cluster.
        #
        # Parameters
        # ----------
        # x: NDArray[float]
        #     (M, D) support points. [NUMPY, canonical order]
        # v: NDArray[float]
        #     (Ns, N1,...,ND) lattice weights. [NUMPY]
        # q: int
        #     Cluster index.
        #
        # Returns
        # -------
        # q: int
        #     Cluster index.
        # w: NDArray[float]
        #     (Ns, Mq) interpolated support weights of q-th cluster.
        dtype = v.dtype

        # Extract relevant fields from `_cl_info`
        a = self._cl_info["cl_bound"][q]
        b = self._cl_info["cl_bound"][q + 1]
        Mq, D = b - a, x.shape[1]
        z_anchor = self._cl_info["z_anchor"][q]  # (D,)
        z_num = S = self._cl_info["z_num"][q]  # (S1,...,SD)
        roi = [slice(n0, n0 + num) for (n0, num) in zip(z_anchor, z_num)]

        # Sub-sample (x, v)
        x = x[a:b]  # (Mq, D)
        v = v[:, *roi]  # (Ns, S1,...,SD)

        # Evaluate 1D kernel weights per support point
        lattice = self._lattice(np, dtype, roi)  # (S1,),...,(SD,)
        kernel = np.zeros((Mq, max(S), D), dtype=dtype, order="C")  # (Mq, S_max, D)
        for d, kern in enumerate(self._kernel):
            kernel[:, : S[d], d] = kern.apply(lattice[d] - x[:, [d]])

        # Interpolate onto support points
        w = np.zeros((Mq, v.shape[0]), dtype=dtype)  # (Mq, Ns)
        axes = (*range(1, self.codim_rank + 1), 0)
        self._nb.f_interpolate(
            v=np.require(v.transpose(axes), requirements="C"),
            kernel=kernel,
            out=w,
        )

        return (q, w.T)

    @staticmethod
    def _gen_code(dim_rank: int, dtype: pxt.DType) -> types.ModuleType:
        # Compile Numba kernels used in _[spread,interpolate]():
        # * void f_spread(w, kernel, out)
        # * void f_interpolate(v, kernel, out)
        #
        # The code is compiled only if unavailable in cache beforehand.
        #
        # Parameters
        # ----------
        # dim_rank: int
        #     Dimensionality D of the support points `x`.
        # dtype: DType
        #     Kernel FP precision.
        #
        # Returns
        # -------
        # jit_module: module
        #     A (loaded) python package containing methods (f_spread, f_interpolate).

        # Generate the code which should be compiled --------------------------
        width = pxrt.Width(dtype)
        _type = {
            pxrt.Width.SINGLE: "float32",
            pxrt.Width.DOUBLE: "float64",
        }[width]
        sig_spread = "".join(
            [
                "void(",
                f"{_type}[:,::1],",  # w C-(M, Ns)
                f"{_type}[:,:,::1],",  # kernel C-(M, S_max, D)
                f"{_type}[" + (":," * dim_rank) + "::1]",  # out C-(S1,...,SD, Ns)
                ")",
            ]
        )
        sig_interpolate = "".join(
            [
                "void(",
                f"{_type}[" + (":," * dim_rank) + "::1],",  # v C-(S1,...,SD, Ns)
                f"{_type}[:,:,::1],",  # kernel C-(M, S_max, D)
                f"{_type}[:,::1]",  # out C-(M, Ns)
                ")",
            ]
        )
        support = "".join(
            [
                "(",
                ",".join([f"ub[{d}] - lb[{d}]" for d in range(dim_rank)]),
                ",)",
            ]
        )
        idx = "".join(
            [
                "(",
                ",".join([f"lb[{d}] + offset[{d}]" for d in range(dim_rank)]),
                ",)",
            ]
        )

        template_file = plib.Path(__file__).parent / "_template.txt"
        with open(template_file, mode="r") as f:
            template = string.Template(f.read())
        code = template.substitute(
            signature_spread=sig_spread,
            signature_interpolate=sig_interpolate,
            support=support,
            idx=idx,
            dim_rank=dim_rank,
        )
        # ---------------------------------------------------------------------

        # Store/update cached version as needed.
        module_name = pxu.cache_module(code)
        pxcfg.cache_dir(load=True)  # make the Pyxu cache importable (if not already done)
        jit_module = pxu.import_module(module_name)
        return jit_module
