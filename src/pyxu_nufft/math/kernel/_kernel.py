import pathlib as plib
import string
import warnings

import numba as nb
import numba.types as nbt
import numpy as np
import pyxu.info.config as pxcfg
import pyxu.info.ptype as pxt
import pyxu.util as pxu
import scipy.integrate as spi
import scipy.special as sps

__all__ = [
    "Box",
    "Triangle",
    "KaiserBessel",
]


# This object is purposefully NOT an Operator()
class _FSSPulse:
    r"""
    Finite-Support Symmetric function :math:`f: \mathbb{R} \to \mathbb{R}`, element-wise.

    Only NUMPY arrays are currently supported.
    """

    def __init__(self, support: pxt.Real):
        r"""
        Parameters
        ----------
        support: Real
            Value :math:`s > 0` such that :math:`f(x) = 0, \; \forall |x| > s`.
        """
        self._support = float(support)
        assert self._support > 0

        # Compile vectorized variants of _nb_apply[F]().
        # NumbaWarning, which may be raised if the kernel is not cacheable, are masked.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", nb.NumbaWarning)
            vectorize = nb.vectorize(
                [
                    nbt.float32(nbt.float32),
                    nbt.float64(nbt.float64),
                ],
                target="cpu",
                cache=True,
            )
            self._apply = vectorize(self._nb_apply())
            self._applyF = vectorize(self._nb_applyF())

    def support(self) -> pxt.Real:
        r"""
        Returns
        -------
        s: Real
            Value :math:`s > 0` such that :math:`f(x) = 0, \; \forall |x| > s`.
        """
        return self._support

    def apply(self, arr: pxt.Real | pxt.NDArray) -> pxt.NDArray:
        r"""
        Evaluate :math:`f(x)`.
        """
        return self._apply(arr)

    def applyF(self, arr: pxt.Real | pxt.NDArray) -> pxt.NDArray:
        r"""
        Evaluate :math:`f^{\mathcal{F}}(v)`.

        The Fourier convention used is

        .. math::

           \mathcal{F}(f)(v) = \int f(x) e^{-j 2\pi v x} dx
        """
        return self._applyF(arr)

    def supportF(self, eps: pxt.Real) -> pxt.Real:
        r"""
        Parameters
        ----------
        eps: Real
            Energy cutoff threshold :math:`\epsilon \in [0, 0.1]`.

        Returns
        -------
        sF: Real
            Value such that

            .. math::

               \int_{-s^{\mathcal{F}}}^{s^{\mathcal{F}}} |f^{\mathcal{F}}(v)|^{2} dv
               \approx
               (1 - \epsilon) \|f\|_{2}^{2}
        """
        eps = float(eps)
        assert 0 <= eps <= 0.1
        tol = 1 - eps

        def energy(f: callable, a: float, b: float) -> float:
            # Estimate \int_{a}^{b} f^{2}(x) dx
            E, _ = spi.quad(lambda _: f(_) ** 2, a, b)
            return E

        if np.isclose(eps, 0):
            sF = np.inf
        else:
            s = self.support()
            E_tot = energy(self._nb_apply(), -s, s)

            # Coarse-grain search for a max bandwidth in v_step increments.
            tolerance_reached = False
            v_step = 1 / s  # slowest decay signal is sinc() -> steps at its zeros.
            v_max = 0
            while not tolerance_reached:
                v_max += v_step
                E = energy(self._nb_applyF(), -v_max, v_max)
                tolerance_reached = E >= tol * E_tot

            # Fine-grained search for a max bandwidth in [v_max - v_step, v_max] region.
            v_fine = np.linspace(v_max - v_step, v_max, 100)
            E = np.array([energy(self._nb_applyF(), -v, v) for v in v_fine])

            sF = v_fine[E >= tol * E_tot].min()
        return sF

    def argscale(self, scalar: pxt.Real) -> "_FSSPulse":
        scalar = float(scalar)
        assert scalar > 0

        cls, kwargs = self._meta()
        kwargs["support"] = kwargs["support"] / scalar
        return cls(**kwargs)

    # Internal Helpers --------------------------------------------------------
    def _meta(self):
        cls = self.__class__
        kwargs = dict(
            support=self.support(),
        )
        return (cls, kwargs)

    def _nb_apply(self) -> callable:
        # Low-level Numba-compiled kernel to evaluate f(x).
        # Must support signatures:
        #     float32(float32)
        #     float64(float64)
        #
        # Override in sub-classes to allow low-level computations.
        raise NotImplementedError

    def _nb_applyF(self) -> callable:
        # Low-level Numba-compiled kernel to evaluate f^{F}(v).
        # Must support signatures:
        #     float32(float32)
        #     float64(float64)
        #
        # Override in sub-classes to allow low-level computations.
        raise NotImplementedError

    def __repr__(self) -> str:
        klass = self.__class__.__name__
        support = self.support()
        return f"{klass}(support={support})"


class Box(_FSSPulse):
    r"""
    Box function.

    Notes
    -----
    * :math:`f(x) = 1_{[-s, s]}(x)`
    * :math:`f^{\mathcal{F}}(v) = 2s \; \text{sinc}(2s v)`
    """

    def __init__(self, support: pxt.Real):
        super().__init__(support=support)

    # Internal Helpers --------------------------------------------------------
    def _nb_apply(self) -> callable:
        template_file = plib.Path(__file__).parent / "_box_template.txt"
        subs = dict(
            support=self.support(),
        )
        return _load_from(template_file, subs, "apply")

    def _nb_applyF(self) -> callable:
        template_file = plib.Path(__file__).parent / "_box_template.txt"
        subs = dict(
            support=self.support(),
        )
        return _load_from(template_file, subs, "applyF")


class Triangle(_FSSPulse):
    r"""
    Triangle function.

    Notes
    -----
    * :math:`f(x) = (1 - |x/s|) 1_{[-s, s]}(x)`
    * :math:`f^{\mathcal{F}}(v) = s \text{sinc}^{2}(s v)`
    """

    def __init__(self, support: pxt.Real):
        super().__init__(support=support)

    # Internal Helpers --------------------------------------------------------
    def _nb_apply(self) -> callable:
        template_file = plib.Path(__file__).parent / "_triangle_template.txt"
        subs = dict(
            support=self.support(),
        )
        return _load_from(template_file, subs, "apply")

    def _nb_applyF(self) -> callable:
        template_file = plib.Path(__file__).parent / "_triangle_template.txt"
        subs = dict(
            support=self.support(),
        )
        return _load_from(template_file, subs, "applyF")


class KaiserBessel(_FSSPulse):
    r"""
    Kaiser-Bessel pulse.

    Notes
    -----
    * :math:`f(x) =
      \frac
      {I_{0}(\beta \sqrt{1 - (x/s)^{2}})}
      {I_{0}(\beta)}
      1_{[-s, s]}(x)`
    * :math:`f^{\mathcal{F}}(v) =
      \frac{2 s}{I_{0}(\beta)}
      \frac
      {\sinh\left[\sqrt{\beta^{2} - (2 \pi s v)^{2}} \right]}
      {\sqrt{\beta^{2} - (2 \pi s v)^{2}}}`
    """

    def __init__(self, beta: pxt.Real, support: pxt.Real):
        self._beta = float(beta)
        assert self._beta > 0
        super().__init__(support=support)

    def supportF(self, eps: pxt.Real) -> pxt.Real:
        if np.isclose(eps, 0):
            # use cut-off frequency: corresponds roughly to eps=1e-10
            sF = self._beta / (2 * np.pi * self.support())
        else:
            sF = super().supportF(eps)
        return sF

    # Internal Helpers --------------------------------------------------------
    def _meta(self):
        cls = self.__class__
        kwargs = dict(
            beta=self._beta,
            support=self.support(),
        )
        return (cls, kwargs)

    def _nb_apply(self) -> callable:
        template_file = plib.Path(__file__).parent / "_kb_template.txt"
        subs = dict(
            support=self.support(),
            beta=self._beta,
            i0_beta=sps.i0(self._beta),
        )
        return _load_from(template_file, subs, "apply")

    def _nb_applyF(self) -> callable:
        template_file = plib.Path(__file__).parent / "_kb_template.txt"
        subs = dict(
            support=self.support(),
            beta=self._beta,
            i0_beta=sps.i0(self._beta),
        )
        return _load_from(template_file, subs, "applyF")

    def __repr__(self) -> str:
        klass = self.__class__.__name__
        support = self.support()
        return f"{klass}(support={support}, beta={self._beta})"


def _load_from(path: plib.Path, subs: dict, func: str) -> callable:
    # * Load code as a string from `path`;
    # * Substitute ${}-terms using `subs`;
    # * Return `func` from the hot-loaded module.

    with open(path, mode="r") as f:
        template = string.Template(f.read())

    code = template.substitute(**subs)

    # Store/update cached version as needed.
    module_name = pxu.cache_module(code)
    pxcfg.cache_dir(load=True)  # make the Pyxu cache importable (if not already done)
    jit_module = pxu.import_module(module_name)

    return getattr(jit_module, func)
