import opt_einsum as oe
import pyxu.info.ptype as pxt

__all__ = [
    "hadamard_outer",
]


def hadamard_outer(
    x: pxt.NDArray,
    *args: list[pxt.NDArray],
) -> pxt.NDArray:
    r"""
    Compute Hadamard product of `x` with outer product of `args`:

    .. math::

       y = x \odot (A_{1} \otimes\cdots\otimes A_{D})

    Parameters
    ----------
    x: NDArray
        (..., N1,...,ND)
    args[k]: NDArray
        (Nk,)

    Returns
    -------
    y: NDArray
        (..., N1,...,ND)

    Note
    ----
    All inputs must share the same dtype precision.
    """
    D = len(args)
    assert all(A.ndim == 1 for A in args)
    sh = tuple(A.size for A in args)

    assert x.ndim >= D
    assert x.shape[-D:] == sh

    outer_args = [None] * (2 * D)
    for d in range(D):
        outer_args[2 * d] = args[d]
        outer_args[2 * d + 1] = (d,)

    x_ind = o_ind = (Ellipsis, *range(D))
    y = oe.contract(  # (..., N1,...,ND)
        *(x, x_ind),
        *outer_args,
        o_ind,
        use_blas=True,
        optimize="auto",
    )
    return y
