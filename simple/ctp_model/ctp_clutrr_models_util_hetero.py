# -*- coding: utf-8 -*-

from torch import Tensor
from typing import Tuple, Optional

def uniform(a: Tensor,
            b: Tensor,
            c: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:

    """If first dimension of given Tensors different, repeat b to obtain the same first dimension.

        Args:
            a (Tensor)
            b (Tensor)
            c (Optional[Tensor]): we repeat elements in c the same amount as we repeat elements in b.

        Returns:
            Tuple[Tensor, Optional[Tensor]] of b and optionally c.

    """

    if a.shape[0] > b.shape[0]:
        m = a.shape[0] // b.shape[0]
        b = b.view(b.shape[0], 1, b.shape[1], b.shape[2]).repeat(1, m, 1, 1).view(-1, b.shape[1], b.shape[2])
        if c is not None:
            c = c.view(-1, 1).repeat(1, m).view(-1)
    return b, c
